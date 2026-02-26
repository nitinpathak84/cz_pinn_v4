# src/train_v4.py
import os
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Adam, lr_scheduler

from physicsnemo.distributed import DistributedManager

from src.geometry import CzGeomParams, build_geometries
from src.sampling import make_volume_sampler, make_surface_sampler
from src.model_v4 import build_model
from src.drift_v4 import DriftNet
from src.sensors_v4 import SensorDataset
from src.losses_v4 import total_loss_v4

def sample_time(cfg, n, device):
    t0, t1 = cfg.time.t_min, cfg.time.t_max
    return (torch.rand(n, 1, device=device) * (t1 - t0) + t0)

@hydra.main(version_base="1.3", config_path="../conf", config_name="config_v4.yaml")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)

    os.makedirs(cfg.run.out_dir, exist_ok=True)

    # Geometry
    p = CzGeomParams(**cfg.geometry)
    geoms = build_geometries(p)

    # Model + drift
    model = build_model(cfg.model.num_layers, cfg.model.layer_size, device=device)

    drift_net = DriftNet(
        hidden=cfg.drift.hidden,
        layers=cfg.drift.layers,
        max_delta_eps=cfg.drift.max_delta_eps
    ).to(device)

    # Sensors + bias parameters
    sensors = SensorDataset(
        meta_path=cfg.sensors.meta_path,
        ts_path=cfg.sensors.ts_path,
        id_col=cfg.sensors.id_col,
        time_col=cfg.sensors.time_col,
        value_col=cfg.sensors.value_col,
        device=device
    )
    bias_params = nn.Parameter(torch.zeros(sensors.num_sensors(), device=device))

    # Optimizer
    params = list(model.parameters()) + ([bias_params] if bias_params is not None else [])
    if cfg.drift.enabled:
        params += list(drift_net.parameters())

    optimizer = Adam(params, lr=cfg.training.lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cfg.training.lr_decay**step)

    # Samplers
    melt_int = make_volume_sampler(geoms.melt, cfg.training.n_int_melt, device=device)
    crys_int = make_volume_sampler(geoms.crystal, cfg.training.n_int_crystal, device=device)

    melt_free_bc = make_surface_sampler(geoms.melt_free_band, cfg.training.n_bc_melt_free, device=device)
    crys_side_bc = make_surface_sampler(geoms.crystal_side_band, cfg.training.n_bc_crystal_side, device=device)
    iface_bc = make_surface_sampler(geoms.interface_band, cfg.training.n_bc_interface, device=device)
    axis_bc = make_surface_sampler(geoms.axis_band, cfg.training.n_bc_axis, device=device)

    for step in range(cfg.run.steps):
        mi_raw = next(iter(melt_int))[0]
        ci_raw = next(iter(crys_int))[0]
        mfb_raw = next(iter(melt_free_bc))[0]
        csb_raw = next(iter(crys_side_bc))[0]
        ifb_raw = next(iter(iface_bc))[0]
        ab_raw  = next(iter(axis_bc))[0]

        # reshape
        mi = {k: v.reshape(-1,1) for k,v in mi_raw.items()}
        ci = {k: v.reshape(-1,1) for k,v in ci_raw.items()}
        mfb = {k: v.reshape(-1,1) for k,v in mfb_raw.items()}
        csb = {k: v.reshape(-1,1) for k,v in csb_raw.items()}
        ifb = {k: v.reshape(-1,1) for k,v in ifb_raw.items()}
        ab  = {k: v.reshape(-1,1) for k,v in ab_raw.items()}

        # attach time
        mi_t  = sample_time(cfg, mi["x"].shape[0], device)
        ci_t  = sample_time(cfg, ci["x"].shape[0], device)
        mfb_t = sample_time(cfg, mfb["x"].shape[0], device)
        csb_t = sample_time(cfg, csb["x"].shape[0], device)
        ifb_t = sample_time(cfg, ifb["x"].shape[0], device)
        ab_t  = sample_time(cfg, ab["x"].shape[0], device)

        batches = {
            "mi_rzt": torch.cat([mi["x"], mi["y"], mi_t], dim=1),
            "mi_sdf": mi["sdf"],
            "ci_rzt": torch.cat([ci["x"], ci["y"], ci_t], dim=1),
            "ci_sdf": ci["sdf"],
            "mfb_rzt": torch.cat([mfb["x"], mfb["y"], mfb_t], dim=1),
            "csb_rzt": torch.cat([csb["x"], csb["y"], csb_t], dim=1),
            "ifb_rzt": torch.cat([ifb["x"], ifb["y"], ifb_t], dim=1),
            "ab_rzt": torch.cat([ab["x"], ab["y"], ab_t], dim=1),
        }

        sensor_batch = sensors.sample_batch(cfg.training.n_sensors_per_step, cfg.training.n_time_per_sensor)

        optimizer.zero_grad()
        loss, d = total_loss_v4(model, drift_net, bias_params, batches, sensor_batch, cfg)
        loss.backward()

        # gradient clipping
        if cfg.training.max_grad_norm is not None and cfg.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.training.max_grad_norm)

        optimizer.step()
        scheduler.step()

        if step % cfg.run.plot_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={step} loss={d['loss_total'].item():.4e} "
                f"pde_m={d['pde_m'].item():.2e} pde_s={d['pde_s'].item():.2e} "
                f"rad_m={d['rad_m'].item():.2e} rad_s={d['rad_s'].item():.2e} "
                f"intT={d['int_T'].item():.2e} intF={d['int_flux'].item():.2e} "
                f"axis={d['axis'].item():.2e} sens={d['sens'].item():.2e} "
                f"lr={lr:.3e}"
            )

if __name__ == "__main__":
    main()