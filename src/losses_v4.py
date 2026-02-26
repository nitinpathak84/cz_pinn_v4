# src/losses_v4.py
import torch
from src.residuals_v4 import axisym_transient_component
from src.bcs_v4 import (
    radiation_bc_const_z, radiation_bc_const_r,
    interface_T_continuity, interface_flux_continuity,
    axis_symmetry_loss
)

def pde_transient_loss(model, rzt, sdf, r_eps, rho, cp, k, comp):
    _, _, _, Tt, lap = axisym_transient_component(model, rzt, r_eps=r_eps, comp=comp)
    res = rho*cp*Tt - k*lap
    res = res * sdf  # sdf weighting
    return torch.mean(res**2)

def sensor_loss(model, rzt, y_meas, sensor_idx, bias_params, field_comp, r_eps):
    T, _, _, _, _ = axisym_transient_component(model, rzt, r_eps=r_eps, comp=field_comp)
    b = bias_params[sensor_idx].reshape(-1,1)
    return torch.mean((T + b - y_meas)**2)

def total_loss_v4(model, drift_net, bias_params, batches, sensor_batch, cfg):
    r_eps = cfg.physics.r_eps

    # PDE losses
    loss_pde_m = pde_transient_loss(
        model, batches["mi_rzt"], batches["mi_sdf"], r_eps,
        cfg.physics.rho_m, cfg.physics.cp_m, cfg.physics.k_m, comp=0
    )
    loss_pde_s = pde_transient_loss(
        model, batches["ci_rzt"], batches["ci_sdf"], r_eps,
        cfg.physics.rho_s, cfg.physics.cp_s, cfg.physics.k_s, comp=1
    )

    # Drift emissivity eps(t)
    if cfg.drift.enabled:
        t_for_drift = batches["mfb_rzt"][:, 2:3]
        d_eps_m, d_eps_s = drift_net(t_for_drift)
        eps_m = torch.clamp(cfg.physics.eps_m0 + d_eps_m, 0.0, 1.0)
        eps_s = torch.clamp(cfg.physics.eps_s0 + d_eps_s, 0.0, 1.0)
        drift_reg = torch.mean(d_eps_m**2) + torch.mean(d_eps_s**2)
    else:
        eps_m = cfg.physics.eps_m0
        eps_s = cfg.physics.eps_s0
        drift_reg = torch.tensor(0.0, device=batches["mi_rzt"].device)

    # Radiation BCs
    loss_rad_m = radiation_bc_const_z(
        model, batches["mfb_rzt"], r_eps, comp=0,
        k=cfg.physics.k_m, eps=eps_m, sigma=cfg.physics.sigma, T_env=cfg.physics.T_env
    )
    loss_rad_s = radiation_bc_const_r(
        model, batches["csb_rzt"], r_eps, comp=1,
        k=cfg.physics.k_s, eps=eps_s, sigma=cfg.physics.sigma, T_env=cfg.physics.T_env
    )

    # Interface
    loss_int_T = interface_T_continuity(model, batches["ifb_rzt"], r_eps)
    loss_int_flux = interface_flux_continuity(model, batches["ifb_rzt"], r_eps, cfg.physics.k_m, cfg.physics.k_s)

    # Axis symmetry (both comps)
    loss_axis = 0.5 * (
        axis_symmetry_loss(model, batches["ab_rzt"], r_eps, comp=0) +
        axis_symmetry_loss(model, batches["ab_rzt"], r_eps, comp=1)
    )

    # Sensors
    rs, zs, ts, ys, sidx = sensor_batch
    sens_rzt = torch.cat([rs, zs, ts], dim=1)
    field_comp = 1 if cfg.sensors.field.lower() == "ts" else 0
    loss_sens = sensor_loss(model, sens_rzt, ys, sidx, bias_params, field_comp, r_eps)

    # Bias regularization
    bias_reg = torch.mean(bias_params**2)

    # Total
    L = (
        cfg.training.w_pde_m * loss_pde_m
        + cfg.training.w_pde_s * loss_pde_s
        + cfg.training.w_rad_m * loss_rad_m
        + cfg.training.w_rad_s * loss_rad_s
        + cfg.training.w_int_T * loss_int_T
        + cfg.training.w_int_flux * loss_int_flux
        + cfg.training.w_axis * loss_axis
        + cfg.training.w_sensors * loss_sens
        + cfg.training.w_bias_reg * bias_reg
        + cfg.training.w_drift_reg * drift_reg
    )

    details = {
        "loss_total": L.detach(),
        "pde_m": loss_pde_m.detach(),
        "pde_s": loss_pde_s.detach(),
        "rad_m": loss_rad_m.detach(),
        "rad_s": loss_rad_s.detach(),
        "int_T": loss_int_T.detach(),
        "int_flux": loss_int_flux.detach(),
        "axis": loss_axis.detach(),
        "sens": loss_sens.detach(),
        "bias_reg": bias_reg.detach(),
        "drift_reg": drift_reg.detach(),
    }
    return L, details