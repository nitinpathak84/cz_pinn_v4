# src/bcs_v4.py
import torch
from src.residuals_v4 import axisym_transient_component

def axis_symmetry_loss(model, rzt, r_eps, comp: int):
    _, Tr, _, _, _ = axisym_transient_component(model, rzt, r_eps=r_eps, comp=comp)
    return torch.mean(Tr**2)

def radiation_bc_const_z(model, rzt, r_eps, comp, k, eps, sigma, T_env):
    # const-z surface => normal derivative ~ dT/dz
    T, _, Tz, _, _ = axisym_transient_component(model, rzt, r_eps=r_eps, comp=comp)
    q_pred = -k * Tz
    q_rad = eps * sigma * (T**4 - (T_env**4))
    return torch.mean((q_pred - q_rad)**2)

def radiation_bc_const_r(model, rzt, r_eps, comp, k, eps, sigma, T_env):
    # const-r surface => normal derivative ~ dT/dr
    T, Tr, _, _, _ = axisym_transient_component(model, rzt, r_eps=r_eps, comp=comp)
    q_pred = -k * Tr
    q_rad = eps * sigma * (T**4 - (T_env**4))
    return torch.mean((q_pred - q_rad)**2)

def interface_T_continuity(model, rzt, r_eps):
    Tm, _, _, _, _ = axisym_transient_component(model, rzt, r_eps=r_eps, comp=0)
    Ts, _, _, _, _ = axisym_transient_component(model, rzt, r_eps=r_eps, comp=1)
    return torch.mean((Tm - Ts)**2)

def interface_flux_continuity(model, rzt, r_eps, k_m, k_s):
    # interface is const-z => enforce km*dTm/dz = ks*dTs/dz
    _, _, Tz_m, _, _ = axisym_transient_component(model, rzt, r_eps=r_eps, comp=0)
    _, _, Tz_s, _, _ = axisym_transient_component(model, rzt, r_eps=r_eps, comp=1)
    return torch.mean((k_m*Tz_m - k_s*Tz_s)**2)
