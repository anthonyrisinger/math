#!/usr/bin/env python3
# morphic_core.py
# Minimal, robust helpers shared by atlas scripts.

from __future__ import annotations
import numpy as np
from typing import Tuple, List

from clifford.g3c import e1, e2, e3, einf, eo
from clifford.tools.g3c import up, down

# ---------------------------
#   Polynomial family modes
# ---------------------------
# "shifted":  τ^3 - (2-k)τ - 1 = 0  (our primary in-session choice)
# "simple":   τ^3 - k τ - 1 = 0

def real_roots(k: float, mode: str = "shifted") -> np.ndarray:
    if mode == "shifted":
        coeffs = [1.0, 0.0, -(2.0 - k), -1.0]
    elif mode == "simple":
        coeffs = [1.0, 0.0, -k, -1.0]
    else:
        raise ValueError("mode must be 'shifted' or 'simple'")
    r = np.roots(coeffs)
    re = r.real[np.abs(r.imag) < 1e-9]
    return np.sort(re)[::-1]  # descending

def discriminant(k: float, mode: str = "shifted") -> float:
    if mode == "shifted":
        # p = -(2-k), q = -1 ⇒ Δ = -(4 p^3 + 27 q^2)
        p = -(2.0 - k)
        return - (4*p**3 + 27)
    else:
        # simple: p = -k, q = -1
        p = -k
        return - (4*p**3 + 27)

def k_perfect_circle(mode: str="shifted") -> float:
    # τ=1 must satisfy the polynomial
    if mode == "shifted":
        # 1 - (2-k)*1 - 1 = k-2 → k=2
        return 2.0
    else:
        # 1 - k - 1 = -k → k=0
        return 0.0

def k_discriminant_zero(mode: str="shifted") -> float:
    # Solve Δ(k)=0
    if mode == "shifted":
        # -(4(2-k)^3 + 27)=0 → (2-k)^3 = -27/4 → 2-k = -(27/4)^(1/3)
        return 2.0 - (27.0/4.0)**(1.0/3.0)
    else:
        # simple: Δ = -(4(-k)^3 + 27)=0 → 4(-k)^3 + 27 = 0 → k = (27/4)^(1/3)
        return (27.0/4.0)**(1.0/3.0)

# ---------------------------
#   CGA rotors & mapping
# ---------------------------
def make_rotor(tau: float):
    """
    Conformal rotor: translate by (tau,0,0), then dilate by tau (clamped).
    Clamp tau to avoid log-domain errors; this keeps visualization stable.
    """
    # translation (τ,0,0)
    T = 1 - 0.5 * (tau * e1) * einf
    # dilation (use positive scale; clamp min)
    s = float(np.clip(tau, 1e-12, None))
    D = 1 + 0.5 * np.log(s) * (einf - eo)
    return T * D

def sample_loop_xyz(tau: float, theta: np.ndarray) -> np.ndarray:
    """
    Map the unit circle (cosθ, sinθ, 0) through the CGA rotor for tau.
    Returns (N,3) array of Euclidean xyz.
    """
    R = make_rotor(tau)
    xyz = []
    for th in theta:
        x, y = np.cos(th), np.sin(th)
        mv = down(R * up(np.array([x,y,0.0])) * ~R)
        # Extract Euclidean coords
        if hasattr(mv, "value"):
            mv = mv.value[0]
        xyz.append([float(np.real(mv[0])), float(np.real(mv[1])), float(np.real(mv[2]))])
    return np.array(xyz)

def curvature_peak(xy: np.ndarray, dtheta: float) -> float:
    """
    Peak curvature |κ| for a closed polyline in XY.
    Guards against divide-by-zero and NaNs.
    """
    x = xy[:,0]; y = xy[:,1]
    with np.errstate(all="ignore"):
        dx, dy   = np.gradient(x, dtheta), np.gradient(y, dtheta)
        ddx, ddy = np.gradient(dx, dtheta), np.gradient(dy, dtheta)
        denom = (dx*dx + dy*dy)**1.5
        denom[denom == 0] = np.nan
        kappa = (dx*ddy - dy*ddx) / denom
        if np.isfinite(kappa).any():
            return float(np.nanmax(np.abs(kappa)))
        return 0.0
