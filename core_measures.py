#!/usr/bin/env python3
"""
Core Dimensional Measures
=========================

The fundamental mathematical framework for dimensional emergence.
Self-contained module with all essential geometric measures and constants.
"""

import numpy as np
from scipy.special import gamma, gammaln

# Fundamental constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio: φ = 1.618...
PSI = 1 / PHI               # Golden conjugate: ψ = 0.618...
VARPI = gamma(0.25)**2 / (2 * np.sqrt(2 * np.pi))  # ϖ = 1.311...
PI = np.pi
E = np.e

# Standard 3D visualization parameters
VIEW_ELEV = np.degrees(PHI - 1)  # ≈ 36.87°
VIEW_AZIM = -45
BOX_ASPECT = (1, 1, 1)

class DimensionalMeasures:
    """Core geometric measures for any real dimension d."""

    @staticmethod
    def ball_volume(d):
        """Volume of unit d-ball: V_d = π^(d/2) / Γ(d/2 + 1)"""
        if abs(d) < 1e-10:
            return 1.0  # V_0 = 1 (point)

        # Use log space for numerical stability
        if d > 170:
            log_vol = (d/2) * np.log(PI) - gammaln(d/2 + 1)
            return np.exp(np.real(log_vol))

        return PI**(d/2) / gamma(d/2 + 1)

    @staticmethod
    def sphere_surface(d):
        """Surface area of unit (d-1)-sphere: S_d = 2π^(d/2) / Γ(d/2)"""
        if abs(d) < 1e-10:
            return 2.0  # S^{-1} convention
        if abs(d - 1) < 1e-10:
            return 2.0  # S^0 = two points

        if d > 170:
            log_surf = np.log(2) + (d/2) * np.log(PI) - gammaln(d/2)
            return np.exp(np.real(log_surf))

        return 2 * PI**(d/2) / gamma(d/2)

    @staticmethod
    def complexity_measure(d):
        """V×S complexity: C_d = V_d × S_d (peaks at d≈6)"""
        return DimensionalMeasures.ball_volume(d) * DimensionalMeasures.sphere_surface(d)

    @staticmethod
    def ratio_measure(d):
        """Surface/Volume ratio: R_d = S_d / V_d"""
        v = DimensionalMeasures.ball_volume(d)
        s = DimensionalMeasures.sphere_surface(d)
        return s / max(v, 1e-10)

    @staticmethod
    def find_peak(measure_func, d_min=0.1, d_max=15, num_points=5000):
        """Find the peak of a measure function."""
        d_range = np.linspace(d_min, d_max, num_points)
        values = np.array([measure_func(d) for d in d_range])
        peak_idx = np.argmax(values)
        return d_range[peak_idx], values[peak_idx]

    @staticmethod
    def critical_dimensions():
        """Return dictionary of critical dimensional values."""
        vol_peak_d, vol_peak_val = DimensionalMeasures.find_peak(DimensionalMeasures.ball_volume)
        surf_peak_d, surf_peak_val = DimensionalMeasures.find_peak(DimensionalMeasures.sphere_surface)
        comp_peak_d, comp_peak_val = DimensionalMeasures.find_peak(DimensionalMeasures.complexity_measure)

        return {
            'volume_peak': (vol_peak_d, vol_peak_val),
            'surface_peak': (surf_peak_d, surf_peak_val),
            'complexity_peak': (comp_peak_d, comp_peak_val),
            'pi_boundary': PI,
            'tau_boundary': 2 * PI,
            'e_natural': E,
            'phi_golden': PHI,
            'psi_conjugate': PSI,
            'varpi_gamma': VARPI
        }

def setup_3d_axis(ax, title="", xlim=None, ylim=None, zlim=None):
    """Standard 3D axis setup with orthographic projection and golden viewing angle."""
    ax.set_proj_type('ortho')
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_box_aspect(BOX_ASPECT)

    if title:
        ax.set_title(title, fontsize=12, pad=15)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)

    # Clean appearance
    ax.grid(True, alpha=0.3)
    return ax

def print_critical_info():
    """Print the critical dimensional information."""
    measures = DimensionalMeasures()
    crits = measures.critical_dimensions()

    print("CRITICAL DIMENSIONAL STRUCTURE")
    print("=" * 50)
    print(f"Volume peak:      d = {crits['volume_peak'][0]:.3f}")
    print(f"Surface peak:     d = {crits['surface_peak'][0]:.3f}")
    print(f"Complexity peak:  d = {crits['complexity_peak'][0]:.3f}")
    print(f"π boundary:       d = {crits['pi_boundary']:.3f}")
    print(f"2π boundary:      d = {crits['tau_boundary']:.3f}")
    print(f"e natural:        d = {crits['e_natural']:.3f}")
    print(f"φ golden:         d = {crits['phi_golden']:.3f}")
    print("=" * 50)

if __name__ == "__main__":
    print_critical_info()