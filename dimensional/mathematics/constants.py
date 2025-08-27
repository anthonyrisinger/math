#!/usr/bin/env python3
"""
Mathematical Constants - Consolidated
=====================================

All fundamental mathematical constants for the dimensional emergence framework.
Consolidated from core/constants.py with enhanced precision and documentation.
"""

from typing import Literal, Optional

import numpy as np
from scipy.special import gamma

# Universal mathematical constants
PI = np.pi
E = np.e
SQRT_PI = np.sqrt(np.pi)

# Golden ratio and related morphic constants
PHI = (1 + np.sqrt(5)) / 2  # φ = 1.618... (golden ratio)
PSI = 1 / PHI  # ψ = 0.618... (golden conjugate)

# Universal dimensional coupling constan
# ϖ = Γ(1/4)²/(4√(2π)) ≈ 1.311...
VARPI = gamma(0.25) ** 2 / (4 * np.sqrt(2 * PI))

# Critical dimensional boundaries and transition points
CRITICAL_DIMENSIONS = {
    # Phase boundaries
    "pi_boundary": PI,  # d = π ≈ 3.14159 (stability boundary)
    "tau_boundary": 2 * PI,  # d = 2π ≈ 6.283 (compression boundary)
    "e_natural": E,  # d = e ≈ 2.718 (natural scale)
    # Golden ratio scales
    "phi_golden": PHI,  # d = φ ≈ 1.618 (morphic scale)
    "psi_conjugate": PSI,  # d = ψ ≈ 0.618 (conjugate scale)
    "varpi_coupling": VARPI,  # d = ϖ ≈ 1.311 (coupling scale)
    # Measure peaks (computed values from dimensional_measures analysis)
    "volume_peak": 5.256389,  # Maximum of V_d = π^(d/2)/Γ(d/2+1)
    "surface_peak": 7.256389,  # Maximum of S_d = 2π^(d/2)/Γ(d/2)
    "complexity_peak": 6.0,  # Approximate maximum of C_d = V_d × S_d
    # Theoretical limits
    "leech_limit": 24,  # Maximum stable dimension (Leech lattice)
    "void_dimension": 0,  # The primordial void
    "unity_dimension": 1,  # First emergent dimension
}

# Standard 3D visualization constants
# Golden ratio viewing angle: degrees(φ - 1) ≈ 35.4°
VIEW_ELEV = np.degrees(PHI - 1)
VIEW_AZIM = -45.0
BOX_ASPECT = (1, 1, 1)

# Numerical precision and stability constants
NUMERICAL_EPSILON = 1e-12
GAMMA_OVERFLOW_THRESHOLD = 170
LOG_SPACE_THRESHOLD = 100

# Type-safe critical dimension names
CriticalDimensionName = Literal[
    "pi_boundary",
    "tau_boundary",
    "e_natural",
    "phi_golden",
    "psi_conjugate",
    "varpi_coupling",
    "volume_peak",
    "surface_peak",
    "complexity_peak",
    "leech_limit",
    "void_dimension",
    "unity_dimension",
]


def get_critical_dimension(name: CriticalDimensionName) -> float:
    """Get a critical dimension by name with type safety."""
    if name not in CRITICAL_DIMENSIONS:
        available = ", ".join(CRITICAL_DIMENSIONS.keys())
        raise ValueError(
            f"Unknown critical dimension '{name}'. Available: {available}"
        )
    return CRITICAL_DIMENSIONS[name]


def is_near_critical(
    d: float, tolerance: float = 1e-6
) -> Optional[CriticalDimensionName]:
    """Check if dimension d is near any critical value with type safety."""
    for name, value in CRITICAL_DIMENSIONS.items():
        if abs(d - value) < tolerance:
            return name  # type: ignore
    return None


def print_constants():
    """Print all fundamental constants with their values."""
    print("FUNDAMENTAL MATHEMATICAL CONSTANTS")
    print("=" * 50)
    print(f"π (pi):                 {PI:.10f}")
    print(f"e (euler):              {E:.10f}")
    print(f"φ (golden ratio):       {PHI:.10f}")
    print(f"ψ (golden conjugate):   {PSI:.10f}")
    print(f"ϖ (dimensional coupling): {VARPI:.10f}")
    print()
    print("CRITICAL DIMENSIONAL BOUNDARIES")
    print("=" * 50)
    for name, value in sorted(CRITICAL_DIMENSIONS.items()):
        print(f"{name:20}: {value:.6f}")


if __name__ == "__main__":
    print_constants()
