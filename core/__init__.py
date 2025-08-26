#!/usr/bin/env python3
"""
Core Mathematical Library
=========================

The essential mathematical foundation for the dimensional emergence framework.
This package contains the minimal, testable mathematical operations that
all other modules depend on.

Modules:
--------
constants  : Fundamental mathematical constants (φ, π, ϖ, critical dimensions)
gamma      : Safe gamma function family with numerical stability
measures   : Dimensional measures (ball volume, sphere surface, complexity)
phase      : Phase dynamics and sapping between dimensions
morphic    : Morphic polynomials and golden ratio operations
view       : Standard 3D visualization setup and camera angles

Quick Import:
-------------
from core import *
# Imports all essential functions and constants

Or specific imports:
from core.constants import PHI, PI, CRITICAL_DIMENSIONS
from core.measures import ball_volume, sphere_surface, complexity_measure
from core.phase import sap_rate, PhaseDynamicsEngine
from core.morphic import morphic_polynomial_roots, golden_ratio_properties
from core.view import setup_3d_axis, create_3d_figure
"""

# Core constants - always available
from .constants import (
    BOX_ASPECT,
    CRITICAL_DIMENSIONS,
    PHI,
    PI,
    PSI,
    VARPI,
    VIEW_AZIM,
    VIEW_ELEV,
    E,
    get_critical_dimension,
    is_near_critical,
)

# Gamma function family
from .gamma import (
    beta_function,
    digamma_safe,
    factorial_extension,
    gamma_ratio_safe,
    gamma_safe,
    gammaln_safe,
    polygamma_safe,
)

# Dimensional measures
from .measures import (
    ball_volume,
    complexity_measure,
    find_all_peaks,
    find_peak,
    integrated_measures,
    phase_capacity,
    ratio_measure,
    sphere_surface,
)

# Morphic mathematics
from .morphic import (
    MorphicAnalyzer,
    discriminant,
    golden_ratio_properties,
    k_discriminant_zero,
    k_perfect_circle,
    morphic_polynomial_roots,
    morphic_scaling_factor,
    stability_regions,
)

# Phase dynamics
from .phase import (
    PhaseDynamicsEngine,
    dimensional_time,
    emergence_threshold,
    phase_coherence,
    phase_evolution_step,
    sap_rate,
    total_phase_energy,
)

# 3D visualization
from .view import (
    View3DManager,
    add_coordinate_frame,
    add_integer_badge,
    create_3d_figure,
    golden_view_rotation,
    set_equal_aspect_3d,
    setup_3d_axis,
)

# Version info
__version__ = "1.0.0"
__author__ = "Dimensional Emergence Framework"

# Module metadata
__all__ = [
    # Constants
    "PI",
    "E",
    "PHI",
    "PSI",
    "VARPI",
    "CRITICAL_DIMENSIONS",
    "VIEW_ELEV",
    "VIEW_AZIM",
    "BOX_ASPECT",
    "get_critical_dimension",
    "is_near_critical",
    # Gamma functions
    "gamma_safe",
    "gammaln_safe",
    "digamma_safe",
    "polygamma_safe",
    "gamma_ratio_safe",
    "factorial_extension",
    "beta_function",
    # Dimensional measures
    "ball_volume",
    "sphere_surface",
    "complexity_measure",
    "ratio_measure",
    "phase_capacity",
    "find_peak",
    "find_all_peaks",
    "integrated_measures",
    # Phase dynamics
    "sap_rate",
    "phase_evolution_step",
    "emergence_threshold",
    "total_phase_energy",
    "phase_coherence",
    "dimensional_time",
    "PhaseDynamicsEngine",
    # Morphic mathematics
    "morphic_polynomial_roots",
    "discriminant",
    "k_perfect_circle",
    "k_discriminant_zero",
    "golden_ratio_properties",
    "morphic_scaling_factor",
    "stability_regions",
    "MorphicAnalyzer",
    # 3D visualization
    "setup_3d_axis",
    "create_3d_figure",
    "set_equal_aspect_3d",
    "golden_view_rotation",
    "add_coordinate_frame",
    "add_integer_badge",
    "View3DManager",
]


def print_library_info():
    """Print core library information and verification."""
    print("CORE MATHEMATICAL LIBRARY")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()

    print("FUNDAMENTAL CONSTANTS:")
    print(f"  π (pi):                 {PI:.10f}")
    print(f"  e (euler):              {E:.10f}")
    print(f"  φ (golden ratio):       {PHI:.10f}")
    print(f"  ψ (golden conjugate):   {PSI:.10f}")
    print(f"  ϖ (dimensional coupling): {VARPI:.10f}")
    print()

    print("CRITICAL DIMENSIONS:")
    for name, value in sorted(CRITICAL_DIMENSIONS.items()):
        print(f"  {name:20}: {value:.6f}")
    print()

    print("3D VISUALIZATION:")
    print(f"  View elevation:         {VIEW_ELEV:.2f}°")
    print(f"  View azimuth:           {VIEW_AZIM:.2f}°")
    print(f"  Box aspect:             {BOX_ASPECT}")
    print()

    # Test core functions
    print("FUNCTION VERIFICATION:")

    # Test gamma functions
    gamma_half = gamma_safe(0.5)
    print(f"  Γ(1/2) = √π:            {gamma_half:.6f} (exact: {(PI**0.5):.6f})")

    # Test dimensional measures
    vol_2d = ball_volume(2)
    surf_2d = sphere_surface(2)
    print(f"  V_2 (disk area):        {vol_2d:.6f} (exact: {PI:.6f})")
    print(f"  S_2 (circle length):    {surf_2d:.6f} (exact: {2 * PI:.6f})")

    # Test golden ratio properties
    props = golden_ratio_properties()
    phi_squared_check = props["phi_squared_equals_phi_plus_one"]
    print(f"  φ² = φ + 1:             {phi_squared_check}")

    # Test peaks
    peaks = find_all_peaks()
    vol_peak = peaks["volume_peak"][0]
    print(f"  Volume peak at d ≈:     {vol_peak:.3f}")

    print()
    print("AVAILABLE MODULES:")
    modules = ["constants", "gamma", "measures", "phase", "morphic", "view"]
    for module in modules:
        print(f"  core.{module}")
    print()
    print("Import with: from core import *")


def verify_mathematical_properties():
    """Verify fundamental mathematical properties hold."""
    import numpy as np

    results = {}
    tolerance = 1e-12

    # Golden ratio properties
    phi_check = abs(PHI**2 - (PHI + 1)) < tolerance
    psi_check = abs(PSI**2 - (1 - PSI)) < tolerance
    product_check = abs(PHI * PSI - 1) < tolerance

    results["golden_ratio"] = {
        "phi_squared_relation": phi_check,
        "psi_squared_relation": psi_check,
        "product_relation": product_check,
        "all_passed": phi_check and psi_check and product_check,
    }

    # Gamma function properties
    gamma_half_check = abs(gamma_safe(0.5) - np.sqrt(PI)) < tolerance
    gamma_one_check = abs(gamma_safe(1.0) - 1.0) < tolerance
    gamma_two_check = abs(gamma_safe(2.0) - 1.0) < tolerance

    results["gamma_functions"] = {
        "gamma_half_sqrt_pi": gamma_half_check,
        "gamma_one_equals_one": gamma_one_check,
        "gamma_two_equals_one": gamma_two_check,
        "all_passed": gamma_half_check and gamma_one_check and gamma_two_check,
    }

    # Dimensional measures
    vol_0_check = abs(ball_volume(0) - 1.0) < tolerance
    vol_2_check = abs(ball_volume(2) - PI) < tolerance
    surf_2_check = abs(sphere_surface(2) - 2 * PI) < tolerance

    results["dimensional_measures"] = {
        "vol_0_equals_one": vol_0_check,
        "vol_2_equals_pi": vol_2_check,
        "surf_2_equals_2pi": surf_2_check,
        "all_passed": vol_0_check and vol_2_check and surf_2_check,
    }

    # Overall verification
    all_modules_passed = all(
        module_results["all_passed"] for module_results in results.values()
    )
    results["overall"] = {"all_tests_passed": all_modules_passed}

    return results


if __name__ == "__main__":
    print_library_info()

    print("MATHEMATICAL VERIFICATION:")
    print("=" * 40)
    verification = verify_mathematical_properties()

    for module_name, tests in verification.items():
        if module_name == "overall":
            continue

        print(f"{module_name.replace('_', ' ').title()}:")
        for test_name, passed in tests.items():
            if test_name != "all_passed":
                status = "PASS" if passed else "FAIL"
                print(f"  {test_name.replace('_', ' ')}: {status}")
        print()

    overall_status = (
        "ALL TESTS PASSED"
        if verification["overall"]["all_tests_passed"]
        else "SOME TESTS FAILED"
    )
    print(f"Overall: {overall_status}")
