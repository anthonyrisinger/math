#!/usr/bin/env python3
"""
Dimensional Mathematics - Consolidated Mathematical Core
========================================================

Unified mathematical foundation consolidating all core mathematical operations
for the dimensional emergence framework. This module provides numerical
stability, type safety, and comprehensive mathematical functionality.

Architecture:
- constants.py: All mathematical constants and critical dimensions
- functions.py: Core mathematical functions (gamma, measures, morphic, phase)
- validation.py: Mathematical property validation and testing framework

Quick Import:
    from dimensional.mathematics import *

    # Core functions
    volume = v(4.0)         # 4D ball volume
    gamma_val = gamma_safe(2.5)          # Safe gamma function
    phi = PHI               # Golden ratio

    # Critical dimensions
    peak = get_critical_dimension('volume_peak')
"""

# Export everything from consolidated modules
from .constants import (
    BOX_ASPECT,
    CRITICAL_DIMENSIONS,
    GAMMA_OVERFLOW_THRESHOLD,
    LOG_SPACE_THRESHOLD,
    NUMERICAL_EPSILON,
    PHI,
    PI,
    PSI,
    SQRT_PI,
    VARPI,
    VIEW_AZIM,
    VIEW_ELEV,
    CriticalDimensionName,
    E,
    get_critical_dimension,
    is_near_critical,
    print_constants,
)
from .functions import (
    C,
    PhaseDynamicsEngine,
    R,
    S,
    V,
    abs_gamma,
    ball_volume,
    beta_function,
    c,
    complexity_measure,
    digamma_safe,
    discriminant,
    emergence_threshold,
    factorial_extension,
    find_all_peaks,
    find_peak,
    gamma_safe,
    gammaln_safe,
    golden_ratio_properties,
    is_critical_dimension,
    k_discriminant_zero,
    k_perfect_circle,
    morphic_polynomial_roots,
    phase_capacity,
    phase_coherence,
    phase_evolution_step,
    quick_gamma_analysis,
    quick_measure_analysis,
    quick_phase_analysis,
    r,
    ratio_measure,
    s,
    sap_rate,
    sphere_surface,
    total_phase_energy,
    # Add lowercase aliases explicitly
    v,
    # Visualization functions
    setup_3d_axis,
    create_3d_figure,
)
from .validation import (
    NumericalStabilityTester,
    PropertyValidator,
    cross_package_consistency_test,
    validate_mathematical_properties,
)

# Version info
__version__ = "2.0.0-consolidated"
__author__ = "Dimensional Mathematics Framework"
__description__ = "Consolidated mathematical core for dimensional emergence theory"
