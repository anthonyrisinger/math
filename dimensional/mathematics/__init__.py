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

Standardized Exception Hierarchy:
- DimensionalError: Base exception for all dimensional math errors
- NumericalInstabilityError: Computation instability detected
- ConvergenceError: Iterative algorithms failed to converge
- InvalidDimensionError: Invalid dimensional inputs

Quick Import:
    from dimensional.mathematics import *

    # Core functions
    volume = v(4.0)         # 4D ball volume
    gamma_val = gamma_safe(2.5)          # Safe gamma function
    phi = PHI               # Golden ratio

    # Critical dimensions
    peak = get_critical_dimension('volume_peak')
"""

# Standardized Exception Classes
class DimensionalError(Exception):
    """Base exception for all dimensional mathematics errors."""
    pass

class NumericalInstabilityError(DimensionalError):
    """Raised when numerical computations become unstable."""
    pass

class ConvergenceError(DimensionalError):
    """Raised when iterative algorithms fail to converge."""
    pass

class InvalidDimensionError(DimensionalError, ValueError):
    """Raised for invalid dimensional inputs."""
    pass

# Export everything from consolidated modules
from .constants import (
    CRITICAL_DIMENSIONS,
    GAMMA_OVERFLOW_THRESHOLD,
    LOG_SPACE_THRESHOLD,
    NUMERICAL_EPSILON,
    PHI,
    PI,
    PSI,
    VARPI,
    E,
)
from .functions import (
    PhaseDynamicsEngine,
    ball_volume,
    complexity_measure,
    create_3d_figure,
    find_peak,
    gamma_safe,
    gammaln_safe,
    morphic_polynomial_roots,
    phase_capacity,
    ratio_measure,
    sap_rate,
    sphere_surface,
)
from .validation import validate_mathematical_properties

# Exception classes available for import
__all__ = [
    # Exception classes
    'DimensionalError', 'NumericalInstabilityError', 
    'ConvergenceError', 'InvalidDimensionError',
    # Constants
    'CRITICAL_DIMENSIONS', 'GAMMA_OVERFLOW_THRESHOLD', 'LOG_SPACE_THRESHOLD',
    'NUMERICAL_EPSILON', 'PHI', 'PI', 'PSI', 'VARPI', 'E',
    # Functions
    'PhaseDynamicsEngine', 'ball_volume', 'complexity_measure', 'create_3d_figure',
    'find_peak', 'gamma_safe', 'gammaln_safe', 'morphic_polynomial_roots',
    'phase_capacity', 'ratio_measure', 'sap_rate', 'sphere_surface',
    # Validation
    'validate_mathematical_properties',
]

# Version info
__version__ = "2.0.0-consolidated"
__author__ = "Dimensional Mathematics Framework"
__description__ = (
    "Consolidated mathematical core for dimensional emergence theory"
)
