"""
Core Mathematical Functions
===========================

Consolidated module containing fundamental mathematical functions,
constants, and algorithms from mathematics, spectral, and phase modules.
"""

# Import everything from submodules
# Import errors from parent module
from ..errors import (
    ArraySizeError,
    CLIUsageError,
    DimensionalError,
    InvalidDimensionError,
    MathematicalError,
)
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
    E,
)
from .functions import (
    ball_volume,
    complexity_measure,
    digamma_safe,
    factorial_extension,
    find_all_peaks,
    find_peak,
    gamma_safe,
    gammaln_safe,
    ratio_measure,
    sphere_surface,
)

# Create aliases for backward compatibility
NumericalInstabilityError = DimensionalError
ConvergenceError = MathematicalError

# Phase dynamics
from .analysis import analytical_continuation, pole_structure
from .core import dimensional_cross_entropy, sap_rate
from .dynamics import PhaseDynamicsEngine

# Don't import validation here to avoid circular imports
# Users should import from dimensional.core.validation if needed

__all__ = [
    # Constants
    'PI', 'PHI', 'PSI', 'VARPI', 'E',
    'NUMERICAL_EPSILON', 'CRITICAL_DIMENSIONS',

    # Core functions
    'ball_volume', 'sphere_surface', 'complexity_measure', 'ratio_measure',
    'gamma_safe', 'gammaln_safe', 'digamma_safe',
    'find_peak', 'find_all_peaks',

    # Errors
    'DimensionalError', 'NumericalInstabilityError', 'ConvergenceError',
    'InvalidDimensionError',

    # Phase dynamics
    'PhaseDynamicsEngine', 'quick_phase_analysis',
    'phase_evolution_step', 'sap_rate', 'total_phase_energy',
    'phase_coherence', 'dimensional_time',
    'advanced_emergence_detection', 'emergence_threshold',

    # Spectral
    'DimensionalOperator', 'dimensional_spectral_density',
    'analyze_critical_point_spectrum', 'analyze_emergence_spectrum',
    'detect_dimensional_resonances', 'dimensional_wavelet_analysis',
    'fractal_harmonic_analysis', 'quick_spectral_analysis',
]
