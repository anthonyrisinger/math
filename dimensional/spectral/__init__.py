"""
Spectral Module
===============

Refactored from monolithic spectral.py (1,103 lines) into focused submodules.
"""

from .functions import (
    DimensionalOperator,
    analyze_critical_point_spectrum,
    analyze_emergence_spectrum,
    detect_dimensional_resonances,
    dimensional_spectral_density,
    dimensional_wavelet_analysis,
    fractal_harmonic_analysis,
    quick_spectral_analysis,
)

__all__ = [
    'DimensionalOperator',
    'analyze_critical_point_spectrum',
    'analyze_emergence_spectrum',
    'detect_dimensional_resonances',
    'dimensional_spectral_density',
    'dimensional_wavelet_analysis',
    'fractal_harmonic_analysis',
    'quick_spectral_analysis',
]
