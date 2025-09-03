#!/usr/bin/env python3
"""Unified dimensional measures with optimal performance.

This module combines the best of measures.py and measures_fast.py:
- Validation for safety (optional)
- Vectorized operations for performance
- All mathematical functions preserved
- No redundant calculations
"""

import warnings
from typing import Any, Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

from .core import (
    NUMERICAL_EPSILON,
    PI,
)


def ball_volume(d: ArrayLike, validate: bool = True) -> Union[float, NDArray[np.float64]]:
    """Volume of unit d-dimensional ball: V_d = π^(d/2) / Γ(d/2 + 1).

    Args:
        d: Dimension(s) to calculate
        validate: Whether to validate input and issue warnings (default: True)

    Returns:
        Volume value(s)

    Note:
        Validation adds ~5% overhead but catches edge cases and provides warnings.
        Disable validation in tight loops with known-good input.
    """
    # Convert to numpy array for vectorization
    d = np.asarray(d, dtype=np.float64)
    scalar_input = (d.ndim == 0)

    # Optional validation (can be skipped for performance)
    if validate:
        _validate_dimension(d, "ball_volume")

    # Vectorized computation using scipy's gammaln (optimal performance)
    log_vol = (d / 2) * np.log(PI) - gammaln(d / 2 + 1)
    result = np.exp(log_vol)

    # Handle d=0 special case
    result = np.where(np.abs(d) < NUMERICAL_EPSILON, 1.0, result)

    # Return scalar if input was scalar
    return float(result) if scalar_input else result


def sphere_surface(d: ArrayLike, validate: bool = True) -> Union[float, NDArray[np.float64]]:
    """Surface area of unit (d-1)-sphere: S_d = 2π^(d/2) / Γ(d/2).

    Args:
        d: Dimension(s) to calculate
        validate: Whether to validate input and issue warnings (default: True)

    Returns:
        Surface area value(s)
    """
    # Convert to numpy array for vectorization
    d = np.asarray(d, dtype=np.float64)
    scalar_input = (d.ndim == 0)

    # Optional validation
    if validate:
        _validate_dimension(d, "sphere_surface")

    # Vectorized computation
    log_surf = np.log(2) + (d / 2) * np.log(PI) - gammaln(d / 2)
    result = np.exp(log_surf)

    # Handle special cases
    result = np.where(np.abs(d) < NUMERICAL_EPSILON, 2.0, result)
    result = np.where(np.abs(d - 1) < NUMERICAL_EPSILON, 2.0, result)

    # Return scalar if input was scalar
    return float(result) if scalar_input else result


def complexity_measure(d: ArrayLike, validate: bool = True) -> Union[float, NDArray[np.float64]]:
    """Combined complexity measure: C(d) = V(d) × S(d).

    Optimized to compute shared gamma values only once.

    Args:
        d: Dimension(s) to calculate
        validate: Whether to validate input (default: True)

    Returns:
        Complexity measure value(s)
    """
    # Convert to numpy array
    d = np.asarray(d, dtype=np.float64)
    scalar_input = (d.ndim == 0)

    # Optional validation
    if validate:
        _validate_dimension(d, "complexity_measure")

    # Optimized computation - shared gamma calculations
    half_d = d / 2
    log_gamma_half = gammaln(half_d)
    log_gamma_half_plus_1 = gammaln(half_d + 1)

    # C(d) = 2π^d / (Γ(d/2) * Γ(d/2 + 1))
    log_complexity = np.log(2) + d * np.log(PI) - log_gamma_half - log_gamma_half_plus_1
    result = np.exp(log_complexity)

    # Handle edge cases
    result = np.where(np.abs(d) < NUMERICAL_EPSILON, 2.0, result)

    return float(result) if scalar_input else result


def ratio_measure(d: ArrayLike, validate: bool = True) -> Union[float, NDArray[np.float64]]:
    """Ratio measure R(d) = S(d) / V(d).

    Optimized to avoid redundant calculations.

    Args:
        d: Dimension(s) to calculate
        validate: Whether to validate input (default: True)

    Returns:
        Ratio measure value(s)
    """
    d = np.asarray(d, dtype=np.float64)
    scalar_input = (d.ndim == 0)

    if validate:
        _validate_dimension(d, "ratio_measure")

    # Optimized: compute ratio directly without computing V and S separately
    # R(d) = S(d)/V(d) = 2*Γ(d/2 + 1) / Γ(d/2)
    # Using log for stability
    log_ratio = np.log(2) + gammaln(d/2 + 1) - gammaln(d/2)
    result = np.exp(log_ratio)

    # Handle edge cases
    result = np.where(np.abs(d) < NUMERICAL_EPSILON, 2.0, result)

    return float(result) if scalar_input else result


def batch_measures(d: ArrayLike, validate: bool = True) -> dict[str, NDArray[np.float64]]:
    """Compute all measures efficiently in a single pass.

    This is more efficient than calling each function separately because
    it reuses the gamma calculations.

    Args:
        d: Dimension(s) to calculate
        validate: Whether to validate input (default: True)

    Returns:
        Dictionary with 'volume', 'surface', 'complexity', and 'ratio' arrays
    """
    d = np.asarray(d, dtype=np.float64)

    if validate:
        _validate_dimension(d, "batch_measures")

    # Compute shared gamma values once
    half_d = d / 2
    log_gamma_half = gammaln(half_d)
    log_gamma_half_plus_1 = gammaln(half_d + 1)
    log_pi_half_d = half_d * np.log(PI)

    # Volume: π^(d/2) / Γ(d/2 + 1)
    log_vol = log_pi_half_d - log_gamma_half_plus_1
    volume = np.exp(log_vol)

    # Surface: 2π^(d/2) / Γ(d/2)
    log_surf = np.log(2) + log_pi_half_d - log_gamma_half
    surface = np.exp(log_surf)

    # Complexity: V * S (computed directly for efficiency)
    log_complexity = np.log(2) + d * np.log(PI) - log_gamma_half - log_gamma_half_plus_1
    complexity = np.exp(log_complexity)

    # Ratio: S / V
    ratio = surface / volume

    # Handle edge cases
    volume = np.where(np.abs(d) < NUMERICAL_EPSILON, 1.0, volume)
    surface = np.where(np.abs(d) < NUMERICAL_EPSILON, 2.0, surface)
    surface = np.where(np.abs(d - 1) < NUMERICAL_EPSILON, 2.0, surface)
    complexity = np.where(np.abs(d) < NUMERICAL_EPSILON, 2.0, complexity)
    ratio = np.where(np.abs(d) < NUMERICAL_EPSILON, 2.0, ratio)

    return {
        "volume": volume,
        "surface": surface,
        "complexity": complexity,
        "ratio": ratio,
    }


def _validate_dimension(d: ArrayLike, function_name: str = "measure") -> None:
    """Validate dimensional input and issue warnings for edge cases.

    This is separated so it can be skipped in performance-critical code.
    """
    d_array = np.asarray(d)

    # Check for negative dimensions
    if np.any(d_array < 0):
        negative_values = d_array[d_array < 0]
        if len(negative_values) == 1:
            warnings.warn(
                f"Negative dimension d={negative_values[0]:.3f} in {function_name}(). "
                f"Returning mathematical extension value. "
                f"Physical dimensions are typically d ≥ 0.",
                UserWarning,
                stacklevel=3,
            )
        else:
            warnings.warn(
                f"Negative dimensions detected in {function_name}() "
                f"(min: {np.min(negative_values):.3f}). "
                f"Returning mathematical extension values. "
                f"Physical dimensions are typically d ≥ 0.",
                UserWarning,
                stacklevel=3,
            )

    # Check for large dimensions
    if np.any(d_array > 100):
        large_values = d_array[d_array > 100]
        if len(large_values) == 1:
            warnings.warn(
                f"Large dimension d={large_values[0]:.1f} in {function_name}() "
                f"may underflow to zero due to gamma function behavior.",
                UserWarning,
                stacklevel=3,
            )
        else:
            warnings.warn(
                f"Large dimensions detected in {function_name}() "
                f"(max: {np.max(large_values):.1f}) "
                f"may underflow to zero due to gamma function behavior.",
                UserWarning,
                stacklevel=3,
            )


# Analysis functions
def find_peak(
    measure_func: Callable[[ArrayLike], Union[float, NDArray]],
    d_range: Optional[tuple[float, float]] = None,
    resolution: int = 1000,
) -> tuple[float, float]:
    """Find the peak (maximum) of a measure function.

    Args:
        measure_func: Function to analyze (e.g., ball_volume)
        d_range: Range to search (default: 0.1 to 20)
        resolution: Number of points to sample

    Returns:
        Tuple of (peak_dimension, peak_value)
    """
    if d_range is None:
        d_range = (0.1, 20.0)

    d = np.linspace(d_range[0], d_range[1], resolution)
    values = measure_func(d, validate=False)  # Skip validation for performance

    max_idx = np.argmax(values)
    return float(d[max_idx]), float(values[max_idx])


def convergence_analysis(
    d_start: float = 1.0,
    d_end: float = 200.0,
    threshold: float = 1e-100,
) -> dict[str, Any]:
    """Analyze convergence behavior of all measures.

    Returns dictionary with convergence dimensions for each measure.
    """
    d_range = np.linspace(d_start, d_end, 1000)

    # Compute all measures
    measures = batch_measures(d_range, validate=False)

    results = {}
    for name, values in measures.items():
        # Find where measure drops below threshold
        below_threshold = values < threshold
        if np.any(below_threshold):
            converge_idx = np.argmax(below_threshold)
            converge_dim = float(d_range[converge_idx])
        else:
            converge_dim = None

        results[name] = {
            "converge_dimension": converge_dim,
            "final_value": float(values[-1]),
            "max_value": float(np.max(values)),
            "max_dimension": float(d_range[np.argmax(values)]),
        }

    return results


# Aliases for backward compatibility (will be removed after migration)
def ball_volume_fast(d):
    return ball_volume(d, validate=False)
def sphere_surface_fast(d):
    return sphere_surface(d, validate=False)
def complexity_measure_fast(d):
    return complexity_measure(d, validate=False)

# Re-export common names for convenience
v = ball_volume
s = sphere_surface
c = complexity_measure
r = ratio_measure

# Uppercase aliases for backward compatibility
V = ball_volume
S = sphere_surface
C = complexity_measure
R = ratio_measure

# Additional exports for compatibility
def find_all_peaks() -> dict:
    """Find peaks for all measures."""
    return {
        "volume": find_peak(ball_volume),
        "surface": find_peak(sphere_surface),
        "complexity": find_peak(complexity_measure),
    }
