#!/usr/bin/env python3
"""Unified gamma functions with optimal performance characteristics.

This module combines the best features of gamma.py and gamma_fast.py:
- Vectorized operations for arrays
- Optional caching for repeated calls
- All mathematical functions from standard version
- Batch operations from fast version
"""

import functools
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import digamma as scipy_digamma
from scipy.special import gamma as scipy_gamma
from scipy.special import gammaln as scipy_gammaln

from .measures import (
    ball_volume as v,
)
from .measures import (
    complexity_measure as c,
)
from .measures import (
    find_peak,
)
from .measures import (
    ratio_measure as r,
)
from .measures import (
    sphere_surface as s,
)


def gamma(z: ArrayLike) -> Union[float, NDArray[np.float64]]:
    """Gamma function with automatic vectorization.

    Optimized using scipy.special.gamma which handles:
    - Vectorized operations for arrays
    - Special cases (poles, negative values)
    - Numerical stability for large values
    """
    z = np.asarray(z, dtype=np.float64)
    scalar_input = (z.ndim == 0)

    # scipy.gamma is already optimally vectorized
    result = scipy_gamma(z)

    return float(result) if scalar_input else result


def gammaln(z: ArrayLike) -> Union[float, NDArray[np.float64]]:
    """Safe log-gamma function with automatic vectorization.

    More stable than log(gamma(z)) for large values.
    """
    z = np.asarray(z, dtype=np.float64)
    scalar_input = (z.ndim == 0)
    result = scipy_gammaln(z)
    return float(result) if scalar_input else result


def digamma(z: ArrayLike) -> Union[float, NDArray[np.float64]]:
    """Safe digamma function ψ(z) = d/dz log(Γ(z))."""
    z = np.asarray(z, dtype=np.float64)
    scalar_input = (z.ndim == 0)
    result = scipy_digamma(z)
    return float(result) if scalar_input else result


def factorial_extension(n: ArrayLike) -> Union[float, NDArray[np.float64]]:
    """Factorial extension for real numbers: n! = Γ(n+1).

    Handles:
    - Integer factorials: 5! = 120
    - Real extensions: 3.5! = Γ(4.5)
    - Negative values with appropriate infinities
    """
    n = np.asarray(n, dtype=np.float64)
    scalar_input = (n.ndim == 0)
    result = scipy_gamma(n + 1)

    # Handle negative integers properly
    neg_int_mask = (n < 0) & (np.abs(n - np.round(n)) < 1e-10)
    if np.any(neg_int_mask):
        result = np.where(neg_int_mask, np.inf, result)

    return float(result) if scalar_input else result


def beta_function(a: ArrayLike, b: ArrayLike) -> Union[float, NDArray[np.float64]]:
    """Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b).

    Uses log-gamma for numerical stability.
    """
    log_result = scipy_gammaln(a) + gammaln(b) - gammaln(a + b)
    return np.exp(log_result)


def batch_gamma_operations(z: ArrayLike) -> dict[str, NDArray[np.float64]]:
    """Compute multiple gamma-related functions efficiently in one pass.

    Returns dict with 'gamma', 'ln_gamma', 'digamma', and 'factorial' values.
    This is more efficient than calling each function separately when you
    need multiple gamma-related values for the same input.
    """
    z = np.asarray(z, dtype=np.float64)

    return {
        "gamma": scipy_gamma(z),
        "ln_gamma": scipy_gammaln(z),
        "digamma": scipy_digamma(z),
        "factorial": scipy_gamma(z + 1),  # n! = Γ(n+1)
    }


# Cache for expensive multi-value calculations
@functools.lru_cache(maxsize=1024)
def _cached_explore_single(d: float) -> dict[str, Any]:
    """Cached computation of dimensional measures for a single value.

    This is beneficial when the same dimension is queried repeatedly,
    such as in optimization loops or interactive exploration.
    """
    return {
        "dimension": d,
        "volume": v(d),
        "surface": s(d),
        "complexity": c(d),
        "ratio": r(d),
        "density": 1.0 / v(d) if v(d) != 0 else np.inf,
        "gamma": scipy_gamma(d) if d > 0 else None,
    }


def explore(d: Union[float, ArrayLike], use_cache: bool = True) -> Union[dict[str, Any], list]:
    """Comprehensive exploration of dimensional measures.

    Args:
        d: Dimension(s) to explore
        use_cache: Whether to use caching for repeated calls (default: True)

    Returns:
        Dictionary of measures for scalar input, list of dicts for array input
    """
    # Handle scalar input with optional caching
    if np.isscalar(d):
        if use_cache:
            return _cached_explore_single(float(d))
        else:
            d_float = float(d)
            return {
                "dimension": d_float,
                "volume": v(d_float),
                "surface": s(d_float),
                "complexity": c(d_float),
                "ratio": r(d_float),
                "density": 1.0 / v(d_float) if v(d_float) != 0 else np.inf,
                "gamma": scipy_gamma(d_float) if d_float > 0 else None,
            }

    # Handle array input (vectorized, no caching)
    d_array = np.asarray(d, dtype=np.float64)
    volumes = v(d_array)
    surfaces = s(d_array)
    complexities = c(d_array)
    ratios = r(d_array)

    results = []
    for i, dim in enumerate(d_array):
        results.append({
            "dimension": float(dim),
            "volume": float(volumes[i]),
            "surface": float(surfaces[i]),
            "complexity": float(complexities[i]),
            "ratio": float(ratios[i]),
            "density": 1.0 / float(volumes[i]) if volumes[i] != 0 else np.inf,
            "gamma": float(gamma(dim)) if dim > 0 else None,
        })

    return results


def clear_cache():
    """Clear the explore cache to free memory or force recalculation."""
    _cached_explore_single.cache_clear()


def get_cache_info():
    """Get cache statistics for performance monitoring."""
    return _cached_explore_single.cache_info()


# Peak finding functions (from standard version)
def v_peak() -> tuple[float, float]:
    """Find volume peak dimension and value."""
    return find_peak(v)


def s_peak() -> tuple[float, float]:
    """Find surface peak dimension and value."""
    return find_peak(s)


def c_peak() -> tuple[float, float]:
    """Find complexity peak dimension and value."""
    return find_peak(c)


def peaks() -> dict[str, tuple[float, float]]:
    """Find all critical peaks in dimensional measures."""
    return {
        "volume_peak": v_peak(),
        "surface_peak": s_peak(),
        "complexity_peak": c_peak(),
        # Also include without _peak suffix for compatibility
        "volume": v_peak(),
        "surface": s_peak(),
        "complexity": c_peak(),
    }


# Convergence diagnostics (from standard version)
def convergence_diagnostics(
    d_range: Optional[ArrayLike] = None,
    measure: str = "volume",
    threshold: float = 1e-10,
) -> dict[str, Any]:
    """Analyze convergence behavior of dimensional measures.

    Args:
        d_range: Dimensions to analyze (default: 1 to 100)
        measure: Which measure to analyze ('volume', 'surface', 'complexity')
        threshold: Value considered effectively zero

    Returns:
        Dictionary with convergence analysis results
    """
    if d_range is None:
        d_range = np.arange(1, 101, 1)
    else:
        d_range = np.asarray(d_range)

    # Select measure function
    measure_func = {"volume": v, "surface": s, "complexity": c}.get(measure, v)

    # Compute values
    values = measure_func(d_range)

    # Find where it effectively reaches zero
    zero_idx = np.where(values < threshold)[0]
    converge_dim = float(d_range[zero_idx[0]]) if len(zero_idx) > 0 else None

    # Compute rate of decay
    if len(values) > 1:
        log_values = np.log(values[values > 0])
        if len(log_values) > 1:
            dims_positive = d_range[values > 0]
            decay_rate = np.polyfit(dims_positive, log_values, 1)[0]
        else:
            decay_rate = None
    else:
        decay_rate = None

    return {
        "measure": measure,
        "threshold": threshold,
        "convergence_dimension": converge_dim,
        "decay_rate": decay_rate,
        "final_value": float(values[-1]),
        "max_value": float(np.max(values)),
        "max_dimension": float(d_range[np.argmax(values)]),
    }


# Quick analysis functions for common use cases
def quick_gamma_analysis(z: Union[float, ArrayLike]) -> dict[str, Any]:
    """Quick analysis of gamma function behavior at given points."""
    if np.isscalar(z):
        result = batch_gamma_operations(np.array([z]))
        values = {k: float(v[0]) for k, v in result.items()}
        # Add expected keys for compatibility
        return {
            'dimension': float(z),
            'gamma_value': values['gamma'],
            'log_gamma': values['ln_gamma'],
            'digamma': values['digamma'],
            'factorial': values['factorial'],
            # Keep original keys too
            **values
        }
    else:
        return batch_gamma_operations(z)


# Aliases for backward compatibility (will be removed after migration)
gamma_fast = gamma  # Fast version IS the standard now
gammaln_fast = gammaln
digamma_fast = digamma
factorial_extension_fast = factorial_extension
explore_fast = explore

# Performance utilities
def ρ(d: float) -> float:
    """Volume density (reciprocal volume)."""
    vol = v(d)
    return 1.0 / vol if vol != 0 else np.inf


# Lab and instant visualization functions (from standard version)
def lab(d: Optional[float] = None, interactive: bool = False) -> dict[str, Any]:
    """Interactive laboratory for exploring dimensional mathematics."""
    if d is None:
        d = 4.0  # Default dimension

    results = explore(d)
    peaks_data = peaks()
    convergence = convergence_diagnostics()

    return {
        "current": results,
        "peaks": peaks_data,
        "convergence": convergence,
        "interactive": interactive,
    }


def instant(d_range: Optional[ArrayLike] = None) -> dict[str, Any]:
    """Instant visualization data for dimensional measures."""
    if d_range is None:
        d_range = np.linspace(0.1, 20, 200)
    else:
        d_range = np.asarray(d_range)

    return {
        "dimensions": d_range.tolist(),
        "volume": v(d_range).tolist(),
        "surface": s(d_range).tolist(),
        "complexity": c(d_range).tolist(),
        "ratio": r(d_range).tolist(),
    }


def fractional_domain_validation(
    z_range: tuple[float, float] = (-5, 5),
    resolution: int = 100
) -> dict[str, Any]:
    """Validate gamma function behavior across fractional domain.

    Tests reflection formula, Stirling approximation, and domain coverage.
    """
    z = np.linspace(z_range[0], z_range[1], resolution)

    # Compute gamma values
    gamma_values = gamma(z)

    # Check finite ratio
    finite_mask = np.isfinite(gamma_values)
    finite_ratio = np.sum(finite_mask) / len(z)

    # Test reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
    reflection_errors = []
    for zi in z[finite_mask]:
        if zi != 0 and zi != 1:  # Avoid singularities
            lhs = gamma(zi) * gamma(1 - zi)
            rhs = np.pi / np.sin(np.pi * zi)
            if np.isfinite(lhs) and np.isfinite(rhs):
                reflection_errors.append(abs(lhs - rhs) / abs(rhs))

    mean_reflection_error = np.mean(reflection_errors) if reflection_errors else np.inf

    # Test Stirling approximation for large values
    stirling_errors = []
    for zi in z[z > 5]:
        if np.isfinite(gamma(zi)):
            exact = gamma(zi)
            stirling = np.sqrt(2 * np.pi * zi) * (zi / np.e) ** zi
            stirling_errors.append(abs(exact - stirling) / abs(exact))

    mean_stirling_error = np.mean(stirling_errors) if stirling_errors else np.inf

    return {
        'finite_ratio': finite_ratio,
        'mean_reflection_error': mean_reflection_error,
        'mean_stirling_error': mean_stirling_error,
        'reflection_accuracy': mean_reflection_error < 1e-6,
        'stirling_accuracy': mean_stirling_error < 0.01,
    }

