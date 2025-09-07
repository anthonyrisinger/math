"""
Dimensional measures compatibility module for tests.
"""

import warnings

import numpy as np

from .core import PI, E
from .core import c as complexity_measure_impl
from .core import r as ratio_measure_impl
from .core import s as sphere_surface_impl

# Import from core
from .core import v as ball_volume_impl

# Add NUMERICAL_EPSILON for tests
NUMERICAL_EPSILON = 1e-15


def ball_volume(d, validate=True):
    """Volume of unit d-dimensional ball."""
    if validate:
        _validate_dimension(d, "ball_volume")
    return ball_volume_impl(d)


def sphere_surface(d, validate=True):
    """Surface area of unit (d-1)-sphere."""
    if validate:
        _validate_dimension(d, "sphere_surface")
    return sphere_surface_impl(d)


def complexity_measure(d, validate=True):
    """Combined complexity measure C(d) = V(d) × S(d)."""
    if validate:
        _validate_dimension(d, "complexity_measure")
    return complexity_measure_impl(d)


def ratio_measure(d, validate=True):
    """Ratio measure R(d) = S(d) / V(d)."""
    if validate:
        _validate_dimension(d, "ratio_measure")
    return ratio_measure_impl(d)


def batch_measures(d, validate=True):
    """Compute all measures efficiently."""
    d = np.asarray(d, dtype=np.float64)
    if validate:
        _validate_dimension(d, "batch_measures")

    return {
        "volume": ball_volume_impl(d),
        "surface": sphere_surface_impl(d),
        "complexity": complexity_measure_impl(d),
        "ratio": ratio_measure_impl(d),
    }


def _validate_dimension(d, function_name="measure"):
    """Validate dimensional input."""
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


def find_peak(measure_func, d_range=None, resolution=1000):
    """Find the peak of a measure function."""
    from scipy.optimize import minimize_scalar

    if measure_func == ball_volume or measure_func.__name__ == 'ball_volume':
        bounds = (1, 10)
    elif measure_func == sphere_surface or measure_func.__name__ == 'sphere_surface':
        bounds = (1, 12)
    elif measure_func == complexity_measure or measure_func.__name__ == 'complexity_measure':
        bounds = (1, 15)
    else:
        bounds = (0.1, 20) if d_range is None else d_range

    # Handle validate parameter if present
    def wrapper(d):
        if 'validate' in measure_func.__code__.co_varnames:
            return -measure_func(d, validate=False)
        return -measure_func(d)

    result = minimize_scalar(wrapper, bounds=bounds, method='bounded')
    return (result.x, -result.fun)


def convergence_analysis(d_start=1.0, d_end=200.0, threshold=1e-100):
    """Analyze convergence behavior of all measures."""
    d_range = np.linspace(d_start, d_end, 1000)
    measures = batch_measures(d_range, validate=False)

    results = {}
    for name, values in measures.items():
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


def find_all_peaks():
    """Find peaks for all measures."""
    return {
        "volume": find_peak(ball_volume),
        "surface": find_peak(sphere_surface),
        "complexity": find_peak(complexity_measure),
    }


# Aliases for backward compatibility
v = ball_volume
s = sphere_surface
c = complexity_measure
r = ratio_measure

V = ball_volume
S = sphere_surface
C = complexity_measure
R = ratio_measure

# Fast versions (no validation)
def ball_volume_fast(d):
    return ball_volume(d, validate=False)
def sphere_surface_fast(d):
    return sphere_surface(d, validate=False)
def complexity_measure_fast(d):
    return complexity_measure(d, validate=False)

# Export all
__all__ = [
    'ball_volume', 'sphere_surface', 'complexity_measure', 'ratio_measure',
    'batch_measures', 'find_peak', 'convergence_analysis', 'find_all_peaks',
    'v', 's', 'c', 'r', 'V', 'S', 'C', 'R',
    'ball_volume_fast', 'sphere_surface_fast', 'complexity_measure_fast',
    'PI', 'E', 'NUMERICAL_EPSILON',
]
