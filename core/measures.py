#!/usr/bin/env python3
"""
Dimensional Measures
====================

Core geometric measures for any real dimension d.
The fundamental mathematical framework for dimensional emergence
based on gamma function extensions of sphere geometry.

All measures are computed with numerical stability and proper
handling of edge cases and critical dimensions.
"""

import numpy as np
import warnings

from .constants import CRITICAL_DIMENSIONS, NUMERICAL_EPSILON, PI
from .gamma import gamma_safe, gammaln_safe


def _validate_dimension(d, function_name="measure"):
    """
    Validate dimensional input and issue warnings for edge cases.
    
    Parameters
    ----------
    d : float or array-like
        Dimension value(s) to validate
    function_name : str
        Name of the calling function for warning messages
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
                UserWarning, stacklevel=3
            )
        else:
            warnings.warn(
                f"Negative dimensions detected in {function_name}() "
                f"(min: {np.min(negative_values):.3f}). "
                f"Returning mathematical extension values. "
                f"Physical dimensions are typically d ≥ 0.", 
                UserWarning, stacklevel=3
            )
    
    # Check for large dimensions
    if np.any(d_array > 100):
        large_values = d_array[d_array > 100]
        if len(large_values) == 1:
            warnings.warn(
                f"Large dimension d={large_values[0]:.1f} in {function_name}() "
                f"may underflow to zero due to gamma function behavior.", 
                UserWarning, stacklevel=3
            )
        else:
            warnings.warn(
                f"Large dimensions detected in {function_name}() "
                f"(max: {np.max(large_values):.1f}) "
                f"may underflow to zero due to gamma function behavior.", 
                UserWarning, stacklevel=3
            )


def ball_volume(d):
    """
    Volume of unit d-dimensional ball.

    V_d = π^(d/2) / Γ(d/2 + 1)

    Parameters
    ----------
    d : float or array-like
        Dimension (can be fractional)

    Returns
    -------
    float or array
        Volume of unit d-ball

    Notes
    -----
    Special cases:
    - V_0 = 1 (point)
    - V_1 = 2 (line segment)
    - V_2 = π (disk)
    - V_3 = 4π/3 (sphere)
    """
    # Validate input and issue warnings
    _validate_dimension(d, "ball_volume")

    d = np.asarray(d)

    # Handle d = 0 exactly
    if np.any(np.abs(d) < NUMERICAL_EPSILON):
        result = np.ones_like(d, dtype=float)
        mask = np.abs(d) >= NUMERICAL_EPSILON
        if np.any(mask):
            result[mask] = ball_volume(d[mask])
        return result if d.ndim > 0 else float(result)

    # Use log space for numerical stability when d is large
    if np.any(d > 170):
        large_mask = d > 170
        result = np.zeros_like(d, dtype=float)

        # Small values: direct computation
        if np.any(~large_mask):
            d_small = d[~large_mask]
            log_vol = (d_small / 2) * np.log(PI) - gammaln_safe(d_small / 2 + 1)
            result[~large_mask] = np.exp(log_vol)

        # Large values: use log space
        if np.any(large_mask):
            d_large = d[large_mask]
            log_vol = (d_large / 2) * np.log(PI) - gammaln_safe(d_large / 2 + 1)
            result[large_mask] = np.exp(np.real(log_vol))

        return result if d.ndim > 0 else float(result)

    # Normal computation
    return PI ** (d / 2) / gamma_safe(d / 2 + 1)


def sphere_surface(d):
    """
    Surface area of unit (d-1)-dimensional sphere in d-dimensional space.

    S_d = 2π^(d/2) / Γ(d/2)

    Parameters
    ----------
    d : float or array-like
        Dimension (can be fractional)

    Returns
    -------
    float or array
        Surface area of (d-1)-sphere

    Notes
    -----
    Special cases:
    - S_1 = 2 (two points, boundary of line segment)
    - S_2 = 2π (circle, boundary of disk)
    - S_3 = 4π (sphere, boundary of ball)
    """
    # Validate input and issue warnings
    _validate_dimension(d, "sphere_surface")

    d = np.asarray(d)

    # Handle d = 0 (convention: S_0 = 2)
    if np.any(np.abs(d) < NUMERICAL_EPSILON):
        result = np.full_like(d, 2.0, dtype=float)
        mask = np.abs(d) >= NUMERICAL_EPSILON
        if np.any(mask):
            result[mask] = sphere_surface(d[mask])
        return result if d.ndim > 0 else float(result)

    # Handle d = 1 exactly
    if np.any(np.abs(d - 1) < NUMERICAL_EPSILON):
        result = np.full_like(d, 2.0, dtype=float)
        mask = np.abs(d - 1) >= NUMERICAL_EPSILON
        if np.any(mask):
            result[mask] = sphere_surface(d[mask])
        return result if d.ndim > 0 else float(result)

    # Use log space for large d
    if np.any(d > 170):
        large_mask = d > 170
        result = np.zeros_like(d, dtype=float)

        # Small values
        if np.any(~large_mask):
            d_small = d[~large_mask]
            log_surf = (
                np.log(2) + (d_small / 2) * np.log(PI) - gammaln_safe(d_small / 2)
            )
            result[~large_mask] = np.exp(log_surf)

        # Large values
        if np.any(large_mask):
            d_large = d[large_mask]
            log_surf = (
                np.log(2) + (d_large / 2) * np.log(PI) - gammaln_safe(d_large / 2)
            )
            result[large_mask] = np.exp(np.real(log_surf))

        return result if d.ndim > 0 else float(result)

    # Normal computation
    return 2 * PI ** (d / 2) / gamma_safe(d / 2)


def complexity_measure(d):
    """
    V×S complexity measure: C_d = V_d × S_d

    This measure captures the total "information capacity" of d-dimensional
    space, combining interior volume and boundary surface area.
    Peaks at d ≈ 6, the "complexity peak" of dimensional space.

    Parameters
    ----------
    d : float or array-like
        Dimension

    Returns
    -------
    float or array
        Complexity measure C_d = V_d × S_d
    """
    # Validate input and issue warnings
    _validate_dimension(d, "complexity_measure")

    return ball_volume(d) * sphere_surface(d)


def ratio_measure(d):
    """
    Surface/Volume ratio: R_d = S_d / V_d

    Measures the relative importance of boundary vs interior.
    Increases without bound as d → ∞.

    Parameters
    ----------
    d : float or array-like
        Dimension

    Returns
    -------
    float or array
        Ratio S_d / V_d
    """
    # Validate input and issue warnings
    _validate_dimension(d, "ratio_measure")

    v = ball_volume(d)
    s = sphere_surface(d)
    # Avoid division by zero
    return s / np.maximum(v, NUMERICAL_EPSILON)


def phase_capacity(d):
    """
    Phase capacity at dimension d.

    In the dimensional emergence framework, this represents the maximum
    phase density that dimension d can sustain. Defined as the ball volume.

    Parameters
    ----------
    d : float or array-like
        Dimension

    Returns
    -------
    float or array
        Phase capacity Λ(d) = V_d
    """
    # Validate input and issue warnings
    _validate_dimension(d, "phase_capacity")

    return ball_volume(np.maximum(d, 0.01))  # Avoid d = 0 issues


def find_peak(measure_func, d_min=0.1, d_max=15, num_points=5000):
    """
    Find the peak (maximum) of a measure function.

    Parameters
    ----------
    measure_func : callable
        Function to find peak of
    d_min, d_max : float
        Search range for dimension
    num_points : int
        Number of sample points

    Returns
    -------
    tuple
        (d_peak, value_peak) - dimension and value at peak
    """
    d_range = np.linspace(d_min, d_max, num_points)
    values = np.array([measure_func(d) for d in d_range])

    # Find peak, handling NaN values
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.nan, np.nan

    finite_values = values[finite_mask]
    finite_d = d_range[finite_mask]
    peak_idx = np.argmax(finite_values)

    return finite_d[peak_idx], finite_values[peak_idx]


def find_all_peaks():
    """
    Find peaks of all standard measures.

    Returns
    -------
    dict
        Dictionary with peak locations and values
    """
    results = {}

    # Volume peak
    vol_peak_d, vol_peak_val = find_peak(ball_volume)
    results["volume_peak"] = (vol_peak_d, vol_peak_val)

    # Surface peak
    surf_peak_d, surf_peak_val = find_peak(sphere_surface)
    results["surface_peak"] = (surf_peak_d, surf_peak_val)

    # Complexity peak
    comp_peak_d, comp_peak_val = find_peak(complexity_measure)
    results["complexity_peak"] = (comp_peak_d, comp_peak_val)

    return results


def evaluate_at_critical_dimensions():
    """
    Evaluate all measures at critical dimensional values.

    Returns
    -------
    dict
        Nested dictionary: {measure: {dimension: value}}
    """
    measures = {
        "volume": ball_volume,
        "surface": sphere_surface,
        "complexity": complexity_measure,
        "ratio": ratio_measure,
        "phase_capacity": phase_capacity,
    }

    results = {}
    for measure_name, measure_func in measures.items():
        results[measure_name] = {}
        for dim_name, dim_value in CRITICAL_DIMENSIONS.items():
            try:
                value = measure_func(dim_value)
                results[measure_name][dim_name] = float(value)
            except Exception as e:
                results[measure_name][dim_name] = f"Error: {e}"

    return results


def integrated_measures(d_max=np.inf):
    """
    Compute integrated measures across all dimensions.

    ∫₀^∞ V_d dd = 2e^(π/4) ≈ 4.381
    ∫₀^∞ S_d dd = 2π^(1/2)e^(π/4) ≈ 7.767

    Parameters
    ----------
    d_max : float
        Upper integration limit (use np.inf for analytical values)

    Returns
    -------
    dict
        Integrated measures
    """
    if d_max == np.inf:
        # Analytical values
        from .constants import E

        vol_integral = 2 * E ** (PI / 4)
        surf_integral = 2 * np.sqrt(PI) * E ** (PI / 4)
        ratio = surf_integral / vol_integral  # Should be √π

        return {
            "volume_integral": vol_integral,
            "surface_integral": surf_integral,
            "ratio": ratio,
            "sqrt_pi_check": np.sqrt(PI),
        }
    else:
        # Numerical integration
        from scipy import integrate

        def integrand_v(d):
            return ball_volume(d)

        def integrand_s(d):
            return sphere_surface(d)

        vol_int, _ = integrate.quad(integrand_v, 0, d_max)
        surf_int, _ = integrate.quad(integrand_s, 0, d_max)

        return {
            "volume_integral": vol_int,
            "surface_integral": surf_int,
            "ratio": surf_int / vol_int if vol_int != 0 else np.inf,
        }


if __name__ == "__main__":
    print("DIMENSIONAL MEASURES")
    print("=" * 50)

    # Test standard dimensions
    dims = [0, 1, 2, 3, 4, 5, 6]
    for d in dims:
        v = ball_volume(d)
        s = sphere_surface(d)
        c = complexity_measure(d)
        print(f"d={d}: V={v:.4f}, S={s:.4f}, C={c:.4f}")

    print("\nPEAK ANALYSIS")
    print("=" * 50)
    peaks = find_all_peaks()
    for peak_name, (d_peak, val_peak) in peaks.items():
        print(f"{peak_name}: d={d_peak:.3f}, value={val_peak:.3f}")

    print("\nINTEGRATED MEASURES")
    print("=" * 50)
    integrals = integrated_measures()
    for name, value in integrals.items():
        print(f"{name}: {value:.6f}")
