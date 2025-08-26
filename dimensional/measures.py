#!/usr/bin/env python3
"""
Dimensional Measures
====================

Complete dimensional measures with numerical stability and analysis tools.
Consolidated mathematical implementation with enhanced exploration capabilities.

Core geometric measures for any real dimension d, based on gamma function 
extensions of sphere geometry.
"""

import warnings
import numpy as np

# Import constants and gamma functions
try:
    from ..core.constants import CRITICAL_DIMENSIONS, NUMERICAL_EPSILON, PI, E
    from .gamma import gamma_safe, gammaln_safe
except ImportError:
    from core.constants import CRITICAL_DIMENSIONS, NUMERICAL_EPSILON, PI, E
    from dimensional.gamma import gamma_safe, gammaln_safe

# CORE MATHEMATICAL FUNCTIONS - CONSOLIDATED FROM CORE/

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


def find_all_peaks(d_min=0.1, d_max=15.0, resolution=10000):
    """
    Find peaks of all standard measures.
    
    Parameters
    ----------
    d_min, d_max : float
        Dimension range to search
    resolution : int
        Number of points to evaluate
    
    Returns
    -------
    dict
        Dictionary with peak locations and values
    """
    results = {}
    
    # Volume peak
    vol_peak_d, vol_peak_val = find_peak(ball_volume, d_min, d_max, resolution)
    results["volume_peak"] = (vol_peak_d, vol_peak_val)
    
    # Surface peak
    surf_peak_d, surf_peak_val = find_peak(sphere_surface, d_min, d_max, resolution)
    results["surface_peak"] = (surf_peak_d, surf_peak_val)
    
    # Complexity peak
    comp_peak_d, comp_peak_val = find_peak(complexity_measure, d_min, d_max, resolution)
    results["complexity_peak"] = (comp_peak_d, comp_peak_val)
    
    return results


# DIMENSIONAL MEASURE ALIASES
v = ball_volume
s = sphere_surface
c = complexity_measure
r = ratio_measure

# Uppercase aliases for backward compatibility
V = ball_volume
S = sphere_surface
C = complexity_measure
R = ratio_measure

# ENHANCED ANALYSIS TOOLS (previously in dimensional/measures.py)

def measures_explorer(d_range=(0.1, 10), num_points=1000, show_peaks=True):
    """
    Enhanced measures exploration with peak analysis.
    
    Parameters
    ----------
    d_range : tuple
        (d_min, d_max) range to explore
    num_points : int
        Number of evaluation points
    show_peaks : bool
        Whether to highlight peaks
    
    Returns
    -------
    dict
        Comprehensive measures analysis
    """
    d_min, d_max = d_range
    dimensions = np.linspace(d_min, d_max, num_points)
    
    # Compute all measures
    volumes = np.array([ball_volume(d) for d in dimensions])
    surfaces = np.array([sphere_surface(d) for d in dimensions])
    complexities = np.array([complexity_measure(d) for d in dimensions])
    ratios = np.array([ratio_measure(d) for d in dimensions])
    
    results = {
        "dimensions": dimensions,
        "volume": volumes,
        "surface": surfaces,
        "complexity": complexities,
        "ratio": ratios,
    }
    
    if show_peaks:
        # Find and include peaks
        peaks = find_all_peaks(d_min, d_max, num_points)
        results["peaks"] = peaks
    
    return results


def peak_finder(measure_name, d_range=(0.1, 15), resolution=10000):
    """
    Find peak of specific measure with high resolution.
    
    Parameters
    ----------
    measure_name : str
        Name of measure ('volume', 'surface', 'complexity', 'ratio')
    d_range : tuple
        Search range (d_min, d_max)
    resolution : int
        Number of evaluation points
    
    Returns
    -------
    tuple
        (peak_dimension, peak_value)
    """
    measure_funcs = {
        "volume": ball_volume,
        "surface": sphere_surface,
        "complexity": complexity_measure,
        "ratio": ratio_measure,
    }
    
    if measure_name not in measure_funcs:
        raise ValueError(f"Unknown measure: {measure_name}. Choose from {list(measure_funcs.keys())}")
    
    return find_peak(measure_funcs[measure_name], d_range[0], d_range[1], resolution)


def critical_analysis(d_values=None):
    """
    Analyze measures at critical dimensional values.
    
    Parameters
    ----------
    d_values : list, optional
        Specific dimensions to analyze. Uses CRITICAL_DIMENSIONS if None.
    
    Returns
    -------
    dict
        Analysis results at critical dimensions
    """
    if d_values is None:
        # Use predefined critical dimensions
        d_values = list(CRITICAL_DIMENSIONS.values())
    
    results = {}
    
    for d in d_values:
        results[f"d={d:.3f}"] = {
            "volume": ball_volume(d),
            "surface": sphere_surface(d),
            "complexity": complexity_measure(d),
            "ratio": ratio_measure(d),
            "phase_capacity": phase_capacity(d),
        }
    
    return results


def comparative_plot(measures=None, d_range=(0.1, 10), num_points=1000):
    """
    Generate comparative analysis data for multiple measures.
    
    Parameters
    ----------
    measures : list, optional
        List of measure names. Defaults to all standard measures.
    d_range : tuple
        Dimension range (d_min, d_max)
    num_points : int
        Number of evaluation points
    
    Returns
    -------
    dict
        Comparative data for plotting
    """
    if measures is None:
        measures = ["volume", "surface", "complexity", "ratio"]
    
    measure_funcs = {
        "volume": ball_volume,
        "surface": sphere_surface,
        "complexity": complexity_measure,
        "ratio": ratio_measure,
    }
    
    d_min, d_max = d_range
    dimensions = np.linspace(d_min, d_max, num_points)
    
    results = {"dimensions": dimensions}
    
    for measure in measures:
        if measure in measure_funcs:
            values = np.array([measure_funcs[measure](d) for d in dimensions])
            results[measure] = values
    
    return results


def quick_measure_analysis(d):
    """
    Quick analysis of all measures at a single dimension.
    
    Parameters
    ----------
    d : float
        Dimension to analyze
    
    Returns
    -------
    dict
        All measure values at dimension d
    """
    return {
        "dimension": d,
        "volume": ball_volume(d),
        "surface": sphere_surface(d),
        "complexity": complexity_measure(d),
        "ratio": ratio_measure(d),
        "phase_capacity": phase_capacity(d),
    }


def is_critical_dimension(d, tolerance=1e-3):
    """
    Check if dimension d is near a critical value.
    
    Parameters
    ----------
    d : float
        Dimension to check
    tolerance : float
        Tolerance for comparison
    
    Returns
    -------
    bool or str
        False if not critical, or name of critical dimension
    """
    for name, value in CRITICAL_DIMENSIONS.items():
        if abs(d - value) < tolerance:
            return name
    return False


def volume_ratio(d1, d2):
    """Volume ratio V(d1)/V(d2)."""
    return ball_volume(d1) / ball_volume(d2)


def surface_ratio(d1, d2):
    """Surface ratio S(d1)/S(d2)."""
    return sphere_surface(d1) / sphere_surface(d2)


if __name__ == "__main__":
    # Validate dimensional measures
    
    # Test standard dimensions
    dims = [0, 1, 2, 3, 4, 5, 6]
    for d in dims:
        v = ball_volume(d)
        s = sphere_surface(d)
        c = complexity_measure(d)
        # Validate mathematical properties
        assert v > 0, f"Invalid volume for dimension {d}"
        assert s > 0, f"Invalid surface for dimension {d}"
        assert np.isfinite(c), f"Invalid complexity for dimension {d}"
    
    # Validate peak analysis
    peaks = find_all_peaks()
    assert len(peaks) > 0, "No peaks found"
    for peak_name, (d_peak, val_peak) in peaks.items():
        assert np.isfinite(d_peak), f"Invalid peak dimension for {peak_name}"
        assert np.isfinite(val_peak), f"Invalid peak value for {peak_name}"