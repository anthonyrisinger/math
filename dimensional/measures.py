#!/usr/bin/env python3
"""
Dimensional Measures
====================

Unified dimensional measures module consolidating all geometric measure
functionality. Provides robust implementations of ball volumes, sphere
surfaces, complexity measures, and peak finding utilities.

This module consolidates:
- core/measures.py (robust numerical implementations)
- core_measures.py (DimensionalMeasures class interface)

Features:
- High-precision gamma function based measures
- Numerical stability for extreme dimensions
- Peak finding and critical point analysis
- Integrated measure calculations
- Clean API for both functional and class-based usage
"""

import numpy as np
from .gamma import gamma_safe, gammaln_safe, PI, PHI, PSI, E, NUMERICAL_EPSILON

# Add VARPI constant (dimensional coupling constant)
VARPI = gamma_safe(0.25)**2 / (4 * np.sqrt(2 * PI))

# ============================================================================
# CORE DIMENSIONAL MEASURES
# ============================================================================

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
    d = np.asarray(d)

    # Handle d = 0 exactly
    if np.any(np.abs(d) < NUMERICAL_EPSILON):
        result = np.ones_like(d, dtype=float)
        mask = np.abs(d) >= NUMERICAL_EPSILON
        if np.any(mask):
            result[mask] = ball_volume(d[mask])
        return result if d.ndim > 0 else float(result)

    # Use gamma_safe for robustness
    try:
        gamma_term = gamma_safe(d/2 + 1)
        with np.errstate(over='ignore', invalid='ignore'):
            result = np.power(PI, d/2) / gamma_term

        # Handle overflow/underflow
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result if d.ndim > 0 else float(result)

    except (ValueError, OverflowError):
        # Fallback to log-space computation
        return np.exp(ball_volume_log(d))


def ball_volume_log(d):
    """
    Log of ball volume for numerical stability.

    Parameters
    ----------
    d : float or array-like
        Dimension

    Returns
    -------
    float or array
        log(V_d)
    """
    d = np.asarray(d)
    return (d/2) * np.log(PI) - gammaln_safe(d/2 + 1)


def sphere_surface(d):
    """
    Surface area of unit (d-1)-dimensional sphere in d-dimensional space.

    S_d = 2π^(d/2) / Γ(d/2)

    Parameters
    ----------
    d : float or array-like
        Dimension

    Returns
    -------
    float or array
        Surface area of unit (d-1)-sphere

    Notes
    -----
    Special cases:
    - S_1 = 2 (two points on line)
    - S_2 = 2π (circle circumference)
    - S_3 = 4π (sphere surface)
    """
    d = np.asarray(d)

    # Handle special cases
    if np.any(np.abs(d) < NUMERICAL_EPSILON):
        result = np.full_like(d, 2.0, dtype=float)  # S^{-1} convention
        mask = np.abs(d) >= NUMERICAL_EPSILON
        if np.any(mask):
            result[mask] = sphere_surface(d[mask])
        return result if d.ndim > 0 else float(result)

    if np.any(np.abs(d - 1) < NUMERICAL_EPSILON):
        result = np.full_like(d, 2.0, dtype=float)  # S^0 = two points
        mask = np.abs(d - 1) >= NUMERICAL_EPSILON
        if np.any(mask):
            result[mask] = sphere_surface(d[mask])
        return result if d.ndim > 0 else float(result)

    # Use gamma_safe for robustness
    try:
        gamma_term = gamma_safe(d/2)
        with np.errstate(over='ignore', invalid='ignore'):
            result = 2 * np.power(PI, d/2) / gamma_term

        # Handle overflow/underflow
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result if d.ndim > 0 else float(result)

    except (ValueError, OverflowError):
        # Fallback to log-space computation
        return np.exp(sphere_surface_log(d))


def sphere_surface_log(d):
    """
    Log of sphere surface for numerical stability.

    Parameters
    ----------
    d : float or array-like
        Dimension

    Returns
    -------
    float or array
        log(S_d)
    """
    d = np.asarray(d)
    return np.log(2) + (d/2) * np.log(PI) - gammaln_safe(d/2)


def complexity_measure(d):
    """
    Complexity measure: C(d) = V(d) × S(d)

    Represents the product of interior capacity and boundary interface,
    showing the total "information capacity" of d-dimensional space.

    Parameters
    ----------
    d : float or array-like
        Dimension

    Returns
    -------
    float or array
        Complexity measure C(d) = V(d) × S(d)
    """
    return ball_volume(d) * sphere_surface(d)


def ratio_measure(d):
    """
    Surface-to-volume ratio: R(d) = S(d) / V(d)

    Parameters
    ----------
    d : float or array-like
        Dimension

    Returns
    -------
    float or array
        Ratio measure R(d) = S(d) / V(d)
    """
    vol = ball_volume(d)
    surf = sphere_surface(d)
    return np.divide(surf, vol, out=np.zeros_like(surf), where=(vol != 0))


def phase_capacity(d):
    """
    Phase capacity: Λ(d) = V(d) (alias for ball_volume)

    In the dimensional emergence framework, the ball volume
    represents the phase capacity at dimension d.

    Parameters
    ----------
    d : float or array-like
        Dimension

    Returns
    -------
    float or array
        Phase capacity Λ(d) = V(d)
    """
    return ball_volume(d)


# ============================================================================
# PEAK FINDING AND ANALYSIS
# ============================================================================

def find_peak(func, d_min=0.1, d_max=15.0, resolution=10000):
    """
    Find the peak (maximum) of a function over dimension range.

    Parameters
    ----------
    func : callable
        Function to find peak of
    d_min, d_max : float
        Dimension range to search
    resolution : int
        Number of points to evaluate

    Returns
    -------
    tuple
        (peak_dimension, peak_value)
    """
    d_range = np.linspace(d_min, d_max, resolution)
    values = func(d_range)

    # Handle NaN/inf values
    valid_mask = np.isfinite(values)
    if not np.any(valid_mask):
        return np.nan, np.nan

    valid_values = values[valid_mask]
    valid_d = d_range[valid_mask]

    peak_idx = np.argmax(valid_values)
    return valid_d[peak_idx], valid_values[peak_idx]


def find_all_peaks(d_min=0.1, d_max=15.0, resolution=10000):
    """
    Find peaks of all standard dimensional measures.

    Parameters
    ----------
    d_min, d_max : float
        Dimension range to search
    resolution : int
        Number of points to evaluate

    Returns
    -------
    dict
        Dictionary with peak information for each measure
    """
    results = {}

    # Volume peak
    results['volume_peak'] = find_peak(ball_volume, d_min, d_max, resolution)

    # Surface peak
    results['surface_peak'] = find_peak(sphere_surface, d_min, d_max, resolution)

    # Complexity peak
    results['complexity_peak'] = find_peak(complexity_measure, d_min, d_max, resolution)

    # Ratio peak (minimum ratio = maximum compactness)
    def neg_ratio(d):
        return -ratio_measure(d)
    ratio_peak_d, neg_ratio_val = find_peak(neg_ratio, d_min, d_max, resolution)
    results['compactness_peak'] = (ratio_peak_d, -neg_ratio_val)

    return results


def integrated_measures(d_min=0.0, d_max=np.inf, method='adaptive'):
    """
    Compute integrated measures over all dimensions.

    ∫₀^∞ V(d) dd, ∫₀^∞ S(d) dd, etc.

    Parameters
    ----------
    d_min, d_max : float
        Integration limits
    method : str
        Integration method ('adaptive', 'fixed')

    Returns
    -------
    dict
        Dictionary with integrated measures
    """
    from scipy.integrate import quad

    results = {}

    try:
        # Volume integral
        vol_integral, vol_error = quad(ball_volume, d_min, d_max)
        results['volume_integral'] = vol_integral
        results['volume_error'] = vol_error

        # Surface integral
        surf_integral, surf_error = quad(sphere_surface, d_min, d_max)
        results['surface_integral'] = surf_integral
        results['surface_error'] = surf_error

        # Ratio of integrals
        results['integral_ratio'] = surf_integral / vol_integral if vol_integral != 0 else np.inf

    except Exception as e:
        results['error'] = str(e)

    return results


# ============================================================================
# CRITICAL DIMENSIONS AND BOUNDARIES
# ============================================================================

# Critical dimensional boundaries
CRITICAL_DIMENSIONS = {
    'pi_boundary': PI,                    # d = π ≈ 3.14159 (stability boundary)
    'e_boundary': E,                      # d = e ≈ 2.71828 (exponential boundary)
    'phi_boundary': PHI,                  # d = φ ≈ 1.61803 (golden boundary)
    'two_pi_boundary': 2*PI,              # d = 2π ≈ 6.28318 (compression boundary)
    'volume_peak': 5.256789,              # Approximate V(d) peak
    'surface_peak': 7.256789,             # Approximate S(d) peak
    'complexity_peak': 6.0,               # Approximate C(d) peak
}


def get_critical_dimension(name):
    """Get a critical dimension by name."""
    return CRITICAL_DIMENSIONS.get(name, None)


def is_near_critical(d, tolerance=0.1):
    """Check if dimension d is near any critical boundary."""
    for name, critical_d in CRITICAL_DIMENSIONS.items():
        if abs(d - critical_d) < tolerance:
            return True, name
    return False, None


# ============================================================================
# CLASS-BASED INTERFACE (for compatibility)
# ============================================================================

class DimensionalMeasures:
    """
    Class-based interface for dimensional measures.
    Provides compatibility with existing code while using
    the unified functional implementations.
    """

    @staticmethod
    def ball_volume(d):
        """Volume of unit d-ball: V_d = π^(d/2) / Γ(d/2 + 1)"""
        return ball_volume(d)

    @staticmethod
    def sphere_surface(d):
        """Surface area of unit (d-1)-sphere: S_d = 2π^(d/2) / Γ(d/2)"""
        return sphere_surface(d)

    @staticmethod
    def complexity(d):
        """Complexity measure: C(d) = V(d) × S(d)"""
        return complexity_measure(d)

    @staticmethod
    def ratio(d):
        """Surface-to-volume ratio: R(d) = S(d) / V(d)"""
        return ratio_measure(d)

    @staticmethod
    def critical_dimensions():
        """Get dictionary of critical dimensions."""
        return CRITICAL_DIMENSIONS.copy()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def V(d):
    """Shorthand for ball_volume(d)"""
    return ball_volume(d)


def S(d):
    """Shorthand for sphere_surface(d)"""
    return sphere_surface(d)


def C(d):
    """Shorthand for complexity_measure(d)"""
    return complexity_measure(d)


def R(d):
    """Shorthand for ratio_measure(d)"""
    return ratio_measure(d)


# Standard 3D visualization parameters
VIEW_ELEV = np.degrees(PHI - 1)  # ≈ 36.87°
VIEW_AZIM = -45
BOX_ASPECT = (1, 1, 1)


# ============================================================================
# VERIFICATION AND TESTING
# ============================================================================

def verify_measures():
    """Verify mathematical properties of dimensional measures."""
    tolerance = 1e-12
    results = {}

    # Test known values
    results['V_0_equals_1'] = abs(ball_volume(0) - 1.0) < tolerance
    results['V_2_equals_pi'] = abs(ball_volume(2) - PI) < tolerance
    results['S_2_equals_2pi'] = abs(sphere_surface(2) - 2*PI) < tolerance

    # Test relationships
    results['S_1_equals_2'] = abs(sphere_surface(1) - 2.0) < tolerance

    # Peak locations (approximate)
    vol_peak_d, _ = find_peak(ball_volume, 4, 6, 1000)
    results['volume_peak_near_5_26'] = abs(vol_peak_d - 5.26) < 0.1

    return results


if __name__ == "__main__":
    # Quick verification
    print("DIMENSIONAL MEASURES VERIFICATION")
    print("=" * 50)

    verification = verify_measures()
    for test, passed in verification.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:25}: {status}")

    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all(verification.values()) else '❌ SOME TESTS FAILED'}")

    # Show peaks
    print("\nPEAK ANALYSIS:")
    print("-" * 30)
    peaks = find_all_peaks()
    for name, (d, value) in peaks.items():
        print(f"{name:20}: d={d:.3f}, value={value:.6f}")
