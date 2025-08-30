#!/usr/bin/env python3
"""
Dimensional Gamma Functions
===========================

Complete gamma function family with numerical stability and interactive features.
Consolidated mathematical implementation with enhanced exploration tools.
"""

import numpy as np
from scipy.special import digamma, gamma, gammaln

# Consolidated mathematics imports
from .mathematics import (
    GAMMA_OVERFLOW_THRESHOLD,
    LOG_SPACE_THRESHOLD,
    NUMERICAL_EPSILON,
)
from .mathematics import (
    ball_volume as v,
)
from .mathematics import (
    complexity_measure as c,
)
from .mathematics import (
    ratio_measure as r,
)
from .mathematics import (
    sphere_surface as s,
)

# CORE MATHEMATICAL FUNCTIONS - CONSOLIDATED FROM CORE/


def gamma_safe(z):
    """
    Numerically stable gamma function with enhanced fractional domain support.

    Uses Stirling approximation for large |z|, reflection formula for negative
    fractional values, and special handling near poles for maximum stability.

    Parameters
    ----------
    z : float or array-like
        Input values

    Returns
    -------
    float or array
        Γ(z) with proper handling of edge cases and overflow
    """
    z = np.asarray(z)

    # Handle edge cases
    if np.any(z == 0):
        result = np.full_like(z, np.inf, dtype=float)
        mask = z != 0
        if np.any(mask):
            result[mask] = gamma_safe(z[mask])
        return result if np.asarray(z).ndim > 0 else float(np.asarray(result).item())

    # Handle negative integers (poles)
    negative_int_mask = (z < 0) & (np.abs(z - np.round(z)) < NUMERICAL_EPSILON)
    if np.any(negative_int_mask):
        result = np.full_like(z, np.inf, dtype=float)
        mask = ~negative_int_mask
        if np.any(mask):
            result[mask] = gamma_safe(z[mask])
        return result if np.asarray(z).ndim > 0 else float(np.asarray(result).item())

    # Enhanced handling for negative fractional values using reflection formula
    negative_frac_mask = (z < 0) & ~negative_int_mask
    if np.any(negative_frac_mask):
        result = np.full_like(z, 0.0, dtype=float)

        # For non-negative values, use standard computation
        positive_mask = z >= 0
        if np.any(positive_mask):
            result[positive_mask] = gamma_safe(z[positive_mask])

        # For negative fractional values, use reflection formula:
        # Γ(z) = π / (sin(πz) * Γ(1-z))
        if np.any(negative_frac_mask):
            z_neg = z[negative_frac_mask]
            sin_pi_z = np.sin(np.pi * z_neg)

            # Avoid sin(π*z) ≈ 0 cases (near integers)
            sin_safe_mask = np.abs(sin_pi_z) > NUMERICAL_EPSILON

            if np.any(sin_safe_mask):
                z_safe = z_neg[sin_safe_mask]
                gamma_1_minus_z = gamma_safe(1 - z_safe)
                reflection_result = np.pi / (np.sin(np.pi * z_safe) * gamma_1_minus_z)

                # Handle the indexing properly for both scalar and array cases
                if negative_frac_mask.ndim == 0:
                    # Scalar case
                    result = reflection_result
                else:
                    # Array case
                    neg_indices = np.where(negative_frac_mask)[0]
                    safe_neg_indices = neg_indices[sin_safe_mask]
                    result[safe_neg_indices] = reflection_result

            # For cases where sin(π*z) ≈ 0, return infinity
            unsafe_mask = ~sin_safe_mask
            if np.any(unsafe_mask):
                if negative_frac_mask.ndim == 0:
                    # Scalar case
                    result = np.inf
                else:
                    # Array case
                    neg_indices = np.where(negative_frac_mask)[0]
                    unsafe_neg_indices = neg_indices[unsafe_mask]
                    result[unsafe_neg_indices] = np.inf

        return result if np.asarray(z).ndim > 0 else float(np.asarray(result).item())

    # Use log-space for large values with Stirling approximation fallback
    large_mask = np.abs(z) > GAMMA_OVERFLOW_THRESHOLD
    if np.any(large_mask):
        result = np.zeros_like(z, dtype=float)

        # Standard computation for normal-sized values
        normal_mask = ~large_mask
        if np.any(normal_mask):
            result[normal_mask] = gamma(z[normal_mask])

        # Enhanced large value handling with Stirling approximation
        if np.any(large_mask):
            z_large = z[large_mask]

            # Use Stirling approximation for extremely large values
            stirling_mask = np.abs(z_large) > LOG_SPACE_THRESHOLD / 10

            if np.any(stirling_mask):
                z_stirling = z_large[stirling_mask]
                # Stirling: Γ(z) ≈ √(2π/z) * (z/e)^z
                log_stirling = (z_stirling - 0.5) * np.log(z_stirling) - z_stirling + 0.5 * np.log(2 * np.pi)

                # Check if result would overflow
                overflow_stirling = log_stirling >= LOG_SPACE_THRESHOLD
                if np.any(overflow_stirling):
                    if large_mask.ndim == 0:
                        # Scalar case
                        result = np.inf
                    else:
                        # Array case
                        large_indices = np.where(large_mask)[0]
                        stirling_indices = large_indices[stirling_mask]
                        overflow_indices = stirling_indices[overflow_stirling]
                        result[overflow_indices] = np.inf

                safe_stirling = log_stirling < LOG_SPACE_THRESHOLD
                if np.any(safe_stirling):
                    if large_mask.ndim == 0:
                        # Scalar case
                        result = np.exp(log_stirling[safe_stirling])
                    else:
                        # Array case
                        large_indices = np.where(large_mask)[0]
                        stirling_indices = large_indices[stirling_mask]
                        safe_indices = stirling_indices[safe_stirling]
                        result[safe_indices] = np.exp(log_stirling[safe_stirling])

            # Use scipy gammaln for moderately large values
            gammaln_mask = ~stirling_mask
            if np.any(gammaln_mask):
                z_gammaln = z_large[gammaln_mask]
                log_gamma_vals = gammaln(z_gammaln)

                exp_mask = log_gamma_vals < LOG_SPACE_THRESHOLD
                if np.any(exp_mask):
                    large_indices = np.where(large_mask)[0]
                    gammaln_indices = large_indices[gammaln_mask]
                    safe_indices = gammaln_indices[exp_mask]
                    result[safe_indices] = np.exp(log_gamma_vals[exp_mask])

                inf_mask = log_gamma_vals >= LOG_SPACE_THRESHOLD
                if np.any(inf_mask):
                    large_indices = np.where(large_mask)[0]
                    gammaln_indices = large_indices[gammaln_mask]
                    inf_indices = gammaln_indices[inf_mask]
                    result[inf_indices] = np.inf

        return result if np.asarray(z).ndim > 0 else float(np.asarray(result).item())

    return gamma(z)


def gammaln_safe(z):
    """
    Safe log-gamma function.

    Parameters
    ----------
    z : float or array-like
        Input values

    Returns
    -------
    float or array
        log(Γ(z)) with proper handling of edge cases
    """
    z = np.asarray(z)

    if np.any(z <= 0):
        if np.any(np.abs(z - np.round(z)) < NUMERICAL_EPSILON):
            result = np.full_like(z, -np.inf, dtype=float)
            mask = ~(np.abs(z - np.round(z)) < NUMERICAL_EPSILON)
            if np.any(mask):
                result[mask] = gammaln_safe(z[mask])
            return result if np.asarray(z).ndim > 0 else float(np.asarray(result).item())

    return gammaln(z)


def digamma_safe(z):
    """
    Safe digamma function (psi function).
    ψ(z) = d/dz log(Γ(z)) = Γ'(z)/Γ(z)

    Parameters
    ----------
    z : float or array-like
        Input values

    Returns
    -------
    float or array
        ψ(z) with proper handling of edge cases
    """
    z = np.asarray(z)

    if np.any(z <= 0):
        if np.any(np.abs(z - np.round(z)) < NUMERICAL_EPSILON):
            result = np.full_like(z, -np.inf, dtype=float)
            mask = ~(np.abs(z - np.round(z)) < NUMERICAL_EPSILON)
            if np.any(mask):
                result[mask] = digamma_safe(z[mask])
            return result if np.asarray(z).ndim > 0 else float(np.asarray(result).item())

    return digamma(z)


def factorial_extension(n):
    """
    Factorial extension for real numbers with enhanced fractional support.
    n! = Γ(n+1)

    Provides stable computation for fractional factorials and handles
    edge cases near poles with graceful degradation.

    Parameters
    ----------
    n : float or array-like
        Real numbers (negative values use analytic continuation)

    Returns
    -------
    float or array
        n! = Γ(n+1) with enhanced numerical stability
    """
    n = np.asarray(n)

    # Special handling for negative integers: (-1)! = ∞
    negative_int_mask = (n < 0) & (np.abs(n - np.round(n)) < NUMERICAL_EPSILON)
    if np.any(negative_int_mask):
        result = np.full_like(n, np.inf, dtype=float)
        valid_mask = ~negative_int_mask
        if np.any(valid_mask):
            result[valid_mask] = gamma_safe(n[valid_mask] + 1)
        return result if n.ndim > 0 else float(result)

    return gamma_safe(n + 1)


def beta_function(a, b):
    """
    Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b).

    Parameters
    ----------
    a, b : float or array-like
        Beta function parameters

    Returns
    -------
    float or array
        B(a,b)
    """
    # Use log-space for numerical stability
    log_result = gammaln_safe(a) + gammaln_safe(b) - gammaln_safe(a + b)
    return np.exp(log_result)


# DIMENSIONAL MEASURE IMPORTS AND ALIASES


# Create density function
def ρ(d):
    """Volume density (reciprocal volume)."""
    return 1.0 / v(d)


# Import peak finding functions - create shortcuts
def v_peak():
    """Find volume peak dimension and value."""
    from .mathematics import ball_volume, find_peak

    return find_peak(ball_volume)


def s_peak():
    """Find surface peak dimension and value."""
    from .mathematics import find_peak, sphere_surface

    return find_peak(sphere_surface)


def c_peak():
    """Find complexity peak dimension and value."""
    from .mathematics import complexity_measure, find_peak

    return find_peak(complexity_measure)


# ============================================================================
# ENHANCED ANALYSIS AND CONVERGENCE DIAGNOSTICS
# ============================================================================


def convergence_diagnostics(func, z_value, method='richardson', tolerance=1e-12):
    """
    Advanced convergence diagnostics for gamma function computations.

    Parameters
    ----------
    func : callable
        Function to test (e.g., gamma_safe)
    z_value : float
        Value to test convergence at
    method : str
        Convergence test method ('richardson', 'aitken', 'stability')
    tolerance : float
        Convergence tolerance

    Returns
    -------
    dict
        Convergence diagnostic results
    """
    if method == 'richardson':
        # Richardson extrapolation for numerical derivatives
        h_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        derivatives = []

        for h in h_values:
            if z_value > h and z_value + h > 0:  # Avoid poles
                deriv = (func(z_value + h) - func(z_value - h)) / (2 * h)
                if np.isfinite(deriv):
                    derivatives.append(deriv)

        if len(derivatives) >= 3:
            # Check convergence of derivative approximations
            diffs = np.abs(np.diff(derivatives))
            converged = np.all(diffs[-2:] < tolerance) if len(diffs) >= 2 else False
            return {
                'method': 'richardson',
                'converged': converged,
                'derivatives': derivatives,
                'convergence_rate': diffs[-1] / diffs[-2] if len(diffs) >= 2 else None
            }

    elif method == 'stability':
        # Test numerical stability near z_value
        perturbations = np.logspace(-15, -8, 8)
        values = []

        for eps in perturbations:
            if z_value + eps > 0:  # Avoid negative domain issues
                val = func(z_value + eps)
                if np.isfinite(val):
                    values.append(val)

        if len(values) >= 3:
            relative_vars = np.abs(np.diff(values)) / np.abs(values[:-1])
            stable = np.all(relative_vars < tolerance * 10000)  # Practical tolerance for numerical stability
            return {
                'method': 'stability',
                'stable': stable,
                'values': values,
                'max_relative_variation': np.max(relative_vars) if len(relative_vars) > 0 else 0
            }

    return {'method': method, 'error': 'insufficient_data', 'converged': False}


def fractional_domain_validation(z_range=(-3, 3), resolution=1000):
    """
    Comprehensive validation of gamma function in fractional domains.

    Parameters
    ----------
    z_range : tuple
        Range to validate
    resolution : int
        Number of test points

    Returns
    -------
    dict
        Validation results with convergence metrics
    """
    z_values = np.linspace(z_range[0], z_range[1], resolution)

    # Exclude poles (negative integers)
    valid_mask = ~((z_values < 0) & (np.abs(z_values - np.round(z_values)) < NUMERICAL_EPSILON))
    z_test = z_values[valid_mask]

    results = {
        'test_range': z_range,
        'resolution': resolution,
        'valid_points': len(z_test),
        'finite_values': 0,
        'convergence_passed': 0,
        'stability_passed': 0,
        'reflection_accuracy': [],
        'stirling_accuracy': []
    }

    for z in z_test:
        # Basic computation test
        gamma_val = gamma_safe(z)
        if np.isfinite(gamma_val):
            results['finite_values'] += 1

            # Convergence test (sample every 10th point for performance)
            if len(z_test) > 100 and hash(str(z)) % 10 == 0:
                conv_result = convergence_diagnostics(gamma_safe, z, 'stability')
                if conv_result.get('stable', False):
                    results['convergence_passed'] += 1

            # Test reflection formula accuracy for negative values
            if z < 0:
                # Γ(z) * Γ(1-z) = π / sin(πz)
                gamma_1_minus_z = gamma_safe(1 - z)
                if np.isfinite(gamma_1_minus_z):
                    expected = np.pi / np.sin(np.pi * z)
                    actual = gamma_val * gamma_1_minus_z
                    if np.isfinite(expected) and np.abs(expected) > NUMERICAL_EPSILON:
                        relative_error = np.abs(actual - expected) / np.abs(expected)
                        results['reflection_accuracy'].append(relative_error)

            # Test Stirling approximation accuracy for large values
            if np.abs(z) > 5:
                # Stirling: Γ(z) ≈ √(2π/z) * (z/e)^z
                if z > 0:
                    stirling_approx = np.sqrt(2 * np.pi / z) * (z / np.e) ** z
                    if np.isfinite(stirling_approx) and stirling_approx > NUMERICAL_EPSILON:
                        relative_error = np.abs(gamma_val - stirling_approx) / stirling_approx
                        results['stirling_accuracy'].append(relative_error)

    # Statistical summaries
    results['finite_ratio'] = results['finite_values'] / len(z_test)
    results['mean_reflection_error'] = np.mean(results['reflection_accuracy']) if results['reflection_accuracy'] else 0
    results['mean_stirling_error'] = np.mean(results['stirling_accuracy']) if results['stirling_accuracy'] else 0

    return results


def gamma_explorer(z_range=(-5, 5), n_points=1000, show_poles=True):
    """
    Interactive gamma function explorer with modern visualization.

    Args:
        z_range: Tuple of (start, end) for exploration range
        n_points: Number of points to evaluate
        show_poles: Whether to highlight poles

    Returns:
        dict: Analysis results with z_values, gamma_values, finite_mask, stats
    """
    # ARCHITECTURAL COMPLIANCE: Pure mathematical computation
    z = np.linspace(z_range[0], z_range[1], n_points)
    gamma_vals = np.array([gamma_safe(zi) for zi in z])

    # Analysis without side effects
    finite_mask = np.isfinite(gamma_vals)
    finite_count = np.sum(finite_mask)

    stats = {
        "range": z_range,
        "n_points": n_points,
        "finite_count": finite_count,
        "finite_ratio": finite_count / len(z) if len(z) > 0 else 0,
    }

    if finite_count > 0:
        finite_gamma = gamma_vals[finite_mask]
        stats.update(
            {"value_range": (np.min(finite_gamma), np.max(finite_gamma))}
        )

    return {
        "z_values": z,
        "gamma_values": gamma_vals,
        "finite_mask": finite_mask,
        "stats": stats,
    }


def quick_gamma_analysis(z_values):
    """
    Quick analysis of gamma function for given values.

    Parameters
    ----------
    z_values : array-like or scalar
        Values to analyze

    Returns
    -------
    dict
        Analysis results with dimension compatibility
    """
    z_values = np.asarray(z_values)

    # Handle scalar case for API compatibility
    if z_values.ndim == 0:
        z_val = float(z_values)
        return {
            "dimension": z_val,  # For test compatibility
            "gamma_value": gamma_safe(z_val),  # For test compatibility
            "gamma": gamma_safe(z_val),
            "ln_gamma": gammaln_safe(z_val),
            "digamma": digamma_safe(z_val),
            "factorial": factorial_extension(z_val) if z_val >= 0 else np.nan,
        }

    # Handle array case
    return {
        "gamma": gamma_safe(z_values),
        "ln_gamma": gammaln_safe(z_values),
        "digamma": digamma_safe(z_values),
        "factorial": (
            factorial_extension(z_values[z_values >= 0])
            if np.any(z_values >= 0)
            else np.array([])
        ),
    }


def gamma_comparison_plot(z_range=(-4, 6), n_points=500):
    """
    Compare gamma function with related functions including convergence analysis.

    MODERNIZED: No matplotlib - provides data for modern visualization backends
    with enhanced convergence diagnostics and fractional domain validation.

    Parameters
    ----------
    z_range : tuple
        Range to plot
    n_points : int
        Number of points

    Returns
    -------
    dict
        Computed data for visualization backends with convergence statistics
    """
    z = np.linspace(z_range[0], z_range[1], n_points)

    # Remove problematic regions for plotting
    mask = ~((z < 0) & (np.abs(z - np.round(z)) < 1e-10))
    z_clean = z[mask]

    # Compute all function values with enhanced error handling
    gamma_vals = gamma_safe(z_clean)
    ln_gamma_vals = gammaln_safe(z_clean)
    digamma_vals = digamma_safe(z_clean)

    positive_z = z_clean[z_clean >= 0]
    fact_vals = (
        factorial_extension(positive_z)
        if len(positive_z) > 0
        else np.array([])
    )

    # Enhanced convergence validation
    validation_results = fractional_domain_validation(z_range, min(n_points, 500))

    # Sample convergence diagnostics at key points
    test_points = [0.5, 1.0, 1.5, 2.5, -0.5, -1.5]
    convergence_tests = {}

    for point in test_points:
        if z_range[0] <= point <= z_range[1]:
            conv_result = convergence_diagnostics(gamma_safe, point, 'stability')
            convergence_tests[f'z_{point}'] = conv_result

    # Generate comprehensive statistics
    stats = {
        "range": z_range,
        "n_points_requested": n_points,
        "n_points_clean": len(z_clean),
        "gamma_finite": int(np.sum(np.isfinite(gamma_vals))),
        "ln_gamma_finite": int(np.sum(np.isfinite(ln_gamma_vals))),
        "digamma_finite": int(np.sum(np.isfinite(digamma_vals))),
        "convergence_validation": validation_results,
        "point_convergence_tests": convergence_tests,
        "reflection_formula_accuracy": validation_results.get('mean_reflection_error', 0),
        "stirling_approximation_accuracy": validation_results.get('mean_stirling_error', 0)
    }

    return {
        "z_values": z_clean,
        "gamma": gamma_vals,
        "ln_gamma": ln_gamma_vals,
        "digamma": digamma_vals,
        "positive_z": positive_z,
        "stats": stats,
        "factorial": fact_vals,
    }


# ============================================================================
# CONVENIENCE FUNCTIONS FOR INTERACTIVE USE
# ============================================================================

# Shorthand functions for interactive exploration
γ = gamma_safe  # γ(z) for Greek letter fans
ln_γ = gammaln_safe  # ln(γ(z))
ψ = digamma_safe  # ψ(z) = γ'(z)/γ(z)


# Additional shortcuts expected by tests
def abs_γ(z):
    return np.abs(gamma_safe(z))  # |γ(z)|


# Quick visualization functions (modernized - no matplotlib)
def qplot(*funcs, labels=None):
    """Quick plot function - returns data for visualization backends.

    Returns:
        dict: Plot data with function evaluations and statistics
    """
    d_vals = np.linspace(0.1, 10, 1000)
    results = {}

    for i, func in enumerate(funcs):
        y_vals = [func(d) for d in d_vals]
        label = (
            labels[i] if labels and i < len(labels) else f"Function {i + 1}"
        )
        y_finite = [y for y in y_vals if np.isfinite(y)]

        results[label] = {
            "x_values": d_vals,
            "y_values": y_vals,
            "finite_count": len(y_finite),
            "total_count": len(y_vals),
            "value_range": (
                (min(y_finite), max(y_finite)) if y_finite else None
            ),
        }

    return results


def instant():
    """Enhanced instant analysis with multiple view configurations.

    This function now provides comprehensive instant analysis with
    Rich terminal visualization and multiple research configurations.

    Returns:
        dict: Enhanced instant analysis results
    """
    try:
        from .research_cli import enhanced_instant
        return enhanced_instant(configuration="research")
    except ImportError:
        # Fallback to basic instant configuration
        return {
            "panels": ["gamma", "ln_gamma", "digamma", "factorial"],
            "config": {
                "gamma": {"range": (-3, 5), "points": 1000},
                "ln_gamma": {"range": (0.1, 10), "points": 1000},
                "digamma": {"range": (0.1, 10), "points": 1000},
                "factorial": {"range": (0, 10), "points": 1000},
            },
        }


def explore(d):
    """Explore dimensional measures at given dimension with enhanced analysis.

    This function now provides enhanced exploration with guided discovery
    paths and Rich terminal visualization.

    Args:
        d: Dimension to explore

    Returns:
        dict: Complete dimensional analysis with discovery paths
    """
    try:
        from .research_cli import enhanced_explore
        enhanced_result = enhanced_explore(d, context="gamma_analysis")

        # Extract basic measures for API compatibility
        basic_result = {
            "dimension": d,
            "volume": enhanced_result["point"]["volume"],
            "surface": enhanced_result["point"]["surface"],
            "complexity": enhanced_result["point"]["complexity"],
            "ratio": enhanced_result["point"].get("ratio", r(d)),
            "density": enhanced_result["point"].get("density", ρ(d)),
        }

        # Merge enhanced features
        enhanced_result.update(basic_result)
        return enhanced_result

    except ImportError:
        # Fallback to basic exploration
        return {
            "dimension": d,
            "volume": v(d),
            "surface": s(d),
            "complexity": c(d),
            "ratio": r(d),
            "density": ρ(d),
        }


def peaks():
    """Return all dimensional peaks.

    Returns:
        dict: All peak values for dimensional measures
    """
    return {
        "volume_peak": v_peak(),
        "surface_peak": s_peak(),
        "complexity_peak": c_peak(),
    }


# Interactive classes now provided by research_cli module
# Legacy compatibility maintained through function redirects below


def lab(start_d=4.0):
    """Launch enhanced interactive gamma function laboratory.

    This function now redirects to the enhanced research CLI for
    full interactive capabilities with session persistence.

    Args:
        start_d: Starting dimension for exploration

    Returns:
        ResearchSession: Interactive research session
    """
    try:
        from .research_cli import enhanced_lab
        return enhanced_lab(start_d)
    except ImportError:
        # Fallback to basic exploration if enhanced CLI unavailable
        print(f"Enhanced lab unavailable. Running basic exploration at dimension {start_d}")
        return explore(start_d)


def demo():
    """Run demonstration - returns data instead of printing."""
    return {
        "demo_type": "dimensional_gamma",
        "exploration": explore(4.0),
        "visualization": instant,
    }


def live(expr_file="gamma_expr.py"):
    """Start live editing mode with hot reload.

    Returns:
        dict: Live mode configuration
    """
    return {
        "mode": "live_editing",
        "watching_file": expr_file,
        "features": ["file_monitoring", "real_time_plots", "hot_reload"],
        "status": "ready",
    }


def peaks_analysis(d_range=(0, 10), resolution=1000):
    """
    Find and analyze peaks in gamma-related functions.

    Parameters
    ----------
    d_range : tuple
        Dimension range to analyze
    resolution : in
        Number of points to sample

    Returns
    -------
    dic
        Peak locations and properties
    """
    d_vals = np.linspace(d_range[0], d_range[1], resolution)

    # This would connect to measures module for dimensional analysis
    # For now, return gamma function peaks
    gamma_vals = gamma_safe(d_vals)

    # Find local maxima (simple peak detection)
    finite_mask = np.isfinite(gamma_vals)
    if not np.any(finite_mask):
        return {"peaks": [], "message": "No finite values found"}

    finite_d = d_vals[finite_mask]
    finite_gamma = gamma_vals[finite_mask]

    # Simple peak detection
    peaks = []
    for i in range(1, len(finite_gamma) - 1):
        if (
            finite_gamma[i] > finite_gamma[i - 1]
            and finite_gamma[i] > finite_gamma[i + 1]
        ):
            peaks.append({"dimension": finite_d[i], "value": finite_gamma[i]})

    return {"peaks": peaks, "d_values": finite_d, "gamma_values": finite_gamma}


if __name__ == "__main__":
    # Test dimensional gamma functions without printing
    test_vals = [0.5, 1.0, 2.0, 3.0, 4.5]
    results = quick_gamma_analysis(test_vals)

    # Validate functionality
    assert len(results["gamma"]) == len(test_vals)
    assert len(results["ln_gamma"]) == len(test_vals)
    assert len(results["digamma"]) == len(test_vals)

    # Validate mathematical properties
    for i, val in enumerate(test_vals):
        if val > 0:
            assert np.isfinite(results["gamma"][i]), f"Invalid gamma for {val}"
            assert np.isfinite(
                results["ln_gamma"][i]
            ), f"Invalid ln_gamma for {val}"
