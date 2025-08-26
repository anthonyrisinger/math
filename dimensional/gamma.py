#!/usr/bin/env python3
"""
Dimensional Gamma Functions
===========================

Enhanced gamma function module that imports robust core functionality
and adds interactive exploration, visualization, and analysis tools.

This module preserves API compatibility while building upon the
robust mathematical implementations in core.gamma.
"""

# Import all robust core functionality

import numpy as np

# ARCHITECT MANDATE: ZERO VISUALIZATION IMPORTS IN MATHEMATICAL MODULES
# Visualization backends must be completely isolated from core mathematics
# _viz_backend = None  # DEPRECATED - Use modern backends through separate interfaces

# Import and re-export constants for API compatibility

# Re-export core gamma functions with hybrid imports for flexibility
try:
    from ..core.gamma import *  # noqa: F401,F403
    from ..core.gamma import (
        digamma_safe,
        factorial_extension,
        gamma_safe,
        gammaln_safe,
    )

    # Import dimensional measures functions
    from ..core.measures import ball_volume as v
    from ..core.measures import complexity_measure as c
    from ..core.measures import ratio_measure as r
    from ..core.measures import sphere_surface as s
except ImportError:
    # Fallback for script execution
    from core.gamma import *  # noqa: F401,F403
    from core.gamma import (
        digamma_safe,
        factorial_extension,
        gamma_safe,
        gammaln_safe,
    )
    from core.measures import ball_volume as v
    from core.measures import complexity_measure as c
    from core.measures import ratio_measure as r
    from core.measures import sphere_surface as s


# Create density function
def ρ(d):
    """Volume density (reciprocal volume)."""
    return 1.0 / v(d)


# Import peak finding functions - create shortcuts
def v_peak():
    """Find volume peak dimension."""
    try:
        from ..core.measures import ball_volume, find_peak
    except ImportError:
        from core.measures import ball_volume, find_peak

    return find_peak(ball_volume)[0]


def s_peak():
    """Find surface peak dimension."""
    try:
        from ..core.measures import find_peak, sphere_surface
    except ImportError:
        from core.measures import find_peak, sphere_surface

    return find_peak(sphere_surface)[0]


def c_peak():
    """Find complexity peak dimension."""
    try:
        from ..core.measures import complexity_measure, find_peak
    except ImportError:
        from core.measures import complexity_measure, find_peak

    return find_peak(complexity_measure)[0]


# ============================================================================
# ENHANCED ANALYSIS AND VISUALIZATION TOOLS
# ============================================================================


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
        stats.update({"value_range": (np.min(finite_gamma), np.max(finite_gamma))})

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
    z_values : array-like
        Values to analyze

    Returns
    -------
    dict
        Analysis results
    """
    z_values = np.asarray(z_values)

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
    Compare gamma function with related functions.

    MODERNIZED: No matplotlib - provides data for modern visualization backends.

    Parameters
    ----------
    z_range : tuple
        Range to plot
    n_points : int
        Number of points

    Returns
    -------
    dict
        Computed data for visualization backends with statistics
    """
    z = np.linspace(z_range[0], z_range[1], n_points)

    # Remove problematic regions for plotting
    mask = ~((z < 0) & (np.abs(z - np.round(z)) < 1e-10))
    z_clean = z[mask]

    # Compute all function values
    gamma_vals = gamma_safe(z_clean)
    ln_gamma_vals = gammaln_safe(z_clean)
    digamma_vals = digamma_safe(z_clean)

    positive_z = z_clean[z_clean >= 0]
    fact_vals = factorial_extension(positive_z) if len(positive_z) > 0 else np.array([])

    # Generate statistics without printing
    stats = {
        "range": z_range,
        "n_points_requested": n_points,
        "n_points_clean": len(z_clean),
        "gamma_finite": int(np.sum(np.isfinite(gamma_vals))),
        "ln_gamma_finite": int(np.sum(np.isfinite(ln_gamma_vals))),
        "digamma_finite": int(np.sum(np.isfinite(digamma_vals))),
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
        try:
            y_vals = [func(d) for d in d_vals]
            label = labels[i] if labels and i < len(labels) else f"Function {i+1}"
            y_finite = [y for y in y_vals if np.isfinite(y)]

            results[label] = {
                "x_values": d_vals,
                "y_values": y_vals,
                "finite_count": len(y_finite),
                "total_count": len(y_vals),
                "value_range": (min(y_finite), max(y_finite)) if y_finite else None,
            }
        except Exception as e:
            results[label] = {"error": str(e)}

    return results


def instant():
    """Instant 4-panel visualization data.

    Returns:
        dict: Configuration for 4-panel gamma function visualization
    """
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
    """Explore dimensional measures at given dimension.

    Returns:
        dict: Complete dimensional analysis at dimension d
    """
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


# Mock interactive classes
class GammaLab:
    def __init__(self, start_d=4.0):
        self.d = start_d
        self.mode = 0
        self.modes = ["volume", "surface", "complexity"]

    def show(self):
        """Show the gamma lab interface."""
        explore(self.d)


class LiveGamma:
    def __init__(self, expr_file=None):
        self.d = 4.0
        self.expr_file = expr_file


def lab(start_d=4.0):
    """Launch gamma lab."""
    lab = GammaLab(start_d=start_d)
    return lab


def demo():
    """Run demonstration - returns data instead of printing."""
    return {
        "demo_type": "dimensional_gamma",
        "exploration": explore(4.0),
        "visualization": instant(),
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
    resolution : int
        Number of points to sample

    Returns
    -------
    dict
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
            assert np.isfinite(results["ln_gamma"][i]), f"Invalid ln_gamma for {val}"
