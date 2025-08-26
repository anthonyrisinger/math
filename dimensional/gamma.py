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
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.gamma import *  # noqa: F401,F403
from core.gamma import (
    gamma_safe,
    gammaln_safe,
    digamma_safe,
    polygamma_safe,
    gamma_ratio_safe,
    factorial_extension,
    double_factorial_extension,
    gamma_half_integer,
    beta_function,
)

import matplotlib.pyplot as plt
import numpy as np

# Re-export constants for API compatibility
from core.constants import PI, PHI, PSI, E, SQRT_PI, NUMERICAL_EPSILON, GAMMA_OVERFLOW_THRESHOLD, LOG_SPACE_THRESHOLD

# ============================================================================
# ENHANCED ANALYSIS AND VISUALIZATION TOOLS
# ============================================================================


def gamma_explorer(z_range=(-5, 5), n_points=1000, plot=True):
    """
    Explore gamma function behavior across a range with visualization.

    Parameters
    ----------
    z_range : tuple
        Range of values to explore (min, max)
    n_points : int
        Number of points to sample
    plot : bool
        Whether to create visualization

    Returns
    -------
    dict
        Exploration results with values, poles, and analysis
    """
    z_vals = np.linspace(z_range[0], z_range[1], n_points)
    gamma_vals = gamma_safe(z_vals)
    
    # Find poles and special points
    poles = z_vals[np.isinf(gamma_vals)]
    finite_mask = np.isfinite(gamma_vals)
    
    results = {
        'z_values': z_vals,
        'gamma_values': gamma_vals,
        'poles': poles,
        'finite_range': (np.min(gamma_vals[finite_mask]), np.max(gamma_vals[finite_mask])) if np.any(finite_mask) else (np.nan, np.nan)
    }
    
    if plot:
        plt.figure(figsize=(10, 6))
        finite_z = z_vals[finite_mask]
        finite_gamma = gamma_vals[finite_mask]
        
        plt.plot(finite_z, finite_gamma, 'b-', linewidth=2, label='Γ(z)')
        if len(poles) > 0:
            for pole in poles:
                plt.axvline(x=pole, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('z')
        plt.ylabel('Γ(z)')
        plt.title('Gamma Function Explorer')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return results


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
        'gamma': gamma_safe(z_values),
        'ln_gamma': gammaln_safe(z_values),
        'digamma': digamma_safe(z_values),
        'factorial': factorial_extension(z_values[z_values >= 0]) if np.any(z_values >= 0) else np.array([])
    }


def gamma_comparison_plot(z_range=(-4, 6), n_points=500):
    """
    Compare gamma function with related functions.
    
    Parameters
    ----------
    z_range : tuple
        Range to plot
    n_points : int
        Number of points
    """
    z = np.linspace(z_range[0], z_range[1], n_points)
    
    # Remove problematic regions for plotting
    mask = ~((z < 0) & (np.abs(z - np.round(z)) < 1e-10))
    z_clean = z[mask]
    
    plt.figure(figsize=(12, 8))
    
    # Gamma function
    plt.subplot(2, 2, 1)
    gamma_vals = gamma_safe(z_clean)
    finite_mask = np.isfinite(gamma_vals) & (np.abs(gamma_vals) < 100)
    plt.plot(z_clean[finite_mask], gamma_vals[finite_mask], 'b-', linewidth=2)
    plt.title('Γ(z)')
    plt.grid(True, alpha=0.3)
    
    # Log gamma
    plt.subplot(2, 2, 2)
    ln_gamma_vals = gammaln_safe(z_clean)
    finite_mask = np.isfinite(ln_gamma_vals)
    plt.plot(z_clean[finite_mask], ln_gamma_vals[finite_mask], 'g-', linewidth=2)
    plt.title('ln Γ(z)')
    plt.grid(True, alpha=0.3)
    
    # Digamma
    plt.subplot(2, 2, 3)
    digamma_vals = digamma_safe(z_clean)
    finite_mask = np.isfinite(digamma_vals) & (np.abs(digamma_vals) < 50)
    plt.plot(z_clean[finite_mask], digamma_vals[finite_mask], 'r-', linewidth=2)
    plt.title('ψ(z) = Γ\'(z)/Γ(z)')
    plt.grid(True, alpha=0.3)
    
    # Factorial extension
    plt.subplot(2, 2, 4)
    positive_z = z_clean[z_clean >= 0]
    if len(positive_z) > 0:
        fact_vals = factorial_extension(positive_z)
        finite_mask = np.isfinite(fact_vals) & (fact_vals < 100)
        plt.plot(positive_z[finite_mask], fact_vals[finite_mask], 'm-', linewidth=2)
    plt.title('z! = Γ(z+1)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# CONVENIENCE FUNCTIONS FOR INTERACTIVE USE
# ============================================================================

# Shorthand functions for interactive exploration
γ = gamma_safe  # γ(z) for Greek letter fans
ln_γ = gammaln_safe  # ln(γ(z))
ψ = digamma_safe  # ψ(z) = γ'(z)/γ(z)

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
        return {'peaks': [], 'message': 'No finite values found'}
    
    finite_d = d_vals[finite_mask]
    finite_gamma = gamma_vals[finite_mask]
    
    # Simple peak detection
    peaks = []
    for i in range(1, len(finite_gamma) - 1):
        if finite_gamma[i] > finite_gamma[i-1] and finite_gamma[i] > finite_gamma[i+1]:
            peaks.append({
                'dimension': finite_d[i],
                'value': finite_gamma[i]
            })
    
    return {'peaks': peaks, 'd_values': finite_d, 'gamma_values': finite_gamma}


if __name__ == "__main__":
    print("DIMENSIONAL GAMMA FUNCTIONS")
    print("=" * 40)
    
    # Quick test of consolidation
    test_vals = [0.5, 1.0, 2.0, 3.0, 4.5]
    results = quick_gamma_analysis(test_vals)
    
    print("Test values:", test_vals)
    print("Γ(z):", results['gamma'])
    print("ln Γ(z):", results['ln_gamma'])
    print("ψ(z):", results['digamma'])
    
    print("\n✅ Gamma function consolidation successful!")
    print("Core functions imported from ../core/gamma")
    print("Enhanced visualization and analysis tools added")