#!/usr/bin/env python3
"""
Dimensional Measures
====================

Enhanced dimensional measures module that imports robust core functionality
and adds visualization, analysis, and peak detection tools.

This module preserves API compatibility while building upon the
robust mathematical implementations in core.measures.
"""

# Import all robust core functionality
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.measures import *  # noqa: F401,F403
from core.measures import (
    ball_volume,
    sphere_surface,
    complexity_measure,
    phase_capacity,
    find_all_peaks,
    evaluate_at_critical_dimensions,
    integrated_measures,
    ratio_measure,
)

import numpy as np
import matplotlib.pyplot as plt

# Re-export constants
from core.constants import PI, PHI, PSI, E, SQRT_PI, NUMERICAL_EPSILON, CRITICAL_DIMENSIONS

# ============================================================================
# ENHANCED VISUALIZATION AND ANALYSIS TOOLS
# ============================================================================


def measures_explorer(d_range=(0, 10), n_points=1000, plot=True):
    """
    Explore all dimensional measures across a range with visualization.

    Parameters
    ----------
    d_range : tuple
        Range of dimensions to explore (min, max)
    n_points : int
        Number of points to sample
    plot : bool
        Whether to create visualization

    Returns
    -------
    dict
        Exploration results with all measure values
    """
    d_vals = np.linspace(d_range[0], d_range[1], n_points)
    
    results = {
        'dimensions': d_vals,
        'ball_volumes': ball_volume(d_vals),
        'sphere_surfaces': sphere_surface(d_vals),
        'complexity_measures': complexity_measure(d_vals),
        'phase_capacities': phase_capacity(d_vals)
    }
    
    if plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Ball volumes
        finite_mask = np.isfinite(results['ball_volumes']) & (results['ball_volumes'] > 0)
        ax1.semilogy(d_vals[finite_mask], results['ball_volumes'][finite_mask], 'b-', linewidth=2)
        ax1.set_title('Ball Volume V(d)')
        ax1.set_xlabel('Dimension d')
        ax1.set_ylabel('V(d)')
        ax1.grid(True, alpha=0.3)
        
        # Sphere surfaces
        finite_mask = np.isfinite(results['sphere_surfaces']) & (results['sphere_surfaces'] > 0)
        ax2.semilogy(d_vals[finite_mask], results['sphere_surfaces'][finite_mask], 'g-', linewidth=2)
        ax2.set_title('Sphere Surface S(d)')
        ax2.set_xlabel('Dimension d')
        ax2.set_ylabel('S(d)')
        ax2.grid(True, alpha=0.3)
        
        # Complexity measure
        finite_mask = np.isfinite(results['complexity_measures'])
        ax3.plot(d_vals[finite_mask], results['complexity_measures'][finite_mask], 'r-', linewidth=2)
        ax3.set_title('Complexity Measure C(d)')
        ax3.set_xlabel('Dimension d')
        ax3.set_ylabel('C(d)')
        ax3.grid(True, alpha=0.3)
        
        # Phase capacity
        finite_mask = np.isfinite(results['phase_capacities']) & (results['phase_capacities'] > 0)
        ax4.plot(d_vals[finite_mask], results['phase_capacities'][finite_mask], 'm-', linewidth=2)
        ax4.set_title('Phase Capacity Λ(d)')
        ax4.set_xlabel('Dimension d')
        ax4.set_ylabel('Λ(d)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return results


def peak_finder(measure_func, d_range=(0, 15), resolution=10000):
    """
    Find peaks in a dimensional measure function.

    Parameters
    ----------
    measure_func : callable
        Function to analyze (e.g., ball_volume, complexity_measure)
    d_range : tuple
        Range of dimensions to search
    resolution : int
        Number of points to sample

    Returns
    -------
    dict
        Peak locations and properties
    """
    d_vals = np.linspace(d_range[0], d_range[1], resolution)
    
    try:
        measure_vals = measure_func(d_vals)
    except Exception as e:
        return {'error': f'Function evaluation failed: {e}', 'peaks': []}
    
    # Filter out non-finite values
    finite_mask = np.isfinite(measure_vals)
    if not np.any(finite_mask):
        return {'peaks': [], 'message': 'No finite values found'}
    
    finite_d = d_vals[finite_mask]
    finite_measures = measure_vals[finite_mask]
    
    # Find local maxima using simple peak detection
    peaks = []
    for i in range(1, len(finite_measures) - 1):
        if (finite_measures[i] > finite_measures[i-1] and 
            finite_measures[i] > finite_measures[i+1] and
            finite_measures[i] > np.max(finite_measures) * 0.1):  # Significant peaks only
            peaks.append({
                'dimension': finite_d[i],
                'value': finite_measures[i],
                'prominence': finite_measures[i] - min(finite_measures[i-1], finite_measures[i+1])
            })
    
    # Sort by prominence
    peaks.sort(key=lambda x: x['prominence'], reverse=True)
    
    return {
        'peaks': peaks,
        'd_values': finite_d,
        'measure_values': finite_measures,
        'function_name': getattr(measure_func, '__name__', 'unknown')
    }


def critical_analysis(d_range=(0, 20), resolution=5000):
    """
    Comprehensive analysis of critical dimensions and transitions.

    Parameters
    ----------
    d_range : tuple
        Range to analyze
    resolution : int
        Sampling resolution

    Returns
    -------
    dict
        Complete critical dimension analysis
    """
    d_vals = np.linspace(d_range[0], d_range[1], resolution)
    
    # Calculate all measures
    volumes = ball_volume(d_vals)
    surfaces = sphere_surface(d_vals)
    complexity = complexity_measure(d_vals)
    capacity = phase_capacity(d_vals)
    
    # Find critical points for each measure
    volume_peaks = peak_finder(ball_volume, d_range, resolution)
    surface_peaks = peak_finder(sphere_surface, d_range, resolution)
    complexity_peaks = peak_finder(complexity_measure, d_range, resolution)
    
    # Find known critical dimensions in range
    known_critical = [d for d in CRITICAL_DIMENSIONS if d_range[0] <= d <= d_range[1]]
    
    return {
        'dimension_range': d_range,
        'known_critical_dimensions': known_critical,
        'volume_analysis': volume_peaks,
        'surface_analysis': surface_peaks,
        'complexity_analysis': complexity_peaks,
        'summary': {
            'total_volume_peaks': len(volume_peaks['peaks']),
            'total_surface_peaks': len(surface_peaks['peaks']),
            'total_complexity_peaks': len(complexity_peaks['peaks']),
            'known_critical_count': len(known_critical)
        }
    }


def comparative_plot(dimensions, measures=None, log_scale=True):
    """
    Create comparative plots of multiple measures for specific dimensions.

    Parameters
    ----------
    dimensions : array-like
        Specific dimensions to compare
    measures : list, optional
        List of measure functions to include
    log_scale : bool
        Whether to use logarithmic scale
    """
    if measures is None:
        measures = [ball_volume, sphere_surface, complexity_measure, phase_capacity]
        measure_names = ['Ball Volume', 'Sphere Surface', 'Complexity', 'Phase Capacity']
    else:
        measure_names = [getattr(f, '__name__', str(f)) for f in measures]
    
    dimensions = np.asarray(dimensions)
    
    plt.figure(figsize=(12, 8))
    
    for i, (measure, name) in enumerate(zip(measures, measure_names)):
        try:
            values = measure(dimensions)
            finite_mask = np.isfinite(values) & (values > 0 if log_scale else True)
            
            if log_scale:
                plt.semilogy(dimensions[finite_mask], values[finite_mask], 
                           'o-', linewidth=2, label=name, markersize=6)
            else:
                plt.plot(dimensions[finite_mask], values[finite_mask], 
                        'o-', linewidth=2, label=name, markersize=6)
        except Exception as e:
            print(f"Warning: Could not plot {name}: {e}")
    
    plt.xlabel('Dimension')
    plt.ylabel('Measure Value' + (' (log scale)' if log_scale else ''))
    plt.title('Comparative Dimensional Measures')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mark critical dimensions
    for critical_d in CRITICAL_DIMENSIONS:
        if dimensions.min() <= critical_d <= dimensions.max():
            plt.axvline(x=critical_d, color='red', linestyle='--', alpha=0.5)
    
    plt.show()


def quick_measure_analysis(dimension):
    """
    Quick analysis of all measures at a specific dimension.

    Parameters
    ----------
    dimension : float
        Dimension to analyze

    Returns
    -------
    dict
        All measure values and properties
    """
    return {
        'dimension': dimension,
        'ball_volume': ball_volume(dimension),
        'sphere_surface': sphere_surface(dimension),
        'complexity_measure': complexity_measure(dimension),
        'phase_capacity': phase_capacity(dimension),
        'is_critical': dimension in CRITICAL_DIMENSIONS,
        'ratio_measure': ratio_measure(dimension)
    }


def is_critical_dimension(dimension, tolerance=1e-6):
    """Check if dimension is critical."""
    return any(abs(dimension - cd) < tolerance for cd in CRITICAL_DIMENSIONS)


def volume_ratio(d1, d2):
    """Volume ratio between two dimensions."""
    return ball_volume(d1) / ball_volume(d2) if ball_volume(d2) != 0 else np.inf


def surface_ratio(d1, d2):
    """Surface ratio between two dimensions."""
    return sphere_surface(d1) / sphere_surface(d2) if sphere_surface(d2) != 0 else np.inf


# ============================================================================
# CONVENIENCE SHORTCUTS FOR INTERACTIVE USE
# ============================================================================

# Short aliases for interactive exploration
V = ball_volume          # V(d) - ball volume
S = sphere_surface       # S(d) - sphere surface
C = complexity_measure   # C(d) - complexity measure
Λ = phase_capacity      # Λ(d) - phase capacity (Greek lambda)

def peaks():
    """Find all major peaks in dimensional measures."""
    print("🔍 Finding peaks in dimensional measures...")
    
    analysis = critical_analysis(d_range=(0, 15), resolution=5000)
    
    print(f"\n📊 CRITICAL DIMENSION ANALYSIS")
    print(f"{'='*50}")
    print(f"Volume peaks found: {analysis['summary']['total_volume_peaks']}")
    print(f"Surface peaks found: {analysis['summary']['total_surface_peaks']}")  
    print(f"Complexity peaks found: {analysis['summary']['total_complexity_peaks']}")
    print(f"Known critical dimensions: {analysis['known_critical_dimensions']}")
    
    # Show top peaks
    for measure_type in ['volume', 'surface', 'complexity']:
        peaks_data = analysis[f'{measure_type}_analysis']['peaks'][:3]  # Top 3
        if peaks_data:
            print(f"\n🏔️ Top {measure_type} peaks:")
            for i, peak in enumerate(peaks_data, 1):
                print(f"  {i}. d={peak['dimension']:.3f}, value={peak['value']:.6f}")
    
    return analysis


if __name__ == "__main__":
    print("DIMENSIONAL MEASURES")
    print("=" * 40)
    
    # Quick test of consolidation
    test_dims = [1, 2, 3, 4, 5]
    
    print("Test dimensions:", test_dims)
    for d in test_dims:
        result = quick_measure_analysis(d)
        print(f"d={d}: V={result['ball_volume']:.4f}, S={result['sphere_surface']:.4f}, C={result['complexity_measure']:.4f}")
    
    print("\n✅ Measures consolidation successful!")
    print("Core functions imported from ../core/measures")
    print("Enhanced visualization and analysis tools added")