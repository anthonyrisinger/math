#!/usr/bin/env python3
"""
Geometric Measures Module
=========================

Unified geometric measures for dimensional analysis, consolidating the best
implementations from dim0-dim5 modules with proper edge case handling.

Core geometric formulas:
- Ball volumes V(d) = π^(d/2) / Γ(d/2 + 1)
- Sphere surfaces S(d) = d × V(d)
- Complexity measures C(d) = V(d) × S(d)
- Critical dimension analysis
"""

import numpy as np
from scipy.special import gamma, gammaln
from typing import Union, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Core mathematical constants
PHI = (1 + np.sqrt(5)) / 2      # Golden ratio ≈ 1.618
PSI = 1 / PHI                   # Golden conjugate ≈ 0.618
VARPI = gamma(0.25)**2 / (2 * np.sqrt(2 * np.pi))  # ≈ 1.311
PI = np.pi
E = np.e

class GeometricMeasures:
    """
    Complete geometric measures with robust numerical handling.

    Handles edge cases properly:
    - 0-sphere (S^0) = two points {-1, +1}, measure = 2
    - 1-ball (B^1) = interval [-1, 1], volume = 2
    - 1-sphere (S^1) = circle, circumference = 2π
    """

    @staticmethod
    def ball_volume(d: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Volume of unit d-ball: V(d) = π^(d/2) / Γ(d/2 + 1)

        Parameters
        ----------
        d : float or array
            Dimension(s)

        Returns
        -------
        float or array
            Ball volume(s)
        """
        d = np.asarray(d)

        # Handle edge cases
        if np.any(np.abs(d) < 1e-10):
            scalar_input = d.ndim == 0
            d = np.atleast_1d(d)
            result = np.ones_like(d, dtype=float)

            # For non-zero dimensions
            mask = np.abs(d) >= 1e-10
            if np.any(mask):
                d_nonzero = d[mask]

                # Use log-space for large dimensions to avoid overflow
                large_mask = d_nonzero > 170
                if np.any(large_mask):
                    d_large = d_nonzero[large_mask]
                    log_vol = (d_large/2) * np.log(PI) - gammaln(d_large/2 + 1)
                    result[mask][large_mask] = np.exp(np.real(log_vol))

                # Direct calculation for reasonable dimensions
                normal_mask = ~large_mask
                if np.any(normal_mask):
                    d_normal = d_nonzero[normal_mask]
                    result[mask][normal_mask] = PI**(d_normal/2) / gamma(d_normal/2 + 1)

            return result[0] if scalar_input else result

        # Direct path for all non-zero dimensions
        if np.any(d > 170):
            # Use log-space for numerical stability
            log_vol = (d/2) * np.log(PI) - gammaln(d/2 + 1)
            return np.exp(np.real(log_vol))
        else:
            return PI**(d/2) / gamma(d/2 + 1)

    @staticmethod
    def sphere_surface(d: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Surface area of unit (d-1)-sphere: S(d) = d × V(d)

        Parameters
        ----------
        d : float or array
            Dimension(s)

        Returns
        -------
        float or array
            Sphere surface area(s)
        """
        return d * GeometricMeasures.ball_volume(d)

    @staticmethod
    def complexity_measure(d: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Complexity measure: C(d) = V(d) × S(d)

        This captures the interplay between volume and surface,
        revealing critical dimensions where complexity peaks.

        Parameters
        ----------
        d : float or array
            Dimension(s)

        Returns
        -------
        float or array
            Complexity measure(s)
        """
        V = GeometricMeasures.ball_volume(d)
        S = GeometricMeasures.sphere_surface(d)
        return V * S

    @staticmethod
    def find_volume_peak() -> float:
        """Find dimension where ball volume peaks."""
        from scipy.optimize import minimize_scalar

        def neg_volume(d):
            return -GeometricMeasures.ball_volume(d)

        result = minimize_scalar(neg_volume, bounds=(0, 10), method='bounded')
        return result.x

    @staticmethod
    def find_complexity_peak() -> float:
        """Find dimension where complexity measure peaks."""
        from scipy.optimize import minimize_scalar

        def neg_complexity(d):
            return -GeometricMeasures.complexity_measure(d)

        result = minimize_scalar(neg_complexity, bounds=(0, 20), method='bounded')
        return result.x

    @staticmethod
    def critical_dimensions() -> Dict[str, float]:
        """Find all critical dimensions and special values."""
        return {
            'volume_peak': GeometricMeasures.find_volume_peak(),
            'complexity_peak': GeometricMeasures.find_complexity_peak(),
            'golden_ratio': PHI,
            'euler_number': E,
            'pi': PI,
            'varpi': VARPI
        }

class DimensionalAnalyzer:
    """
    Complete dimensional analysis with visualization and insights.
    """

    def __init__(self):
        self.measures = GeometricMeasures()

    def analyze_dimension(self, d: float) -> Dict[str, Any]:
        """Comprehensive analysis of a specific dimension."""
        V = self.measures.ball_volume(d)
        S = self.measures.sphere_surface(d)
        C = self.measures.complexity_measure(d)

        # Find local gradients
        epsilon = 1e-6
        dV_dd = (self.measures.ball_volume(d + epsilon) - V) / epsilon
        dS_dd = (self.measures.sphere_surface(d + epsilon) - S) / epsilon
        dC_dd = (self.measures.complexity_measure(d + epsilon) - C) / epsilon

        return {
            'dimension': d,
            'ball_volume': V,
            'sphere_surface': S,
            'complexity': C,
            'volume_gradient': dV_dd,
            'surface_gradient': dS_dd,
            'complexity_gradient': dC_dd,
            'is_volume_peak': abs(dV_dd) < 1e-6,
            'is_complexity_peak': abs(dC_dd) < 1e-6,
            'critical_distances': self._critical_distances(d)
        }

    def _critical_distances(self, d: float) -> Dict[str, float]:
        """Calculate distances to critical dimensions."""
        criticals = self.measures.critical_dimensions()
        return {
            name: abs(d - value)
            for name, value in criticals.items()
        }

    def explore_dimension(self, d: float, show_plot: bool = True) -> Dict[str, Any]:
        """Explore a dimension with optional visualization."""
        analysis = self.analyze_dimension(d)

        if show_plot:
            self.plot_dimensional_analysis(d)

        return analysis

    def plot_dimensional_analysis(self, d_focus: float, d_range: Tuple[float, float] = (0, 10)):
        """Plot comprehensive dimensional analysis."""
        d_values = np.linspace(d_range[0], d_range[1], 1000)

        V_values = self.measures.ball_volume(d_values)
        S_values = self.measures.sphere_surface(d_values)
        C_values = self.measures.complexity_measure(d_values)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Dimensional Analysis: Focus on d = {d_focus:.3f}', fontsize=16)

        # Volume plot
        axes[0,0].plot(d_values, V_values, 'b-', linewidth=2, label='V(d)')
        axes[0,0].axvline(d_focus, color='red', linestyle='--', alpha=0.7)
        axes[0,0].set_xlabel('Dimension d')
        axes[0,0].set_ylabel('Ball Volume V(d)')
        axes[0,0].set_title('Ball Volume')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()

        # Surface plot
        axes[0,1].plot(d_values, S_values, 'g-', linewidth=2, label='S(d)')
        axes[0,1].axvline(d_focus, color='red', linestyle='--', alpha=0.7)
        axes[0,1].set_xlabel('Dimension d')
        axes[0,1].set_ylabel('Sphere Surface S(d)')
        axes[0,1].set_title('Sphere Surface')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()

        # Complexity plot
        axes[1,0].plot(d_values, C_values, 'purple', linewidth=2, label='C(d)')
        axes[1,0].axvline(d_focus, color='red', linestyle='--', alpha=0.7)
        axes[1,0].set_xlabel('Dimension d')
        axes[1,0].set_ylabel('Complexity C(d)')
        axes[1,0].set_title('Complexity Measure')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()

        # Combined plot with log scale
        axes[1,1].semilogy(d_values, V_values, 'b-', label='V(d)', linewidth=2)
        axes[1,1].semilogy(d_values, S_values, 'g-', label='S(d)', linewidth=2)
        axes[1,1].semilogy(d_values, C_values, 'purple', label='C(d)', linewidth=2)
        axes[1,1].axvline(d_focus, color='red', linestyle='--', alpha=0.7)
        axes[1,1].set_xlabel('Dimension d')
        axes[1,1].set_ylabel('Measures (log scale)')
        axes[1,1].set_title('All Measures (Log Scale)')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()

        plt.tight_layout()
        plt.show()

# Convenience functions for package interface
def V(d):
    """Ball volume shorthand."""
    return GeometricMeasures.ball_volume(d)

def S(d):
    """Sphere surface shorthand."""
    return GeometricMeasures.sphere_surface(d)

def C(d):
    """Complexity measure shorthand."""
    return GeometricMeasures.complexity_measure(d)

def analyze(d):
    """Quick analysis shorthand."""
    analyzer = DimensionalAnalyzer()
    return analyzer.analyze_dimension(d)

def explore(d):
    """Quick exploration with plots."""
    analyzer = DimensionalAnalyzer()
    return analyzer.explore_dimension(d)

def find_peaks():
    """Find all critical peaks."""
    return GeometricMeasures.critical_dimensions()

# Module test
def test_geometric_measures():
    """Test the geometric measures module."""
    print("GEOMETRIC MEASURES MODULE TEST")
    print("=" * 50)

    # Test basic measures
    print("Basic measures:")
    for d in [0, 1, 2, 3, 4, 5]:
        V_d = V(d)
        S_d = S(d)
        C_d = C(d)
        print(f"  d={d}: V={V_d:.6f}, S={S_d:.6f}, C={C_d:.6f}")

    # Test peaks
    print(f"\nCritical dimensions:")
    criticals = find_peaks()
    for name, value in criticals.items():
        print(f"  {name}: {value:.6f}")

    # Test fractional dimensions
    print(f"\nFractional dimensions:")
    for d in [PHI, E, PI]:
        analysis = analyze(d)
        print(f"  d={d:.3f}: V={analysis['ball_volume']:.6f}, complexity={analysis['complexity']:.6f}")

    print("\n✅ All geometric measures tests completed!")

if __name__ == "__main__":
    test_geometric_measures()
