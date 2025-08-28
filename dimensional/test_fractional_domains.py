#!/usr/bin/env python3
"""
Test Suite: Fractional Domain Analysis and Convergence Diagnostics
==================================================================

Comprehensive tests for fractional dimension capabilities with enhanced
numerical stability and convergence validation.
"""

import numpy as np
import pytest

from .gamma import (
    convergence_diagnostics,
    factorial_extension,
    fractional_domain_validation,
    gamma_safe,
)
from .mathematics import (
    NUMERICAL_EPSILON,
    ball_volume,
    complexity_measure,
    sphere_surface,
)
from .mathematics.validation import ConvergenceDiagnostics, NumericalStabilityTester


class TestFractionalGammaEnhancements:
    """Test enhanced gamma function for fractional domains."""

    def test_negative_fractional_reflection_formula(self):
        """Test reflection formula for negative fractional values."""
        test_values = [-0.5, -1.5, -2.5, -3.25, -0.75]

        for z in test_values:
            gamma_z = gamma_safe(z)

            # Verify reflection formula: Γ(z) * Γ(1-z) = π/sin(πz)
            gamma_1_minus_z = gamma_safe(1 - z)
            expected = np.pi / np.sin(np.pi * z)
            actual = gamma_z * gamma_1_minus_z

            if np.isfinite(expected) and np.isfinite(actual):
                relative_error = abs(actual - expected) / abs(expected)
                assert relative_error < 1e-10, f"Reflection formula failed for z={z}"

    def test_stirling_approximation_accuracy(self):
        """Test Stirling approximation for large values."""
        large_values = [10, 20, 50, 100]

        for z in large_values:
            gamma_z = gamma_safe(z)

            # Stirling approximation: Γ(z) ≈ √(2π/z) * (z/e)^z
            stirling_approx = np.sqrt(2 * np.pi / z) * (z / np.e) ** z

            if np.isfinite(gamma_z) and np.isfinite(stirling_approx):
                relative_error = abs(gamma_z - stirling_approx) / stirling_approx
                # Stirling is asymptotic, so allow larger error for smaller values
                tolerance = 0.01 if z >= 50 else 0.1
                assert relative_error < tolerance, f"Stirling approximation poor for z={z}"

    def test_factorial_extension_negative_domain(self):
        """Test factorial extension in negative domain."""
        # Test negative non-integer values
        test_values = [-0.5, -1.5, -2.5]

        for n in test_values:
            factorial_n = factorial_extension(n)
            gamma_n_plus_1 = gamma_safe(n + 1)

            if np.isfinite(factorial_n) and np.isfinite(gamma_n_plus_1):
                assert abs(factorial_n - gamma_n_plus_1) < NUMERICAL_EPSILON

        # Test that negative integers give infinity
        for n in [-1, -2, -3]:
            factorial_n = factorial_extension(n)
            assert np.isinf(factorial_n), f"(-{-n-1})! should be infinite"

    def test_pole_avoidance_near_negative_integers(self):
        """Test stability near poles (negative integers)."""
        pole_locations = [-1, -2, -3, -4]
        offsets = [1e-8, 1e-6, 1e-4, 1e-2]

        for pole in pole_locations:
            for offset in offsets:
                # Test both sides of pole
                z_left = pole - offset
                z_right = pole + offset

                gamma_left = gamma_safe(z_left)
                gamma_right = gamma_safe(z_right)

                # Values should be finite near (but not at) poles
                assert np.isfinite(gamma_left) or np.isinf(gamma_left), f"Invalid result at z={z_left}"
                assert np.isfinite(gamma_right) or np.isinf(gamma_right), f"Invalid result at z={z_right}"

                # Values should have opposite signs on either side of pole
                if np.isfinite(gamma_left) and np.isfinite(gamma_right):
                    assert np.sign(gamma_left) != np.sign(gamma_right), f"Sign inconsistency near pole {pole}"


class TestConvergenceDiagnostics:
    """Test advanced convergence diagnostic capabilities."""

    def setup_method(self):
        """Setup convergence diagnostics instance."""
        self.diagnostics = ConvergenceDiagnostics(tolerance=1e-10)

    def test_richardson_extrapolation_convergence(self):
        """Test Richardson extrapolation for derivative convergence."""
        # Test on smooth function (gamma) at well-behaved points
        test_points = [1.5, 2.5, 3.5]

        for x in test_points:
            result = self.diagnostics.richardson_extrapolation(gamma_safe, x)

            # Should have enough data points
            assert 'derivatives' in result
            assert len(result['derivatives']) >= 3

            # Should show convergence for smooth function
            if 'converged' in result:
                # Allow realistic tolerance for numerical derivatives of special functions
                assert result.get('errors', [])[-1] < 5e-5, f"Poor convergence at x={x}"

    def test_aitken_acceleration(self):
        """Test Aitken acceleration for sequence convergence."""
        # Create a slowly converging sequence: 1/n
        slow_sequence = [1/n for n in range(1, 20)]

        result = self.diagnostics.aitken_acceleration(slow_sequence)

        assert 'accelerated_sequence' in result
        assert len(result['accelerated_sequence']) > 0

        # Acceleration should improve convergence
        original_diff = abs(slow_sequence[-1] - slow_sequence[-2])
        if 'convergence_improvement' in result:
            assert result['convergence_improvement'] <= original_diff

    def test_fractional_convergence_test(self):
        """Test convergence in fractional domain."""
        test_functions = [
            (ball_volume, 2.5),
            (sphere_surface, 3.7),
            (gamma_safe, 1.25)
        ]

        for func, x_base in test_functions:
            result = self.diagnostics.fractional_convergence_test(func, x_base)

            # Should have convergence data
            if 'forward_differences' in result:
                assert len(result['forward_differences']) > 0
                assert len(result['backward_differences']) > 0

                # Consistency between forward and backward should be reasonable
                if 'avg_consistency' in result:
                    assert result['avg_consistency'] < 1e-3, f"Poor consistency for {func.__name__} at {x_base}"


class TestFractionalDomainValidation:
    """Test comprehensive fractional domain validation."""

    def test_fractional_domain_validation_gamma(self):
        """Test gamma function validation across fractional domain."""
        result = fractional_domain_validation(z_range=(-2.5, 5.5), resolution=200)

        # Should have validation metrics
        assert 'finite_ratio' in result
        assert 'mean_reflection_error' in result
        assert 'mean_stirling_error' in result

        # Expect reasonable finite ratio (avoiding poles)
        assert result['finite_ratio'] > 0.8, "Too many invalid values in gamma domain"

        # Reflection formula should be accurate
        if result['reflection_accuracy']:
            assert result['mean_reflection_error'] < 1e-6, "Poor reflection formula accuracy"

    def test_convergence_diagnostics_integration(self):
        """Test integration of convergence diagnostics with validation."""
        result = convergence_diagnostics(gamma_safe, 2.5, method='stability')

        # Should return stability information
        assert 'method' in result
        assert result['method'] == 'stability'

        if 'stable' in result:
            # For well-behaved point, should be stable
            assert result['stable'], "Gamma function unstable at well-behaved point"


class TestNumericalStabilityEnhancements:
    """Test enhanced numerical stability testing."""

    def setup_method(self):
        """Setup stability tester."""
        self.stability_tester = NumericalStabilityTester()

    def test_enhanced_gamma_stability_testing(self):
        """Test enhanced gamma stability with convergence metrics."""
        results = self.stability_tester.test_gamma_stability()

        # Should have enhanced fractional testing
        assert 'fractional_range' in results
        assert 'negative_fractional' in results

        # Enhanced metrics should be available
        fractional_results = results['fractional_range']
        assert 'convergence_passed' in fractional_results
        assert 'convergence_tested' in fractional_results

        # Negative fractional should handle reflection formula
        negative_results = results['negative_fractional']
        assert 'reflection_formula_tests' in negative_results

        # Large values should validate Stirling approximation
        large_results = results['large_values']
        assert 'stirling_approximation_errors' in large_results
        if large_results['stirling_approximation_errors']:
            assert large_results['mean_stirling_error'] < 0.1, "Poor Stirling approximation"

    def test_enhanced_measure_stability_testing(self):
        """Test enhanced dimensional measure stability."""
        results = self.stability_tester.test_measure_stability()

        # Should have enhanced convergence testing
        fractional_results = results['fractional_dimensions']
        assert 'convergence_tests' in fractional_results
        assert 'convergence_passed' in fractional_results
        assert 'peak_analysis' in fractional_results

        # Peak analysis should identify known peaks
        peak_analysis = fractional_results['peak_analysis']
        if 'volume_peak_dimension' in peak_analysis:
            # Volume peak should be near known value (≈5.26)
            vol_peak = peak_analysis['volume_peak_dimension']
            assert 5.0 < vol_peak < 6.0, f"Volume peak at unexpected location: {vol_peak}"

        if 'surface_peak_dimension' in peak_analysis:
            # Surface peak should be near known value (≈7.26)
            surf_peak = peak_analysis['surface_peak_dimension']
            assert 6.5 < surf_peak < 8.0, f"Surface peak at unexpected location: {surf_peak}"


class TestFractionalEdgeCases:
    """Test edge cases in fractional domain computations."""

    def test_very_small_fractional_values(self):
        """Test gamma function for very small fractional values."""
        tiny_values = [1e-10, 1e-8, 1e-6, 1e-4]

        for z in tiny_values:
            gamma_z = gamma_safe(z)

            # Should be finite and very large (since Γ(z) ≈ 1/z for small z)
            assert np.isfinite(gamma_z), f"Gamma not finite for z={z}"
            assert gamma_z > 1/z * 0.5, f"Gamma too small for z={z}"

    def test_fractional_values_near_integers(self):
        """Test fractional values very close to integers."""
        epsilons = [1e-12, 1e-10, 1e-8, 1e-6]
        integers = [1, 2, 3, 4]

        for n in integers:
            for eps in epsilons:
                z_left = n - eps
                z_right = n + eps

                gamma_left = gamma_safe(z_left)
                gamma_right = gamma_safe(z_right)
                gamma_n = gamma_safe(n)

                # Should be continuous across integer
                if np.isfinite(gamma_n):
                    assert np.isfinite(gamma_left), f"Discontinuity at {n}-{eps}"
                    assert np.isfinite(gamma_right), f"Discontinuity at {n}+{eps}"

                    # Values should be close to integer value (allow realistic numerical tolerance)
                    assert abs(gamma_left - gamma_n) < 1e-5, f"Poor continuity at {n}"
                    assert abs(gamma_right - gamma_n) < 1e-5, f"Poor continuity at {n}"

    def test_dimensional_measures_fractional_edge_cases(self):
        """Test dimensional measures at fractional edge cases."""
        # Test near critical dimensions
        critical_tests = [
            (5.26, ball_volume),      # Near volume peak
            (7.26, sphere_surface),   # Near surface peak
            (6.35, complexity_measure) # Near complexity peak
        ]

        epsilons = [1e-6, 1e-4, 1e-2]

        for critical_d, measure_func in critical_tests:
            for eps in epsilons:
                # Test both sides of critical point
                left_val = measure_func(critical_d - eps)
                right_val = measure_func(critical_d + eps)
                center_val = measure_func(critical_d)

                # All values should be finite and positive
                assert np.isfinite(left_val) and left_val > 0
                assert np.isfinite(right_val) and right_val > 0
                assert np.isfinite(center_val) and center_val > 0

                # Should show continuity
                assert abs(left_val - center_val) / center_val < 1e-3
                assert abs(right_val - center_val) / center_val < 1e-3


class TestMathematicalConsistency:
    """Test mathematical consistency across fractional domain."""

    def test_gamma_recurrence_fractional(self):
        """Test gamma recurrence relation for fractional values."""
        fractional_values = [0.25, 0.5, 0.75, 1.25, 1.5, 2.25, 2.75]

        for z in fractional_values:
            gamma_z = gamma_safe(z)
            gamma_z_plus_1 = gamma_safe(z + 1)

            if np.isfinite(gamma_z) and np.isfinite(gamma_z_plus_1):
                # Γ(z+1) = z * Γ(z)
                expected = z * gamma_z
                relative_error = abs(gamma_z_plus_1 - expected) / abs(expected)
                assert relative_error < 1e-12, f"Recurrence failed for z={z}"

    def test_beta_function_fractional_consistency(self):
        """Test beta function for fractional values."""
        from .gamma import beta_function

        fractional_pairs = [(0.5, 1.5), (0.25, 0.75), (1.25, 2.75)]

        for a, b in fractional_pairs:
            beta_ab = beta_function(a, b)
            beta_ba = beta_function(b, a)  # Symmetry

            # Beta function should be symmetric
            assert abs(beta_ab - beta_ba) < 1e-12, f"Beta function not symmetric for ({a}, {b})"

            # Check relation with gamma function
            gamma_a = gamma_safe(a)
            gamma_b = gamma_safe(b)
            gamma_a_plus_b = gamma_safe(a + b)

            if all(np.isfinite([gamma_a, gamma_b, gamma_a_plus_b])):
                expected_beta = gamma_a * gamma_b / gamma_a_plus_b
                relative_error = abs(beta_ab - expected_beta) / abs(expected_beta)
                assert relative_error < 1e-10, f"Beta-gamma relation failed for ({a}, {b})"

    def test_dimensional_measure_relationships_fractional(self):
        """Test relationships between dimensional measures for fractional d."""
        fractional_dims = [0.5, 1.5, 2.5, 3.5, 4.5]

        for d in fractional_dims:
            volume = ball_volume(d)
            surface = sphere_surface(d)

            if np.isfinite(volume) and np.isfinite(surface) and volume > 0:
                # Surface should be derivative of volume times d
                # S_d = d * V_d (exact relationship)
                expected_surface = d * volume
                relative_error = abs(surface - expected_surface) / surface
                assert relative_error < 1e-12, f"Surface-volume relation failed for d={d}"


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v"])
