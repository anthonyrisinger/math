"""
Comprehensive Test Suite for Gamma Function Extensions
=======================================================

Demonstrates the complete testing approach for mathematical functions.
This serves as a template for testing all mathematical components.

Tests cover:
- Mathematical property validation
- Numerical stability and edge cases
- Property-based testing with Hypothesis
- Performance benchmarks
- Integration with other components
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# Import the gamma functions (adjust import path as needed)
try:
    from core.gamma import gamma_ratio_safe, gamma_safe, gammaln_safe

    GAMMA_AVAILABLE = True
except ImportError:
    GAMMA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GAMMA_AVAILABLE, reason="Gamma functions not available"
)


class TestGammaMathematicalProperties:
    """Test fundamental mathematical properties of the gamma function."""

    @pytest.mark.mathematical
    def test_factorial_property(self, known_gamma_values, math_tolerance):
        """Test Γ(n+1) = n! for positive integers."""
        for n in range(1, 8):
            gamma_value = gamma_safe(n + 1)
            factorial_value = math.factorial(n)
            assert (
                abs(gamma_value - factorial_value) < math_tolerance
            ), f"Γ({n+1}) = {gamma_value}, but {n}! = {factorial_value}"

    @pytest.mark.mathematical
    def test_known_exact_values(self, known_gamma_values, math_tolerance):
        """Test gamma function against known exact values."""
        for x, expected in known_gamma_values.items():
            actual = gamma_safe(x)
            assert (
                abs(actual - expected) < math_tolerance
            ), f"Γ({x}) = {actual}, expected {expected}"

    @pytest.mark.mathematical
    def test_recurrence_relation(self, math_tolerance):
        """Test Γ(z+1) = z·Γ(z) for z > 0."""
        test_values = [0.1, 0.5, 1.0, 1.7, 2.3, 3.5, 4.9]

        for z in test_values:
            gamma_z = gamma_safe(z)
            gamma_z_plus_1 = gamma_safe(z + 1)
            expected = z * gamma_z

            assert (
                abs(gamma_z_plus_1 - expected) < math_tolerance
            ), f"Recurrence failed at z={z}: Γ({z+1}) = {gamma_z_plus_1}, z·Γ({z}) = {expected}"

    @pytest.mark.mathematical
    def test_reflection_formula(self, math_tolerance):
        """Test Γ(z)Γ(1-z) = π/sin(πz) for non-integer z."""
        test_values = [0.1, 0.3, 0.7, 0.9]  # Avoid integers and half-integers

        for z in test_values:
            gamma_z = gamma_safe(z)
            gamma_1_minus_z = gamma_safe(1 - z)
            left_side = gamma_z * gamma_1_minus_z

            right_side = math.pi / math.sin(math.pi * z)

            relative_error = abs(left_side - right_side) / abs(right_side)
            assert (
                relative_error < math_tolerance
            ), f"Reflection formula failed at z={z}: {left_side} ≠ {right_side}"

    @pytest.mark.mathematical
    def test_duplication_formula(self, math_tolerance):
        """Test Γ(z)Γ(z+1/2) = √π · 2^(1-2z) · Γ(2z)."""
        test_values = [0.5, 0.7, 1.0, 1.3, 1.8]

        for z in test_values:
            gamma_z = gamma_safe(z)
            gamma_z_half = gamma_safe(z + 0.5)
            gamma_2z = gamma_safe(2 * z)

            left_side = gamma_z * gamma_z_half
            right_side = math.sqrt(math.pi) * (2 ** (1 - 2 * z)) * gamma_2z

            relative_error = abs(left_side - right_side) / abs(right_side)
            assert (
                relative_error < math_tolerance
            ), f"Duplication formula failed at z={z}: {left_side} ≠ {right_side}"


class TestGammaNumericalStability:
    """Test numerical stability and edge case handling."""

    @pytest.mark.numerical
    def test_pole_handling(self):
        """Test that poles (negative integers) are handled correctly."""
        # Γ(0), Γ(-1), Γ(-2), etc. should return +inf
        for n in [0, -1, -2, -3, -5]:
            result = gamma_safe(n)
            assert np.isinf(result) and result > 0, f"Γ({n}) should be +∞, got {result}"

    @pytest.mark.numerical
    def test_large_argument_stability(self):
        """Test stability for large arguments."""
        large_values = [50, 100, 150, 170]

        for x in large_values:
            result = gamma_safe(x)

            # Should be finite (may be very large) or controlled overflow
            if not np.isinf(result):
                assert np.isfinite(
                    result
                ), f"Γ({x}) should be finite or inf, got {result}"
                assert result > 0, f"Γ({x}) should be positive, got {result}"

    @pytest.mark.numerical
    def test_small_positive_values(self, numerical_tolerance):
        """Test behavior near zero from the positive side."""
        small_values = [1e-10, 1e-6, 1e-3, 0.01, 0.1]

        for x in small_values:
            result = gamma_safe(x)

            # Γ(x) → +∞ as x → 0+
            assert (
                np.isfinite(result) and result > 0
            ), f"Γ({x}) should be large and positive, got {result}"

            # Should be approximately 1/x for very small x
            if x < 0.001:
                expected_approx = 1.0 / x
                # Allow factor of 10 difference due to other terms
                assert (
                    0.1 * expected_approx < result < 10 * expected_approx
                ), f"Γ({x}) = {result} not approximately 1/x = {expected_approx}"

    @pytest.mark.numerical
    def test_negative_non_integer_values(self):
        """Test gamma function for negative non-integer values."""
        negative_values = [-0.5, -1.5, -2.3, -3.7]

        for x in negative_values:
            result = gamma_safe(x)

            # Should be finite (and typically negative for certain ranges)
            assert np.isfinite(result), f"Γ({x}) should be finite, got {result}"

            # Verify using reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
            gamma_complement = gamma_safe(1 - x)
            if np.isfinite(gamma_complement):
                product = result * gamma_complement
                expected = math.pi / math.sin(math.pi * x)

                if np.isfinite(expected):
                    relative_error = abs(product - expected) / abs(expected)
                    assert (
                        relative_error < 1e-8
                    ), f"Reflection formula violated for x={x}"


class TestGammaPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(st.floats(min_value=0.1, max_value=10.0))
    @pytest.mark.property
    def test_gamma_monotonicity_properties(self, x):
        """Test monotonicity properties of log(Γ(x))."""
        # log(Γ(x)) is convex for x > 0
        epsilon = 0.01

        if x + epsilon < 10.0:  # Stay within bounds
            log_gamma_x = math.log(gamma_safe(x))
            log_gamma_x_plus = math.log(gamma_safe(x + epsilon))
            log_gamma_x_minus = math.log(gamma_safe(max(0.1, x - epsilon)))

            # Second derivative should be positive (convexity)
            if x > 0.2:  # Avoid issues near zero
                second_deriv_approx = (
                    log_gamma_x_plus - 2 * log_gamma_x + log_gamma_x_minus
                ) / (epsilon**2)
                assert second_deriv_approx > -0.1, f"log(Γ) not convex at x={x}"

    @given(st.floats(min_value=0.1, max_value=5.0))
    @pytest.mark.property
    def test_gamma_scaling_invariance(self, x):
        """Test that Γ(x) behaves correctly under scaling."""
        # Γ(2x) should relate to Γ(x) via duplication formula
        if 2 * x <= 5.0:  # Keep in reasonable range
            gamma_x = gamma_safe(x)
            gamma_2x = gamma_safe(2 * x)
            gamma_x_half = gamma_safe(x + 0.5)

            # Duplication formula: Γ(x)Γ(x+1/2) = √π · 2^(1-2x) · Γ(2x)
            left = gamma_x * gamma_x_half
            right = math.sqrt(math.pi) * (2 ** (1 - 2 * x)) * gamma_2x

            if np.isfinite(left) and np.isfinite(right) and right != 0:
                relative_error = abs(left - right) / abs(right)
                assert relative_error < 1e-6, f"Duplication formula failed at x={x}"

    @given(arrays(np.float64, shape=(10,), elements=st.floats(0.1, 5.0)))
    @pytest.mark.property
    def test_gamma_array_consistency(self, x_array):
        """Test that gamma function is consistent between scalar and array inputs."""
        # Test individual vs vectorized computation
        individual_results = [gamma_safe(x) for x in x_array]

        # If we had vectorized gamma_safe, we'd test it here
        # For now, just verify individual computations are consistent
        for i, x in enumerate(x_array):
            result1 = gamma_safe(x)
            result2 = gamma_safe(x)  # Should be identical

            assert result1 == result2, f"Gamma function not deterministic at x={x}"
            assert result1 == individual_results[i], f"Inconsistent results at x={x}"


class TestGammaIntegration:
    """Test integration with other mathematical components."""

    @pytest.mark.integration
    def test_gamma_with_dimensional_measures(self, test_dimensions):
        """Test gamma function integration with dimensional measures."""
        # Ball volume formula: V_d = π^(d/2) / Γ(d/2 + 1)
        for d in test_dimensions:
            if d > 0:  # Skip d=0 special case
                try:
                    gamma_term = gamma_safe(d / 2 + 1)

                    # Gamma term should be positive and finite
                    assert (
                        np.isfinite(gamma_term) and gamma_term > 0
                    ), f"Invalid gamma term at d={d}: Γ({d/2 + 1}) = {gamma_term}"

                    # Volume calculation should work
                    pi_term = math.pi ** (d / 2)
                    if np.isfinite(pi_term) and gamma_term > 0:
                        volume = pi_term / gamma_term
                        assert (
                            np.isfinite(volume) and volume > 0
                        ), f"Invalid volume at d={d}: V = {volume}"

                except (OverflowError, ZeroDivisionError):
                    # Accept controlled failures for extreme dimensions
                    pass

    @pytest.mark.integration
    def test_gamma_ratio_stability(self):
        """Test the gamma ratio function for numerical stability."""
        # Test cases where direct computation might overflow but ratio is stable
        test_cases = [
            (50, 49),  # Large values
            (100, 99),  # Very large values
            (0.1, 0.2),  # Small values
            (1.5, 2.5),  # Fractional values
        ]

        for a, b in test_cases:
            try:
                ratio = gamma_ratio_safe(a, b)

                # Should be finite and positive
                assert (
                    np.isfinite(ratio) and ratio > 0
                ), f"Invalid ratio Γ({a})/Γ({b}) = {ratio}"

                # Cross-check with individual computation if feasible
                gamma_a = gamma_safe(a)
                gamma_b = gamma_safe(b)

                if np.isfinite(gamma_a) and np.isfinite(gamma_b) and gamma_b != 0:
                    direct_ratio = gamma_a / gamma_b
                    relative_error = abs(ratio - direct_ratio) / abs(direct_ratio)
                    assert (
                        relative_error < 1e-10
                    ), f"Ratio function inconsistent: {ratio} vs {direct_ratio}"

            except (OverflowError, ZeroDivisionError):
                # Some extreme cases may legitimately fail
                pass


class TestGammaPerformance:
    """Performance and benchmark tests."""

    @pytest.mark.benchmark
    def test_gamma_computation_speed(self, benchmark):
        """Benchmark gamma function computation speed."""
        test_values = np.linspace(0.1, 10, 1000)

        def compute_gamma_array():
            return [gamma_safe(x) for x in test_values]

        results = benchmark(compute_gamma_array)

        # Verify all results are reasonable
        assert len(results) == 1000
        assert all(np.isfinite(r) and r > 0 for r in results)

    @pytest.mark.benchmark
    def test_gamma_memory_usage(self):
        """Test memory usage patterns for large-scale gamma computations."""
        # This would typically use memory profiling tools
        # For now, just verify no memory leaks in repeated computation

        import gc

        initial_objects = len(gc.get_objects())

        # Perform many gamma computations
        for _ in range(1000):
            result = gamma_safe(np.random.uniform(0.1, 10))
            del result

        gc.collect()
        final_objects = len(gc.get_objects())

        # Should not have created excessive persistent objects
        object_growth = final_objects - initial_objects
        assert object_growth < 100, f"Possible memory leak: {object_growth} new objects"


class TestGammaRegression:
    """Regression tests to prevent breaking changes."""

    @pytest.mark.parametrize(
        "x,expected",
        [
            (1.0, 1.0),
            (2.0, 1.0),
            (3.0, 2.0),
            (4.0, 6.0),
            (0.5, 1.7724538509055159),  # √π
            (1.5, 0.8862269254527580),  # √π/2
        ],
    )
    def test_gamma_regression_values(self, x, expected):
        """Test that specific gamma values never change (regression prevention)."""
        actual = gamma_safe(x)
        # Use tight tolerance for regression testing
        assert (
            abs(actual - expected) < 1e-14
        ), f"Regression detected: Γ({x}) = {actual}, was {expected}"

    def test_gamma_api_stability(self):
        """Test that the gamma function API remains stable."""
        # Test that function signature and behavior are preserved

        # Should accept scalar input
        result_scalar = gamma_safe(2.5)
        assert isinstance(result_scalar, (int, float, np.number))

        # Should handle edge cases consistently
        assert np.isinf(gamma_safe(0))
        assert np.isinf(gamma_safe(-1))
        assert np.isfinite(gamma_safe(0.1))


# ===== Helper Functions for Complex Test Scenarios =====


def verify_gamma_fundamental_properties():
    """Comprehensive verification of all fundamental gamma properties."""
    errors = []

    try:
        # Test factorial property
        for n in range(1, 6):
            if abs(gamma_safe(n + 1) - math.factorial(n)) > 1e-12:
                errors.append(f"Factorial property failed at n={n}")

        # Test reflection formula
        for z in [0.3, 0.7]:
            left = gamma_safe(z) * gamma_safe(1 - z)
            right = math.pi / math.sin(math.pi * z)
            if abs(left - right) / right > 1e-10:
                errors.append(f"Reflection formula failed at z={z}")

        # Test recurrence relation
        for z in [0.5, 1.5, 2.3]:
            if abs(gamma_safe(z + 1) - z * gamma_safe(z)) > 1e-12:
                errors.append(f"Recurrence relation failed at z={z}")

    except Exception as e:
        errors.append(f"Exception in property verification: {e}")

    return errors


# Run comprehensive verification if this file is executed directly
if __name__ == "__main__":
    print("Running comprehensive gamma function verification...")

    if not GAMMA_AVAILABLE:
        print("❌ Gamma functions not available for testing")
        exit(1)

    errors = verify_gamma_fundamental_properties()

    if errors:
        print("❌ Gamma function verification failed:")
        for error in errors:
            print(f"  - {error}")
        exit(1)
    else:
        print("✅ All gamma function properties verified!")

    # Run the full test suite
    pytest.main([__file__, "-v"])
