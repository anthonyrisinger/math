#!/usr/bin/env python3
"""
Property-based tests for gamma function mathematical invariants.

Tests fundamental mathematical properties that must hold for any
valid gamma function implementation across all parameter ranges.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from core.constants import NUMERICAL_EPSILON
from core.gamma import beta_function, factorial_extension, gamma_safe, gammaln_safe


class TestGammaRecurrenceRelations:
    """Test gamma function recurrence relations: Γ(z+1) = z·Γ(z)"""

    @given(
        st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=1000)
    def test_gamma_recurrence_positive(self, z):
        """Test Γ(z+1) = z·Γ(z) for positive z"""
        gamma_z = gamma_safe(z)
        gamma_z_plus_1 = gamma_safe(z + 1)

        # Skip if either value is infinite (overflow case)
        assume(np.isfinite(gamma_z) and np.isfinite(gamma_z_plus_1))

        expected = z * gamma_z
        relative_error = abs(gamma_z_plus_1 - expected) / max(
            abs(expected), NUMERICAL_EPSILON
        )

        assert (
            relative_error < 1e-12
        ), f"Γ({z}+1) ≠ {z}·Γ({z}): {gamma_z_plus_1} ≠ {expected}"

    @given(
        st.floats(
            min_value=-49.9, max_value=-0.1, allow_nan=False, allow_infinity=False
        )
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.filter_too_much])
    def test_gamma_recurrence_negative(self, z):
        """Test Γ(z+1) = z·Γ(z) for negative z (avoiding poles)"""
        # Avoid negative integers (poles)
        assume(abs(z - np.round(z)) > 0.1)

        gamma_z = gamma_safe(z)
        gamma_z_plus_1 = gamma_safe(z + 1)

        # Skip if either value is infinite
        assume(np.isfinite(gamma_z) and np.isfinite(gamma_z_plus_1))
        assume(abs(gamma_z) > NUMERICAL_EPSILON)  # Avoid division by tiny numbers

        expected = z * gamma_z
        relative_error = abs(gamma_z_plus_1 - expected) / max(
            abs(expected), NUMERICAL_EPSILON
        )

        assert (
            relative_error < 1e-10
        ), f"Γ({z}+1) ≠ {z}·Γ({z}): {gamma_z_plus_1} ≠ {expected}"

    @given(
        arrays(
            dtype=np.float64,
            shape=st.integers(1, 10),
            elements=st.floats(
                min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    @settings(max_examples=200)
    def test_gamma_recurrence_vectorized(self, z_array):
        """Test recurrence relation holds for vectorized inputs"""
        gamma_z = gamma_safe(z_array)
        gamma_z_plus_1 = gamma_safe(z_array + 1)

        # Filter out infinite values
        finite_mask = np.isfinite(gamma_z) & np.isfinite(gamma_z_plus_1)
        assume(np.any(finite_mask))

        z_finite = z_array[finite_mask]
        gamma_z_finite = gamma_z[finite_mask]
        gamma_z_plus_1_finite = gamma_z_plus_1[finite_mask]

        expected = z_finite * gamma_z_finite
        relative_errors = np.abs(gamma_z_plus_1_finite - expected) / np.maximum(
            np.abs(expected), NUMERICAL_EPSILON
        )

        assert np.all(
            relative_errors < 1e-12
        ), f"Vectorized recurrence failed: max error = {np.max(relative_errors)}"


class TestGammaSymmetryProperties:
    """Test gamma function symmetry and reflection properties"""

    @given(
        st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=500)
    def test_reflection_formula(self, z):
        """Test Γ(z)Γ(1-z) = π/sin(πz) for 0 < z < 1"""
        gamma_z = gamma_safe(z)
        gamma_1_minus_z = gamma_safe(1 - z)

        assume(np.isfinite(gamma_z) and np.isfinite(gamma_1_minus_z))

        product = gamma_z * gamma_1_minus_z
        expected = np.pi / np.sin(np.pi * z)

        relative_error = abs(product - expected) / max(abs(expected), NUMERICAL_EPSILON)
        assert (
            relative_error < 1e-10
        ), f"Reflection formula failed for z={z}: {product} ≠ {expected}"

    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=50)
    def test_integer_factorial_property(self, n):
        """Test Γ(n+1) = n! for positive integers"""
        gamma_result = gamma_safe(n + 1)
        factorial_result = factorial_extension(n)

        # Direct factorial calculation for verification
        import math

        expected = math.factorial(n)

        assert (
            abs(gamma_result - expected) < NUMERICAL_EPSILON
        ), f"Γ({n+1}) ≠ {n}!: {gamma_result} ≠ {expected}"
        assert (
            abs(factorial_result - expected) < NUMERICAL_EPSILON
        ), f"factorial_extension({n}) ≠ {n}!: {factorial_result} ≠ {expected}"


class TestGammaLogSpace:
    """Test log-space gamma function properties"""

    @given(
        st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=500)
    def test_log_gamma_consistency(self, z):
        """Test log(Γ(z)) = gammaln(z) consistency"""
        gamma_z = gamma_safe(z)
        log_gamma_z = gammaln_safe(z)

        if np.isfinite(gamma_z) and gamma_z > 0:
            expected_log = np.log(gamma_z)
            error = abs(log_gamma_z - expected_log)
            assert (
                error < 1e-12
            ), f"log(Γ({z})) inconsistent: {log_gamma_z} ≠ {expected_log}"

    @given(
        st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=500)
    def test_log_gamma_recurrence(self, z):
        """Test log(Γ(z+1)) = log(z) + log(Γ(z))"""
        log_gamma_z = gammaln_safe(z)
        log_gamma_z_plus_1 = gammaln_safe(z + 1)

        assume(np.isfinite(log_gamma_z) and np.isfinite(log_gamma_z_plus_1))
        assume(z > NUMERICAL_EPSILON)  # Avoid log(0)

        expected = np.log(z) + log_gamma_z
        error = abs(log_gamma_z_plus_1 - expected)
        assert (
            error < 1e-12
        ), f"Log-gamma recurrence failed for z={z}: {log_gamma_z_plus_1} ≠ {expected}"


class TestBetaFunctionProperties:
    """Test beta function mathematical properties"""

    @given(
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=500)
    def test_beta_symmetry(self, a, b):
        """Test B(a,b) = B(b,a)"""
        beta_ab = beta_function(a, b)
        beta_ba = beta_function(b, a)

        assume(np.isfinite(beta_ab) and np.isfinite(beta_ba))

        relative_error = abs(beta_ab - beta_ba) / max(abs(beta_ab), NUMERICAL_EPSILON)
        assert relative_error < 1e-12, f"Beta symmetry failed: B({a},{b}) ≠ B({b},{a})"

    @given(
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300)
    def test_beta_gamma_relation(self, a, b):
        """Test B(a,b) = Γ(a)Γ(b)/Γ(a+b)"""
        beta_ab = beta_function(a, b)

        gamma_a = gamma_safe(a)
        gamma_b = gamma_safe(b)
        gamma_a_plus_b = gamma_safe(a + b)

        assume(
            np.isfinite(gamma_a)
            and np.isfinite(gamma_b)
            and np.isfinite(gamma_a_plus_b)
        )
        assume(gamma_a_plus_b > NUMERICAL_EPSILON)  # Avoid division by zero

        expected = (gamma_a * gamma_b) / gamma_a_plus_b
        relative_error = abs(beta_ab - expected) / max(abs(expected), NUMERICAL_EPSILON)

        assert (
            relative_error < 1e-10
        ), f"Beta-gamma relation failed: B({a},{b}) = {beta_ab} ≠ {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
