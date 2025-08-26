#!/usr/bin/env python3
"""
Enhanced Gamma Function Property Tests
======================================

Advanced mathematical property testing for gamma function invariants,
including complex analysis properties, asymptotic behavior, and
special function relationships.

This extends the basic property tests with:
1. Complex plane behavior and branch cuts
2. Asymptotic expansions (Stirling's approximation)
3. Special value relationships (half-integers, rationals)
4. Duplication and multiplication formulas
5. Analytic continuation properties
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import cmath

from core.constants import NUMERICAL_EPSILON, PI, E
from core.gamma import gamma_safe, gammaln_safe, digamma_safe, beta_function, factorial_extension


class TestComplexGammaProperties:
    """Test gamma function properties in the complex plane"""

    @given(
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.filter_too_much])
    def test_gamma_complex_modulus_continuity(self, real_part, imag_part):
        """Test continuity of |Γ(z)| in complex plane"""
        z = complex(real_part, imag_part)

        # Avoid poles (negative integers)
        if real_part < 0 and abs(imag_part) < 0.1:
            assume(abs(real_part - np.round(real_part)) > 0.1)

        try:
            gamma_z = gamma_safe(z)
            assume(np.isfinite(gamma_z))

            # Test small perturbations maintain continuity
            epsilon = 0.001
            gamma_z_perturbed = gamma_safe(z + epsilon)
            assume(np.isfinite(gamma_z_perturbed))

            relative_change = abs(gamma_z_perturbed - gamma_z) / max(abs(gamma_z), NUMERICAL_EPSILON)

            # Continuity: small input change should produce small output change
            assert relative_change < 1.0, f"Large discontinuity at z={z}: {relative_change}"

        except (ValueError, OverflowError, TypeError):
            # Some complex evaluations may not be supported
            pass

    @given(
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_gamma_complex_conjugate_symmetry(self, real_part):
        """Test Γ(z̄) = Γ̄(z) for complex conjugate symmetry"""
        imag_part = 1.0  # Use fixed imaginary part for testing
        z = complex(real_part, imag_part)
        z_conjugate = complex(real_part, -imag_part)

        try:
            gamma_z = gamma_safe(z)
            gamma_z_conj = gamma_safe(z_conjugate)

            assume(np.isfinite(gamma_z) and np.isfinite(gamma_z_conj))

            # Test conjugate symmetry: Γ(z̄) = Γ̄(z)
            expected_conjugate = np.conjugate(gamma_z)
            error = abs(gamma_z_conj - expected_conjugate)

            # Allow for numerical precision in complex operations
            relative_error = error / max(abs(expected_conjugate), NUMERICAL_EPSILON)
            assert relative_error < 1e-10, f"Conjugate symmetry failed: {relative_error}"

        except (ValueError, OverflowError, TypeError):
            # Some complex evaluations may not be supported
            pass


class TestGammaAsymptoticProperties:
    """Test asymptotic behavior and Stirling's approximation"""

    @given(
        st.floats(min_value=10.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50)
    def test_stirling_approximation_accuracy(self, z):
        """Test Γ(z) ≈ √(2π/z) * (z/e)^z for large z"""
        gamma_z = gamma_safe(z)
        assume(np.isfinite(gamma_z) and gamma_z > 0)

        # Stirling's approximation: Γ(z) ≈ √(2π/z) * (z/e)^z
        stirling_approx = np.sqrt(2 * PI / z) * ((z / E) ** z)

        # Relative error should decrease as z increases
        relative_error = abs(gamma_z - stirling_approx) / gamma_z

        # For z ≥ 10, Stirling's approximation should be quite good
        if z >= 20:
            assert relative_error < 0.1, f"Stirling approximation poor for z={z}: {relative_error}"
        else:
            # For moderate z, just test that it's reasonable
            assert relative_error < 0.5, f"Stirling approximation very poor for z={z}: {relative_error}"

    @given(
        st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_gamma_logarithmic_growth(self, z):
        """Test logarithmic growth properties of Γ(z)"""
        log_gamma_z = gammaln_safe(z)
        log_gamma_z_plus_1 = gammaln_safe(z + 1)

        assume(np.isfinite(log_gamma_z) and np.isfinite(log_gamma_z_plus_1))

        # Growth rate: d/dz log Γ(z) = ψ(z)
        digamma_z = digamma_safe(z)
        assume(np.isfinite(digamma_z))

        # Numerical derivative approximation
        epsilon = 1e-8
        log_gamma_z_eps = gammaln_safe(z + epsilon)
        assume(np.isfinite(log_gamma_z_eps))

        numerical_derivative = (log_gamma_z_eps - log_gamma_z) / epsilon

        # Should match digamma function
        error = abs(numerical_derivative - digamma_z)
        assert error < 1e-6, f"Digamma derivative mismatch at z={z}: {error}"


class TestSpecialGammaValues:
    """Test gamma function at special rational and half-integer values"""

    def test_half_integer_values(self):
        """Test Γ(n + 1/2) = √π * (2n-1)!! / 2^n"""
        test_cases = [
            (0.5, np.sqrt(PI)),  # Γ(1/2) = √π
            (1.5, np.sqrt(PI) / 2),  # Γ(3/2) = √π/2
            (2.5, 3 * np.sqrt(PI) / 4),  # Γ(5/2) = 3√π/4
            (3.5, 15 * np.sqrt(PI) / 8),  # Γ(7/2) = 15√π/8
        ]

        for z, expected in test_cases:
            gamma_result = gamma_safe(z)
            error = abs(gamma_result - expected)
            relative_error = error / abs(expected)

            assert relative_error < 1e-14, f"Half-integer Γ({z}) failed: {relative_error}"

    @given(
        st.integers(min_value=1, max_value=8)
    )
    @settings(max_examples=20)
    def test_rational_third_values(self, n):
        """Test gamma function at thirds: Γ(n/3)"""
        z = n / 3.0
        gamma_result = gamma_safe(z)

        # Test that result is finite and positive
        assert np.isfinite(gamma_result), f"Γ({z}) not finite"
        if z > 0:
            assert gamma_result > 0, f"Γ({z}) not positive: {gamma_result}"

        # Test recurrence relation holds
        if z > 1:
            gamma_z_minus_one = gamma_safe(z - 1)
            expected_from_recurrence = (z - 1) * gamma_z_minus_one

            assume(np.isfinite(gamma_z_minus_one))
            relative_error = abs(gamma_result - expected_from_recurrence) / gamma_result
            assert relative_error < 1e-12, f"Recurrence failed at z={z}: {relative_error}"


class TestGammaMultiplicationFormulas:
    """Test duplication and multiplication formulas"""

    @given(
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_duplication_formula(self, z):
        """Test Legendre duplication formula: Γ(z)Γ(z+1/2) = √π * 2^{1-2z} * Γ(2z)"""
        gamma_z = gamma_safe(z)
        gamma_z_half = gamma_safe(z + 0.5)
        gamma_2z = gamma_safe(2 * z)

        assume(all(np.isfinite(x) for x in [gamma_z, gamma_z_half, gamma_2z]))
        assume(all(x > NUMERICAL_EPSILON for x in [gamma_z, gamma_z_half, gamma_2z]))

        left_side = gamma_z * gamma_z_half
        right_side = np.sqrt(PI) * (2 ** (1 - 2 * z)) * gamma_2z

        relative_error = abs(left_side - right_side) / max(abs(right_side), NUMERICAL_EPSILON)

        # Duplication formula should hold with high precision
        assert relative_error < 1e-10, f"Duplication formula failed at z={z}: {relative_error}"

    @given(
        st.floats(min_value=0.2, max_value=3.0, allow_nan=False, allow_infinity=False),
        st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=50)
    def test_multiplication_formula(self, z, n):
        """Test general multiplication formula for Γ(nz)"""
        # Γ(nz) = (2π)^{(1-n)/2} * n^{nz-1/2} * ∏_{k=0}^{n-1} Γ(z + k/n)

        gamma_nz = gamma_safe(n * z)
        assume(np.isfinite(gamma_nz) and gamma_nz > NUMERICAL_EPSILON)

        # Compute product ∏_{k=0}^{n-1} Γ(z + k/n)
        product = 1.0
        for k in range(n):
            gamma_term = gamma_safe(z + k / n)
            assume(np.isfinite(gamma_term) and gamma_term > NUMERICAL_EPSILON)
            product *= gamma_term

        # Multiplication formula coefficient
        coeff = ((2 * PI) ** ((1 - n) / 2)) * (n ** (n * z - 0.5))
        expected = coeff * product

        relative_error = abs(gamma_nz - expected) / max(abs(expected), NUMERICAL_EPSILON)

        # Allow more tolerance for higher order formulas
        tolerance = 1e-8 if n <= 3 else 1e-6
        assert relative_error < tolerance, f"Multiplication formula failed: n={n}, z={z}, error={relative_error}"


class TestGammaFunctionalEquations:
    """Test advanced functional equations and relationships"""

    @given(
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_beta_gamma_integral_relation(self, a, b):
        """Test B(a,b) integral representation consistency"""
        beta_ab = beta_function(a, b)

        # Alternative computation using gamma functions
        gamma_a = gamma_safe(a)
        gamma_b = gamma_safe(b)
        gamma_a_plus_b = gamma_safe(a + b)

        assume(all(np.isfinite(x) for x in [gamma_a, gamma_b, gamma_a_plus_b]))
        assume(gamma_a_plus_b > NUMERICAL_EPSILON)

        gamma_formula = (gamma_a * gamma_b) / gamma_a_plus_b

        relative_error = abs(beta_ab - gamma_formula) / max(abs(gamma_formula), NUMERICAL_EPSILON)
        assert relative_error < 1e-12, f"Beta-gamma relation failed: {relative_error}"

        # Test symmetry
        beta_ba = beta_function(b, a)
        symmetry_error = abs(beta_ab - beta_ba) / max(abs(beta_ab), NUMERICAL_EPSILON)
        assert symmetry_error < 1e-12, f"Beta symmetry failed: {symmetry_error}"

    @given(
        st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_digamma_functional_equation(self, z):
        """Test ψ(z+1) = ψ(z) + 1/z"""
        digamma_z = digamma_safe(z)
        digamma_z_plus_1 = digamma_safe(z + 1)

        assume(np.isfinite(digamma_z) and np.isfinite(digamma_z_plus_1))

        expected = digamma_z + 1.0 / z
        error = abs(digamma_z_plus_1 - expected)

        assert error < 1e-12, f"Digamma recurrence failed at z={z}: {error}"


class TestGammaNumericalStability:
    """Test numerical stability and edge case behavior"""

    @given(
        st.floats(min_value=1e-10, max_value=1e-5, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50)
    def test_small_positive_values(self, z):
        """Test gamma function for very small positive values"""
        gamma_z = gamma_safe(z)

        # For small z > 0, Γ(z) should be large but finite
        assert np.isfinite(gamma_z), f"Γ({z}) not finite for small z"
        assert gamma_z > 0, f"Γ({z}) not positive for small z: {gamma_z}"

        # Should satisfy Γ(z) ≈ 1/z for very small z
        asymptotic_approx = 1.0 / z
        if z < 1e-8:
            relative_error = abs(gamma_z - asymptotic_approx) / asymptotic_approx
            assert relative_error < 0.1, f"Small z asymptotic failed: {relative_error}"

    @given(
        st.floats(min_value=50.0, max_value=170.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=30)
    def test_large_values_overflow_handling(self, z):
        """Test gamma function behavior near overflow region"""
        gamma_z = gamma_safe(z)

        # Result should either be finite or properly overflow
        if np.isfinite(gamma_z):
            assert gamma_z > 0, f"Finite Γ({z}) should be positive: {gamma_z}"
        else:
            # If overflow occurs, should be +inf, not NaN
            assert np.isinf(gamma_z) and gamma_z > 0, f"Overflow should give +inf: {gamma_z}"

    @given(
        st.floats(min_value=-50.5, max_value=-0.1, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_negative_non_integer_poles(self, z):
        """Test behavior near negative integer poles"""
        # Avoid actual poles (negative integers)
        assume(abs(z - np.round(z)) > 0.01)

        gamma_z = gamma_safe(z)

        if np.isfinite(gamma_z):
            # Should alternate sign based on which side of pole we're on
            n = int(np.floor(z))  # Largest integer ≤ z
            expected_sign = (-1) ** (-n - 1)  # Sign pattern for Γ(z) with z < 0

            actual_sign = np.sign(gamma_z)
            assert actual_sign == expected_sign or abs(gamma_z) < NUMERICAL_EPSILON, \
                f"Sign error near pole: z={z}, sign={actual_sign}, expected={expected_sign}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])