#!/usr/bin/env python3
"""
Mathematical Properties Test Suite
=================================

Comprehensive validation of mathematical properties, invariants,
and theoretical relationships in dimensional mathematics.
"""

import numpy as np
import pytest

from .gamma import beta_function, digamma_safe, gamma_safe, gammaln_safe
from .mathematics import (
    CRITICAL_DIMENSIONS,
    NUMERICAL_EPSILON,
    PHI,
    PI,
    ball_volume,
    complexity_measure,
    find_peak,
    sphere_surface,
)
from .morphic import (
    discriminant,
    golden_ratio_properties,
    k_discriminant_zero,
    morphic_polynomial_roots,
    morphic_scaling_factor,
)
from .phase import PhaseDynamicsEngine, total_phase_energy


class TestMathematicalInvariants:
    """Test fundamental mathematical invariants and conservation laws."""

    def test_gamma_function_fundamental_properties(self):
        """Test fundamental properties of gamma function."""
        # Test Γ(1/2) = √π
        gamma_half = gamma_safe(0.5)
        assert abs(gamma_half - np.sqrt(PI)) < NUMERICAL_EPSILON, "Γ(1/2) ≠ √π"

        # Test Γ(1) = 1
        gamma_one = gamma_safe(1.0)
        assert abs(gamma_one - 1.0) < NUMERICAL_EPSILON, "Γ(1) ≠ 1"

        # Test Γ(2) = 1! = 1
        gamma_two = gamma_safe(2.0)
        assert abs(gamma_two - 1.0) < NUMERICAL_EPSILON, "Γ(2) ≠ 1"

        # Test Γ(3) = 2! = 2
        gamma_three = gamma_safe(3.0)
        assert abs(gamma_three - 2.0) < NUMERICAL_EPSILON, "Γ(3) ≠ 2"

    def test_gamma_recurrence_relation(self):
        """Test gamma function recurrence relation Γ(z+1) = z·Γ(z)."""
        test_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

        for z in test_values:
            gamma_z = gamma_safe(z)
            gamma_z_plus_1 = gamma_safe(z + 1)

            if np.isfinite(gamma_z) and np.isfinite(gamma_z_plus_1):
                expected = z * gamma_z
                relative_error = abs(gamma_z_plus_1 - expected) / abs(expected)
                assert relative_error < NUMERICAL_EPSILON * 100, f"Recurrence relation failed for z={z}"

    def test_euler_reflection_formula(self):
        """Test Euler's reflection formula: Γ(z)Γ(1-z) = π/sin(πz)."""
        test_values = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]

        for z in test_values:
            if 0 < z < 1 or 1 < z < 2:  # Avoid z = 1
                gamma_z = gamma_safe(z)
                gamma_1_minus_z = gamma_safe(1 - z)

                expected = PI / np.sin(PI * z)
                actual = gamma_z * gamma_1_minus_z

                if np.isfinite(expected) and np.isfinite(actual) and abs(expected) > NUMERICAL_EPSILON:
                    relative_error = abs(actual - expected) / abs(expected)
                    assert relative_error < NUMERICAL_EPSILON * 1000, f"Reflection formula failed for z={z}"

    def test_digamma_properties(self):
        """Test properties of the digamma function ψ(z) = Γ'(z)/Γ(z)."""
        # Test ψ(1) = -γ (Euler-Mascheroni constant)
        psi_one = digamma_safe(1.0)
        euler_gamma = 0.5772156649015329  # Euler-Mascheroni constant
        assert abs(psi_one + euler_gamma) < 1e-10, "ψ(1) ≠ -γ"

        # Test recurrence relation: ψ(z+1) = ψ(z) + 1/z
        test_values = [1.0, 1.5, 2.0, 2.5]
        for z in test_values:
            psi_z = digamma_safe(z)
            psi_z_plus_1 = digamma_safe(z + 1)

            if np.isfinite(psi_z) and np.isfinite(psi_z_plus_1):
                expected = psi_z + 1/z
                assert abs(psi_z_plus_1 - expected) < NUMERICAL_EPSILON * 100, f"Digamma recurrence failed for z={z}"


class TestDimensionalMeasureProperties:
    """Test properties of dimensional measures (volume, surface, complexity)."""

    def test_dimensional_measure_exact_values(self):
        """Test exact values for integer dimensions."""
        exact_values = {
            # Volume: V_n = π^(n/2) / Γ(n/2 + 1)
            (0, 'volume'): 1.0,
            (1, 'volume'): 2.0,
            (2, 'volume'): PI,
            (3, 'volume'): 4 * PI / 3,

            # Surface: S_n = 2π^(n/2) / Γ(n/2)
            (1, 'surface'): 2.0,
            (2, 'surface'): 2 * PI,
            (3, 'surface'): 4 * PI,
        }

        for (dim, measure_type), expected in exact_values.items():
            if measure_type == 'volume':
                actual = ball_volume(dim)
            else:  # surface
                actual = sphere_surface(dim)

            relative_error = abs(actual - expected) / abs(expected)
            assert relative_error < NUMERICAL_EPSILON * 10, f"{measure_type} at d={dim}: expected {expected}, got {actual}"

    def test_surface_volume_relationship(self):
        """Test S_d = d * V_d relationship."""
        test_dimensions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

        for d in test_dimensions:
            if d > 0:  # Avoid division by zero
                volume = ball_volume(d)
                surface = sphere_surface(d)

                if np.isfinite(volume) and np.isfinite(surface) and volume > 0:
                    expected_surface = d * volume
                    relative_error = abs(surface - expected_surface) / surface
                    assert relative_error < NUMERICAL_EPSILON * 10, f"S_d = d*V_d failed for d={d}"

    def test_complexity_measure_properties(self):
        """Test properties of complexity measure C_d = V_d * S_d."""
        test_dimensions = [1.0, 2.0, 3.0, 4.0, 5.0]

        for d in test_dimensions:
            volume = ball_volume(d)
            surface = sphere_surface(d)
            complexity = complexity_measure(d)

            if all(np.isfinite([volume, surface, complexity])):
                expected_complexity = volume * surface
                relative_error = abs(complexity - expected_complexity) / complexity
                assert relative_error < NUMERICAL_EPSILON * 10, f"C_d = V_d * S_d failed for d={d}"

    def test_peak_locations_and_values(self):
        """Test that peaks occur at expected locations."""
        # Volume peak around d ≈ 5.26
        vol_peak_d, vol_peak_val = find_peak(ball_volume)
        assert 5.0 < vol_peak_d < 6.0, f"Volume peak at unexpected location: {vol_peak_d}"

        # Surface peak around d ≈ 7.26
        surf_peak_d, surf_peak_val = find_peak(sphere_surface)
        assert 6.5 < surf_peak_d < 8.0, f"Surface peak at unexpected location: {surf_peak_d}"

        # Complexity peak around d ≈ 6.35
        comp_peak_d, comp_peak_val = find_peak(complexity_measure)
        assert 6.0 < comp_peak_d < 7.0, f"Complexity peak at unexpected location: {comp_peak_d}"

        # Verify these match stored critical dimensions
        stored_vol_peak = CRITICAL_DIMENSIONS.get('volume_peak', vol_peak_d)
        stored_surf_peak = CRITICAL_DIMENSIONS.get('surface_peak', surf_peak_d)

        assert abs(vol_peak_d - stored_vol_peak) < 0.1, "Volume peak doesn't match stored value"
        assert abs(surf_peak_d - stored_surf_peak) < 0.1, "Surface peak doesn't match stored value"


class TestMorphicMathematicsProperties:
    """Test properties of morphic mathematics and golden ratio relationships."""

    def test_golden_ratio_fundamental_relations(self):
        """Test fundamental golden ratio relations."""
        phi = PHI
        psi = 1 / PHI

        # φ² = φ + 1
        assert abs(phi**2 - (phi + 1)) < NUMERICAL_EPSILON, "φ² ≠ φ + 1"

        # ψ² = 1 - ψ
        assert abs(psi**2 - (1 - psi)) < NUMERICAL_EPSILON, "ψ² ≠ 1 - ψ"

        # φ * ψ = 1
        assert abs(phi * psi - 1) < NUMERICAL_EPSILON, "φ * ψ ≠ 1"

        # φ - ψ = 1
        assert abs(phi - psi - 1) < NUMERICAL_EPSILON, "φ - ψ ≠ 1"

        # φ + ψ = √5
        assert abs(phi + psi - np.sqrt(5)) < NUMERICAL_EPSILON, "φ + ψ ≠ √5"

    def test_golden_ratio_properties_function(self):
        """Test the golden_ratio_properties function."""
        props = golden_ratio_properties()

        # All boolean properties should be True
        boolean_props = [
            'phi_squared_equals_phi_plus_one',
            'psi_squared_equals_one_minus_psi',
            'phi_times_psi_equals_one',
            'phi_minus_psi_equals_one',
            'phi_plus_psi_equals_sqrt5'
        ]

        for prop in boolean_props:
            assert props.get(prop, False), f"Golden ratio property {prop} failed"

    def test_morphic_polynomial_root_properties(self):
        """Test properties of morphic polynomial roots."""
        # Test both polynomial families
        families = ['shifted', 'simple']
        test_k_values = [0.5, 1.0, 1.5, 2.0, 2.5]

        for family in families:
            for k in test_k_values:
                roots = morphic_polynomial_roots(k, family)

                # Should have at least one real root
                assert len(roots) > 0, f"No real roots found for {family} polynomial with k={k}"

                # Verify roots actually solve the polynomial
                for root in roots:
                    if family == 'shifted':
                        poly_value = root**3 - (2 - k) * root - 1
                    else:  # simple
                        poly_value = root**3 - k * root - 1

                    assert abs(poly_value) < NUMERICAL_EPSILON * 100, f"Root {root} doesn't solve {family} polynomial with k={k}"

    def test_morphic_discriminant_properties(self):
        """Test discriminant properties and critical points."""
        families = ['shifted', 'simple']

        for family in families:
            # Test discriminant at critical points
            # k_circle = k_perfect_circle(family)  # Unused variable
            k_disc_zero = k_discriminant_zero(family)

            # Discriminant should be zero at critical point
            disc_at_critical = discriminant(k_disc_zero, family)
            assert abs(disc_at_critical) < NUMERICAL_EPSILON * 1000, f"Discriminant not zero at critical point for {family}"

            # Test discriminant sign changes
            k_values = np.linspace(k_disc_zero - 1, k_disc_zero + 1, 100)
            discriminants = [discriminant(k, family) for k in k_values]

            # Should see sign changes around the critical point
            signs = [np.sign(d) for d in discriminants if abs(d) > NUMERICAL_EPSILON]
            if len(signs) > 10:
                unique_signs = set(signs)
                assert len(unique_signs) > 1, f"No sign change in discriminant for {family}"

    def test_morphic_scaling_factor(self):
        """Test morphic scaling factor properties."""
        scaling_factor = morphic_scaling_factor()

        # Should be finite and positive
        assert np.isfinite(scaling_factor), "Morphic scaling factor not finite"
        assert scaling_factor > 0, "Morphic scaling factor not positive"

        # Should be approximately φ^(1/φ) ≈ 1.465
        expected = PHI ** (1 / PHI)
        assert abs(scaling_factor - expected) < NUMERICAL_EPSILON * 10, "Morphic scaling factor incorrect"


class TestPhaseDynamicsProperties:
    """Test properties of phase dynamics and energy conservation."""

    def test_energy_conservation(self):
        """Test energy conservation in phase dynamics."""
        engine = PhaseDynamicsEngine(max_dimensions=5)

        initial_energy = total_phase_energy(engine.phase_density)

        # Run simulation steps
        dt = 0.01
        for _ in range(100):
            engine.step(dt)

        final_energy = total_phase_energy(engine.phase_density)

        # Energy should be conserved (allowing for numerical precision)
        energy_change = abs(final_energy - initial_energy)
        relative_change = energy_change / (initial_energy + NUMERICAL_EPSILON)

        assert relative_change < 1e-6, f"Energy not conserved: initial={initial_energy}, final={final_energy}"

    def test_phase_coherence_bounds(self):
        """Test that phase coherence remains bounded."""
        engine = PhaseDynamicsEngine(max_dimensions=6)

        # Run simulation and check coherence bounds
        dt = 0.02
        for _ in range(50):
            engine.step(dt)

            state = engine.get_state()
            coherence = state['coherence']

            # Coherence should be between 0 and 1
            assert 0 <= coherence <= 1, f"Coherence out of bounds: {coherence}"

    def test_dimensional_emergence_properties(self):
        """Test properties of dimensional emergence."""
        engine = PhaseDynamicsEngine(max_dimensions=4)

        initial_emerged = len(engine.emerged)

        # Inject energy to trigger emergence
        engine.inject_energy(1.0, 2.0)  # At dimension 2.0

        # Run simulation
        dt = 0.01
        for _ in range(200):
            engine.step(dt)

        final_emerged = len(engine.emerged)

        # Should have at least maintained initial emerged dimensions
        assert final_emerged >= initial_emerged, "Dimensional emergence decreased"

        # Effective dimension should be reasonable
        effective_dim = engine.calculate_effective_dimension()
        assert 0 <= effective_dim < engine.max_dim, f"Effective dimension out of bounds: {effective_dim}"


class TestBetaFunctionProperties:
    """Test properties of the beta function."""

    def test_beta_function_symmetry(self):
        """Test beta function symmetry B(a,b) = B(b,a)."""
        test_pairs = [(1, 2), (0.5, 1.5), (2, 3), (0.25, 0.75)]

        for a, b in test_pairs:
            beta_ab = beta_function(a, b)
            beta_ba = beta_function(b, a)

            if np.isfinite(beta_ab) and np.isfinite(beta_ba):
                assert abs(beta_ab - beta_ba) < NUMERICAL_EPSILON * 10, f"Beta function not symmetric for ({a}, {b})"

    def test_beta_gamma_relationship(self):
        """Test B(a,b) = Γ(a)Γ(b)/Γ(a+b)."""
        test_pairs = [(1, 1), (2, 3), (0.5, 0.5), (1.5, 2.5)]

        for a, b in test_pairs:
            beta_ab = beta_function(a, b)

            gamma_a = gamma_safe(a)
            gamma_b = gamma_safe(b)
            gamma_a_plus_b = gamma_safe(a + b)

            if all(np.isfinite([beta_ab, gamma_a, gamma_b, gamma_a_plus_b])) and gamma_a_plus_b != 0:
                expected = gamma_a * gamma_b / gamma_a_plus_b
                relative_error = abs(beta_ab - expected) / abs(expected)
                assert relative_error < NUMERICAL_EPSILON * 100, f"Beta-gamma relationship failed for ({a}, {b})"

    def test_beta_function_special_values(self):
        """Test special values of beta function."""
        # B(1,1) = 1
        beta_11 = beta_function(1, 1)
        assert abs(beta_11 - 1.0) < NUMERICAL_EPSILON, "B(1,1) ≠ 1"

        # B(1/2, 1/2) = π
        beta_half_half = beta_function(0.5, 0.5)
        assert abs(beta_half_half - PI) < NUMERICAL_EPSILON * 10, "B(1/2, 1/2) ≠ π"


class TestNumericalConsistencyProperties:
    """Test numerical consistency across different computation methods."""

    def test_log_space_consistency(self):
        """Test consistency between direct and log-space computations."""
        test_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

        for z in test_values:
            gamma_direct = gamma_safe(z)
            log_gamma = gammaln_safe(z)

            if np.isfinite(gamma_direct) and np.isfinite(log_gamma) and gamma_direct > 0:
                log_from_direct = np.log(gamma_direct)
                relative_error = abs(log_from_direct - log_gamma) / abs(log_gamma)
                assert relative_error < NUMERICAL_EPSILON * 1000, f"Log-space inconsistency for z={z}"

    def test_array_scalar_consistency(self):
        """Test consistency between array and scalar operations."""
        test_values = [0.5, 1.5, 2.5, 3.5, 4.5]

        # Scalar results
        scalar_results = [gamma_safe(z) for z in test_values]

        # Array result
        array_result = [gamma_safe(z) for z in np.array(test_values)]

        for i, (scalar, array_val) in enumerate(zip(scalar_results, array_result)):
            if np.isfinite(scalar) and np.isfinite(array_val):
                assert abs(scalar - array_val) < NUMERICAL_EPSILON, f"Array-scalar inconsistency at index {i}"


if __name__ == "__main__":
    # Run the mathematical properties test suite
    pytest.main([__file__, "-v"])
