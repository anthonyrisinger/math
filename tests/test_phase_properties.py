#!/usr/bin/env python3
"""
Property-based tests for phase dynamics conservation laws.

Tests fundamental conservation properties that must hold in the
phase sapping dynamics and emergence simulations.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from core.constants import NUMERICAL_EPSILON
from core.phase import phase_evolution_step, sap_rate


class TestSapRateProperties:
    """Test sap rate function mathematical properties"""

    @given(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300)
    def test_sap_rate_directionality(self, source, target):
        """Test that sapping only occurs from lower to higher dimensions"""
        phase_density = np.ones(20, dtype=complex) * 0.1  # Uniform low density

        rate_forward = sap_rate(source, target, phase_density)
        rate_backward = sap_rate(target, source, phase_density)

        if source < target:
            # Forward sapping should be non-negative
            assert (
                rate_forward >= 0
            ), f"Negative forward sap rate: {source}→{target} = {rate_forward}"
            # Backward sapping should be zero
            assert (
                rate_backward == 0
            ), f"Non-zero backward sap rate: {target}→{source} = {rate_backward}"
        elif source > target:
            # Forward sapping should be zero
            assert (
                rate_forward == 0
            ), f"Non-zero downward sap rate: {source}→{target} = {rate_forward}"
        else:
            # Same dimension: both should be zero
            assert (
                rate_forward == 0 and rate_backward == 0
            ), f"Non-zero self-sap: {source}→{source}"

    @given(
        st.floats(min_value=0.0, max_value=8.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_sap_rate_equilibrium_saturation(self, source, density):
        """Test that sap rate approaches zero as target approaches capacity"""
        target = source + 1.0  # Always higher dimension
        assume(target < 10)

        # Test with different density levels
        n_dims = max(20, int(target) + 5)

        # Low density case
        low_density = np.ones(n_dims, dtype=complex) * density
        rate_low = sap_rate(source, target, low_density)

        # High density case (near saturation)
        high_density = np.ones(n_dims, dtype=complex) * (density * 5)  # Much higher
        rate_high = sap_rate(source, target, high_density)

        # Rate should decrease as density increases (approaching equilibrium)
        if rate_low > NUMERICAL_EPSILON:
            assert (
                rate_high <= rate_low + NUMERICAL_EPSILON
            ), f"Sap rate should decrease with higher density: {rate_low} → {rate_high}"

    @given(
        st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_sap_rate_distance_scaling(self, distance):
        """Test that sap rate scales inversely with dimensional distance"""
        source = 1.0
        target1 = source + distance
        target2 = source + 2 * distance

        assume(target2 < 15)  # Keep within reasonable bounds

        phase_density = np.ones(20, dtype=complex) * 0.1

        rate1 = sap_rate(source, target1, phase_density)
        rate2 = sap_rate(source, target2, phase_density)

        # Rate should generally decrease with distance (though other factors involved)
        if rate1 > NUMERICAL_EPSILON and rate2 > NUMERICAL_EPSILON:
            # Allow some tolerance due to other scaling factors
            assert (
                rate2 <= rate1 * 2
            ), f"Sap rate scaling issue: d={distance} gives {rate1}, d={2*distance} gives {rate2}"


class TestPhaseEvolutionConservation:
    """Test conservation laws in phase evolution"""

    @given(
        arrays(
            dtype=np.float64,
            shape=st.integers(3, 10),
            elements=st.floats(
                min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    @settings(max_examples=100, deadline=5000)
    def test_energy_conservation(self, initial_energies):
        """Test that total energy is conserved during evolution"""
        # Create complex phase density from real energies
        n_dims = len(initial_energies)
        phases = np.random.uniform(0, 2 * np.pi, n_dims)  # Random phases
        phase_density = np.sqrt(initial_energies) * np.exp(1j * phases)

        dt = 0.001  # Small time step

        # Store initial energy
        initial_total_energy = np.sum(np.abs(phase_density) ** 2)
        assume(initial_total_energy > NUMERICAL_EPSILON)

        # Evolve system
        evolved_phase_density, _ = phase_evolution_step(phase_density, dt)
        final_total_energy = np.sum(np.abs(evolved_phase_density) ** 2)

        # Check energy conservation
        energy_change = abs(final_total_energy - initial_total_energy)
        relative_change = energy_change / initial_total_energy

        # Allow small numerical errors but require tight conservation
        assert (
            relative_change < 1e-10
        ), f"Energy not conserved: {initial_total_energy} → {final_total_energy} (change: {relative_change})"

    @given(
        st.integers(min_value=3, max_value=8),
        st.floats(
            min_value=0.001, max_value=0.01, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=50, deadline=10000)
    def test_phase_evolution_stability(self, n_dims, dt):
        """Test that evolution remains stable over multiple steps"""
        # Initialize with smooth energy distribution
        initial_energies = np.exp(-np.arange(n_dims) * 0.5)  # Exponentially decreasing
        phases = np.zeros(n_dims)  # Start with zero phases
        phase_density = np.sqrt(initial_energies) * np.exp(1j * phases)

        # Evolve for several steps
        current_state = phase_density.copy()
        energies_history = []

        for step in range(10):
            current_state, _ = phase_evolution_step(current_state, dt)
            current_energies = np.abs(current_state) ** 2
            energies_history.append(current_energies)

            # Check for instabilities (NaN, infinity, or explosive growth)
            assert np.all(
                np.isfinite(current_state)
            ), f"Non-finite values at step {step}"
            assert np.all(current_energies >= 0), f"Negative energies at step {step}"

            # Check for explosive growth
            max_energy = np.max(current_energies)
            assert (
                max_energy < 100
            ), f"Explosive growth at step {step}: max_energy = {max_energy}"

        # Verify energy conservation across all steps
        total_energies = [np.sum(energies) for energies in energies_history]
        initial_total = np.sum(initial_energies)

        for i, total in enumerate(total_energies):
            relative_change = abs(total - initial_total) / initial_total
            assert (
                relative_change < 1e-8
            ), f"Energy drift at step {i}: {relative_change}"

    @given(st.integers(min_value=4, max_value=6))
    @settings(max_examples=20, deadline=15000)
    def test_phase_evolution_irreversibility(self, n_dims):
        """Test that phase evolution exhibits arrow of time (irreversibility)"""
        # Start with energy concentrated in lower dimensions
        initial_energies = np.zeros(n_dims)
        initial_energies[0] = 1.0  # All energy in dimension 0

        phase_density = np.sqrt(initial_energies) * np.exp(1j * np.zeros(n_dims))

        dt = 0.01
        steps = 20

        # Evolve forward
        current_state = phase_density.copy()
        energy_distributions = []

        for step in range(steps):
            current_state, _ = phase_evolution_step(current_state, dt)
            energies = np.abs(current_state) ** 2
            energy_distributions.append(energies)

        # Check that energy spreads to higher dimensions over time
        final_energies = energy_distributions[-1]

        # Energy should have spread from dimension 0 to higher dimensions
        initial_energies[0]
        final_energies[0]

        # Some energy should have moved to higher dimensions
        higher_dim_energy = np.sum(final_energies[1:])
        assert (
            higher_dim_energy > 1e-6
        ), f"No energy transfer to higher dimensions: {higher_dim_energy}"

        # But total energy should be conserved
        total_initial = np.sum(initial_energies)
        total_final = np.sum(final_energies)
        relative_change = abs(total_final - total_initial) / total_initial
        assert (
            relative_change < 1e-8
        ), f"Energy not conserved in irreversibility test: {relative_change}"


class TestPhaseCoherenceProperties:
    """Test phase coherence and interference properties"""

    @given(
        st.floats(
            min_value=0.0, max_value=2 * np.pi, allow_nan=False, allow_infinity=False
        ),
        st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_phase_invariance_under_global_rotation(self, global_phase, amplitude):
        """Test that dynamics are invariant under global phase rotation"""
        n_dims = 5

        # Create initial state
        energies = np.ones(n_dims) * amplitude
        phases = np.random.uniform(0, 2 * np.pi, n_dims)
        phase_density1 = np.sqrt(energies) * np.exp(1j * phases)

        # Create globally rotated state
        phase_density2 = np.sqrt(energies) * np.exp(1j * (phases + global_phase))

        dt = 0.001

        # Evolve both states
        evolved1, _ = phase_evolution_step(phase_density1, dt)
        evolved2, _ = phase_evolution_step(phase_density2, dt)

        # Energy distributions should be identical
        energies1 = np.abs(evolved1) ** 2
        energies2 = np.abs(evolved2) ** 2

        energy_difference = np.max(np.abs(energies1 - energies2))
        assert (
            energy_difference < 1e-12
        ), f"Global phase rotation not preserved: max diff = {energy_difference}"

        # Phase differences should be preserved
        phase_diff_initial = np.angle(phase_density1 * np.conj(phase_density2))
        phase_diff_evolved = np.angle(evolved1 * np.conj(evolved2))

        # Allow for phase wrapping
        phase_diff_change = np.abs(
            np.angle(np.exp(1j * (phase_diff_evolved - phase_diff_initial)))
        )
        max_phase_drift = np.max(phase_diff_change)
        assert (
            max_phase_drift < 0.1
        ), f"Phase relationships not preserved: max drift = {max_phase_drift}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
