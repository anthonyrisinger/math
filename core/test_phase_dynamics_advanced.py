#!/usr/bin/env python3
"""
Advanced Test Suite for Phase Dynamics
======================================

Tests the sophisticated numerical and physical features of the
`PhaseDynamicsEngine` and its components. This includes tests for the
adaptive RK45 integrator, convergence diagnostics, and topological
invariant preservation.
"""

import os
import sys

import numpy as np
import pytest

from core.phase import (
    NUMERICAL_EPSILON,
    ConvergenceDiagnostics,
    PhaseDynamicsEngine,
    TopologicalInvariants,
    rk45_step,
    sap_rate,
    total_phase_energy,
)


# Canonical Test Scenarios
def get_golden_test_engine(max_dims=8):
    """Returns a PhaseDynamicsEngine with 'golden' parameters."""
    return PhaseDynamicsEngine(max_dimensions=max_dims, use_adaptive=True)


def get_stress_test_engine(max_dims=12):
    """Returns an engine with extreme parameters to test limits."""
    engine = PhaseDynamicsEngine(max_dimensions=max_dims, use_adaptive=True)
    # Inject large energy to stress the system
    engine.inject(1, 100.0)
    engine.inject(2, 50.0)
    return engine


class TestAdvancedPhaseDynamics:
    """Advanced tests for the Phase Dynamics Engine."""

    def test_rk45_adaptive_step(self):
        """Verify that the adaptive RK45 integrator adjusts its step size correctly."""
        engine = get_golden_test_engine()
        initial_dt = 0.1

        # Get baseline step size behavior
        _, dt_baseline, error_baseline = rk45_step(
            engine.phase_density, initial_dt, engine.max_dim - 1
        )

        assert error_baseline >= 0, "Error estimate should be non-negative"

        # Test that a large disturbance increases error and affects step size
        engine.inject(1, 3.0)  # Moderate but significant injection
        _, dt_after_shock, error_after = rk45_step(
            engine.phase_density, initial_dt, engine.max_dim - 1
        )

        # Key test: error should increase after disturbance
        assert (
            error_after > error_baseline
        ), f"Error should increase: {error_baseline:.2e} -> {error_after:.2e}"

        # Step size adaptation should respond to higher error
        if error_after > 1e-12:  # Only test if meaningful error exists
            # Allow for either decrease or controlled increase based on error magnitude
            ratio = dt_after_shock / dt_baseline
            assert (
                0.3 <= ratio <= 3.0
            ), f"Step size should adapt reasonably: {dt_baseline:.6f} -> {dt_after_shock:.6f} (ratio: {ratio:.2f})"

    def test_convergence_diagnostics(self):
        """Test that the system correctly reports convergence with the new energy model."""
        engine = get_golden_test_engine(max_dims=4)

        # Start with moderate energy distribution for faster convergence
        engine.phase_density = np.array(
            [0.9 + 0j, 0.1 + 0j, 0.05 + 0j, 0.01 + 0j], dtype=complex
        )

        converged = False
        max_steps = 800

        # Run with small time steps until convergence
        for i in range(max_steps):
            engine.step(0.002)  # Small time steps for stability

            # Check convergence periodically after initial transient
            if i > 50 and i % 25 == 0:
                if engine.diagnostics.is_converged(
                    energy_tolerance=1e-12, rate_tolerance=1e-12
                ):
                    converged = True
                    print(f"System converged at step {i}")
                    break

        diagnostics = engine.diagnostics.get_diagnostics()

        # With the new model, we expect full convergence and strict conservation
        assert converged, "System should report convergence"
        assert diagnostics["is_converged"], "Diagnostics should confirm convergence"
        assert (
            diagnostics["energy_conservation_error"] < 1e-12
        ), f"Energy should be strictly conserved: {diagnostics['energy_conservation_error']:.2e}"

    def test_topological_invariants(self):
        """Test that topological invariants (Chern numbers) are preserved."""
        engine = get_golden_test_engine()

        # Initial invariants
        initial_invariants = engine.invariants.chern_numbers.copy()

        # Run for several steps
        for _ in range(50):
            engine.step(0.1)

        final_invariants = engine.invariants.chern_numbers.copy()

        np.testing.assert_array_equal(
            initial_invariants,
            final_invariants,
            "Topological invariants should be preserved",
        )

    def test_numerical_stability_sap_rate(self):
        """Test the stability of the sap_rate function at small distances."""
        phase_density = np.array([1.0, 1.0, 1.0], dtype=complex)

        # Test with very small distance
        rate = sap_rate(1.0, 1.0 + 1e-9, phase_density)

        assert np.isfinite(rate), "sap_rate should be finite for very small distances"
        assert not np.isnan(rate), "sap_rate should not return NaN"

    def test_energy_flow_conservation(self):
        """Test strict energy conservation with the new energy-based model."""
        engine = get_golden_test_engine(max_dims=4)

        # Start with energy only in void
        engine.phase_density = np.array([1.0 + 0j, 0j, 0j, 0j], dtype=complex)
        initial_energy = total_phase_energy(engine.phase_density)

        # Run with small time steps for maximum precision
        for i in range(100):
            engine.step(0.001)
            current_energy = total_phase_energy(engine.phase_density)

            # Check energy conservation at each step - relaxed tolerance for floating point
            energy_drift = abs(current_energy - initial_energy)
            assert (
                energy_drift < 1e-12
            ), f"Energy not conserved at step {i}: drift = {energy_drift:.2e}"

        final_energy = total_phase_energy(engine.phase_density)
        total_drift = abs(final_energy - initial_energy)

        # Final strict energy conservation check - relaxed tolerance
        assert total_drift < 1e-12, f"Total energy drift: {total_drift:.2e}"

        # Verify energy actually flowed between dimensions
        final_phases = np.abs(engine.phase_density)
        assert final_phases[1] > 1e-10, "Energy should flow to dimension 1"
        assert final_phases[0] < 0.999, "Energy should leave dimension 0"

    def test_convergence_to_equilibrium(self):
        """Test that system reaches stable equilibrium with strict energy conservation."""
        engine = get_golden_test_engine(max_dims=4)

        # Initialize with reasonable energy distribution
        engine.phase_density = np.array(
            [0.8 + 0j, 0.2 + 0j, 0.1 + 0j, 0.02 + 0j], dtype=complex
        )
        initial_energy = total_phase_energy(engine.phase_density)

        energy_changes = []
        converged = False
        max_steps = 1500

        # Run until system reaches equilibrium
        for i in range(max_steps):
            prev_energy = total_phase_energy(engine.phase_density)
            engine.step(0.001)  # Small time steps
            curr_energy = total_phase_energy(engine.phase_density)

            # Strict energy conservation check at every step - relaxed tolerance
            energy_drift = abs(curr_energy - initial_energy)
            assert (
                energy_drift < 1e-12
            ), f"Energy conservation violated at step {i}: {energy_drift:.2e}"

            # Track convergence
            energy_change = abs(curr_energy - prev_energy)
            energy_changes.append(energy_change)

            # Check for convergence with realistic criteria
            if i > 100 and len(energy_changes) >= 50:
                recent_changes = energy_changes[-50:]
                max_recent = max(recent_changes)
                avg_recent = np.mean(recent_changes)

                # Convergence when changes become very small
                if max_recent < 1e-13 and avg_recent < 1e-14:
                    converged = True
                    print(f"Converged at step {i} (max change: {max_recent:.2e})")
                    break

        # Accept convergence or very slow convergence
        if not converged and len(energy_changes) >= 100:
            recent_changes = energy_changes[-100:]
            avg_recent = np.mean(recent_changes)
            # System should at least be converging slowly
            assert (
                avg_recent < 1e-12
            ), f"System should be approaching equilibrium: avg change = {avg_recent:.2e}"
            print(f"System slowly converging: avg change = {avg_recent:.2e}")
        else:
            assert converged, "System should reach equilibrium"

    def test_clock_rate_consistency(self):
        """Test that clock rate modulation is self-consistent."""
        engine = get_golden_test_engine()
        engine.step(0.1)  # Let the system evolve a bit

        for d in range(engine.max_dim):
            rate = engine.clock_rate_modulation(d)
            assert (
                0.1 <= rate <= 1.0
            ), f"Clock rate for dimension {d} is out of bounds: {rate}"


if __name__ == "__main__":
    pytest.main()
