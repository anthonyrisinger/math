#!/usr/bin/env python3
"""
Debug Phase Dynamics Energy Conservation
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from core import PhaseDynamicsEngine, total_phase_energy


def debug_phase_evolution():
    print("PHASE DYNAMICS ENERGY DEBUG")
    print("=" * 50)

    engine = PhaseDynamicsEngine(max_dimensions=6)

    print("Initial state:")
    state = engine.get_state()
    print(f"  Phase densities: {np.abs(state['phase_densities'])}")
    print(f"  Total energy: {state['total_energy']:.6f}")
    print(f"  Emerged dimensions: {state['emerged_dimensions']}")

    # Track energy over time
    energies = []
    times = []

    for step in range(20):
        engine.step(0.05)  # Smaller steps

        state = engine.get_state()
        energies.append(state["total_energy"])
        times.append(state["time"])

        if step % 5 == 0:
            print(f"\nStep {step:2d} (t={state['time']:.2f}):")
            print(f"  Phase densities: {np.abs(state['phase_densities'])}")
            print(f"  Total energy: {state['total_energy']:.6f}")
            print(f"  Emerged: {state['emerged_dimensions']}")

    # Analyze energy changes
    energy_changes = np.diff(energies)
    print("\nEnergy Analysis:")
    print(f"  Initial energy: {energies[0]:.6f}")
    print(f"  Final energy: {energies[-1]:.6f}")
    print(f"  Total change: {energies[-1] - energies[0]:.6f}")
    print(f"  Relative change: {(energies[-1] - energies[0]) / energies[0] * 100:.1f}%")
    print(f"  Max single step change: {np.max(np.abs(energy_changes)):.6f}")
    print(f"  Average step change: {np.mean(np.abs(energy_changes)):.6f}")

    # Check if energy changes correlate with emergence
    print(f"\nFinal emerged dimensions: {engine.emerged}")
    print(f"Number of new emergences: {len(engine.emerged) - 1}")  # -1 for initial void

    if len(engine.emerged) > 1:
        print("Energy change might be due to dimensional emergence")
        return True
    else:
        print("Energy change without emergence - potential bug")
        return False


def test_simple_transfer():
    """Test simple energy transfer between two dimensions."""
    print("\n" + "=" * 50)
    print("SIMPLE TRANSFER TEST")
    print("=" * 50)

    from core.phase import phase_evolution_step

    # Create simple two-dimension system
    phase_density = np.array([1.0, 0.1], dtype=complex)
    initial_energy = total_phase_energy(phase_density)

    print(f"Initial: ρ₀={abs(phase_density[0]):.3f}, ρ₁={abs(phase_density[1]):.3f}")
    print(f"Initial energy: {initial_energy:.6f}")

    # Single evolution step
    new_phase_density, flow_matrix = phase_evolution_step(phase_density, 0.01, 1)
    final_energy = total_phase_energy(new_phase_density)

    print(
        f"Final: ρ₀={abs(new_phase_density[0]):.3f}, ρ₁={abs(new_phase_density[1]):.3f}"
    )
    print(f"Final energy: {final_energy:.6f}")
    print(f"Energy change: {final_energy - initial_energy:.6f}")
    print(f"Flow matrix: {flow_matrix}")

    return abs(final_energy - initial_energy) < 0.01


if __name__ == "__main__":
    # Run debug
    emergence_explanation = debug_phase_evolution()
    conservation_in_simple = test_simple_transfer()

    print("\n" + "=" * 50)
    print("CONCLUSIONS")
    print("=" * 50)

    if emergence_explanation:
        print("✅ Energy change likely due to dimensional emergence - this is expected")
    else:
        print("❌ Energy change without clear cause")

    if conservation_in_simple:
        print("✅ Simple transfers conserve energy")
    else:
        print("❌ Energy not conserved even in simple transfers")
