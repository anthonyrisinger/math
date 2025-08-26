#!/usr/bin/env python3
"""
Dimensional Phase Dynamics
==========================

Unified phase dynamics module consolidating all phase sapping functionality.
Implements the fundamental mechanism by which higher dimensions "feed" on
lower ones, driving dimensional emergence and creating the arrow of time.

This module consolidates:
- core/phase.py (robust mathematical implementations)
- phase_dynamics.py (visualization and interaction classes)

Core Theory:
∂ρ_d/∂t = Σ_s R(s→d)ρ_s - Σ_t R(d→t)ρ_t

Where R(s→t) is the sapping rate from dimension s to dimension t.

Features:
- Energy-conserving phase sapping dynamics
- Phase capacity thresholds and emergence detection
- Interactive visualization and simulation
- Dimensional time evolution
- Comprehensive emergence engine
"""

import numpy as np

from .gamma import PHI
from .measures import phase_capacity

# ============================================================================
# CORE PHASE SAPPING FUNCTIONS
# ============================================================================


def sap_rate(source, target, phase_density=None, phi=PHI, min_distance=1e-3):
    """
    Calculate energy-based sapping rate with proper equilibrium.

    Parameters
    ----------
    source : float
        Source dimension
    target : float
        Target dimension
    phase_density : array-like, optional
        Current phase density state
    phi : float
        Golden ratio parameter for distance regularization
    min_distance : float
        Minimum distance for numerical stability

    Returns
    -------
    float
        Sapping rate from source to target
    """
    source = float(source)
    target = float(target)

    if source >= target:
        return 0.0

    # Distance calculation with golden ratio regularization
    distance = target - source
    if distance < min_distance:
        regularized_distance = min_distance + phi * (distance / min_distance) ** 2
    else:
        regularized_distance = distance + phi

    # Target energy (if phase density provided)
    target_energy = 0.0
    if phase_density is not None:
        target_idx = int(target) if target == int(target) else None
        if target_idx is not None and 0 <= target_idx < len(phase_density):
            target_energy = abs(phase_density[target_idx]) ** 2

    # Capacity calculations with error handling
    try:
        capacity_magnitude = phase_capacity(target)
        capacity_energy = capacity_magnitude**2
    except (ValueError, OverflowError):
        return 0.0

    # Energy deficit calculation
    energy_deficit = max(0.0, capacity_energy - target_energy)

    # Frequency ratio
    freq_ratio = np.sqrt((target + 1) / (source + 1))

    # Equilibrium factor with phase dynamics
    equilibrium_factor = np.exp(1j * np.pi * target / 6)

    # Combined sapping rate
    base_rate = energy_deficit / regularized_distance
    rate = base_rate * freq_ratio * abs(equilibrium_factor)

    return max(0.0, rate)


def phase_evolution_step(phase_density, dt=0.01, max_dimensions=None):
    """
    Evolve phase density by one time step using sapping dynamics.

    Parameters
    ----------
    phase_density : array-like
        Current phase density state
    dt : float
        Time step size
    max_dimensions : int, optional
        Maximum dimensions to consider

    Returns
    -------
    array
        Updated phase density
    """
    phase_density = np.array(phase_density, dtype=complex)
    if max_dimensions is None:
        max_dimensions = len(phase_density)

    # Calculate sapping rates between all dimension pairs
    new_phase = phase_density.copy()

    for d in range(max_dimensions):
        # Inflow from lower dimensions
        inflow = 0.0
        for s in range(d):
            rate = sap_rate(s, d, phase_density)
            inflow += rate * abs(phase_density[s]) ** 2

        # Outflow to higher dimensions
        outflow = 0.0
        for t in range(d + 1, max_dimensions):
            rate = sap_rate(d, t, phase_density)
            outflow += rate * abs(phase_density[d]) ** 2

        # Update phase density
        new_phase[d] += dt * (inflow - outflow)

    return new_phase


def emergence_threshold(d, threshold_factor=0.9):
    """
    Calculate emergence threshold for dimension d.

    Parameters
    ----------
    d : float
        Dimension
    threshold_factor : float
        Fraction of capacity required for emergence

    Returns
    -------
    float
        Emergence threshold
    """
    return threshold_factor * phase_capacity(d)


def total_phase_energy(phase_density):
    """
    Calculate total phase energy in the system.

    Parameters
    ----------
    phase_density : array-like
        Phase density state

    Returns
    -------
    float
        Total energy
    """
    return np.sum(np.abs(phase_density) ** 2)


def phase_coherence(phase_density):
    """
    Calculate phase coherence across dimensions.

    Parameters
    ----------
    phase_density : array-like
        Phase density state

    Returns
    -------
    float
        Coherence measure (0 = incoherent, 1 = fully coherent)
    """
    phases = np.angle(phase_density)
    # Calculate phase spread
    phase_spread = np.std(phases)
    return np.exp(-phase_spread)


def dimensional_time(dimension, phi=PHI):
    """
    Convert dimension to time using golden ratio coupling.

    t = φ(d + 1)

    Parameters
    ----------
    dimension : float
        Current dimension
    phi : float
        Golden ratio coupling constant

    Returns
    -------
    float
        Corresponding time
    """
    return phi * (dimension + 1)


# ============================================================================
# PHASE DYNAMICS ENGINE
# ============================================================================


class PhaseDynamicsEngine:
    """
    Complete phase dynamics simulation engine.

    Manages the evolution of phase densities across dimensions,
    tracks emergence events, and provides analysis tools.
    """

    def __init__(self, max_dimensions=12, initial_state=None):
        """
        Initialize the phase dynamics engine.

        Parameters
        ----------
        max_dimensions : int
            Maximum number of dimensions to simulate
        initial_state : array-like, optional
            Initial phase density state
        """
        self.max_dim = max_dimensions

        # Initialize phase state
        if initial_state is not None:
            self.phase_density = np.array(initial_state, dtype=complex)
        else:
            self.phase_density = np.zeros(max_dimensions, dtype=complex)
            self.phase_density[0] = 1.0  # Start with unity at the void

        # Evolution tracking
        self.time = 0.0
        self.dt = 0.01
        self.history = []

        # Emergence tracking
        self.emerged_dimensions = {0}  # Dimension 0 (void) always exists
        self.emergence_times = {0: 0.0}
        self.emergence_thresholds = {}

        # Calculate emergence thresholds
        for d in range(max_dimensions):
            self.emergence_thresholds[d] = emergence_threshold(d)

    def evolve(self, time_steps=1):
        """
        Evolve the system for given number of time steps.

        Parameters
        ----------
        time_steps : int
            Number of time steps to evolve

        Returns
        -------
        dict
            Evolution summary
        """
        initial_time = self.time
        new_emergences = []

        for _ in range(time_steps):
            # Store current state
            self.history.append(
                {
                    "time": self.time,
                    "phase_density": self.phase_density.copy(),
                    "total_energy": total_phase_energy(self.phase_density),
                    "coherence": phase_coherence(self.phase_density),
                }
            )

            # Check for new emergences before evolution
            for d in range(self.max_dim):
                if d not in self.emerged_dimensions:
                    phase_magnitude = abs(self.phase_density[d])
                    if phase_magnitude >= self.emergence_thresholds[d]:
                        self.emerged_dimensions.add(d)
                        self.emergence_times[d] = self.time
                        new_emergences.append((d, self.time))

            # Evolve phase state
            self.phase_density = phase_evolution_step(
                self.phase_density, self.dt, self.max_dim
            )
            self.time += self.dt

        return {
            "time_range": (initial_time, self.time),
            "new_emergences": new_emergences,
            "current_emerged": list(self.emerged_dimensions),
            "total_energy": total_phase_energy(self.phase_density),
            "coherence": phase_coherence(self.phase_density),
        }

    def reset(self, initial_state=None):
        """Reset the simulation to initial conditions."""
        if initial_state is not None:
            self.phase_density = np.array(initial_state, dtype=complex)
        else:
            self.phase_density = np.zeros(self.max_dim, dtype=complex)
            self.phase_density[0] = 1.0

        self.time = 0.0
        self.history = []
        self.emerged_dimensions = {0}
        self.emergence_times = {0: 0.0}

    def get_current_state(self):
        """Get current system state."""
        return {
            "time": self.time,
            "phase_density": self.phase_density.copy(),
            "emerged_dimensions": list(self.emerged_dimensions),
            "emergence_times": self.emergence_times.copy(),
            "total_energy": total_phase_energy(self.phase_density),
            "coherence": phase_coherence(self.phase_density),
            "effective_dimension": self.calculate_effective_dimension(),
        }

    def calculate_effective_dimension(self):
        """Calculate the effective dimension of the system."""
        # Weighted average dimension based on phase densities
        weights = np.abs(self.phase_density) ** 2
        total_weight = np.sum(weights)

        if total_weight == 0:
            return 0.0

        dimensions = np.arange(self.max_dim)
        return np.sum(dimensions * weights) / total_weight

    def analyze_convergence(self, window_size=100):
        """
        Analyze convergence properties of the evolution.

        Parameters
        ----------
        window_size : int
            Number of recent steps to analyze

        Returns
        -------
        dict
            Convergence analysis
        """
        if len(self.history) < window_size:
            return {"error": "Insufficient history for analysis"}

        recent_history = self.history[-window_size:]

        # Energy stability
        energies = [state["total_energy"] for state in recent_history]
        energy_variance = np.var(energies)

        # Coherence stability
        coherences = [state["coherence"] for state in recent_history]
        coherence_variance = np.var(coherences)

        # Dimensional drift
        effective_dims = []
        for state in recent_history:
            weights = np.abs(state["phase_density"]) ** 2
            total_weight = np.sum(weights)
            if total_weight > 0:
                dims = np.arange(len(state["phase_density"]))
                eff_dim = np.sum(dims * weights) / total_weight
                effective_dims.append(eff_dim)

        dimensional_variance = np.var(effective_dims) if effective_dims else 0.0

        return {
            "energy_variance": energy_variance,
            "coherence_variance": coherence_variance,
            "dimensional_variance": dimensional_variance,
            "is_converged": (
                energy_variance < 1e-6
                and coherence_variance < 1e-6
                and dimensional_variance < 1e-6
            ),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def run_emergence_simulation(max_time=10.0, max_dimensions=8, dt=0.01):
    """
    Run a complete emergence simulation.

    Parameters
    ----------
    max_time : float
        Maximum simulation time
    max_dimensions : int
        Number of dimensions to simulate
    dt : float
        Time step size

    Returns
    -------
    dict
        Complete simulation results
    """
    engine = PhaseDynamicsEngine(max_dimensions)
    engine.dt = dt

    time_steps = int(max_time / dt)
    results = engine.evolve(time_steps)

    return {
        "engine": engine,
        "final_state": engine.get_current_state(),
        "evolution_summary": results,
        "convergence": engine.analyze_convergence(),
    }


def quick_phase_analysis(dimension=4.0, time_steps=1000):
    """
    Quick analysis of phase dynamics around a specific dimension.

    Parameters
    ----------
    dimension : float
        Dimension to analyze
    time_steps : int
        Evolution time steps

    Returns
    -------
    dict
        Analysis results
    """
    max_dim = int(dimension) + 3
    engine = PhaseDynamicsEngine(max_dim)

    # Inject phase at target dimension
    target_idx = int(dimension)
    if target_idx < max_dim:
        engine.phase_density[target_idx] = 0.5

    # Evolve
    results = engine.evolve(time_steps)

    return {
        "target_dimension": dimension,
        "final_state": engine.get_current_state(),
        "emergence_pattern": results["new_emergences"],
        "energy_conservation": total_phase_energy(engine.phase_density),
    }


# ============================================================================
# VERIFICATION AND TESTING
# ============================================================================


def verify_phase_dynamics():
    """Verify phase dynamics properties."""
    results = {}

    # Energy conservation test
    engine = PhaseDynamicsEngine(max_dimensions=5)
    initial_energy = total_phase_energy(engine.phase_density)
    engine.evolve(100)
    final_energy = total_phase_energy(engine.phase_density)

    energy_conservation_error = abs(final_energy - initial_energy) / initial_energy
    results["energy_conserved"] = energy_conservation_error < 0.1  # 10% tolerance

    # Sapping rate properties
    results["no_reverse_sapping"] = sap_rate(5, 3) == 0.0  # Higher can't sap lower
    results["self_sapping_zero"] = sap_rate(3, 3) == 0.0  # No self-sapping

    # Emergence threshold consistency
    threshold_4 = emergence_threshold(4)
    capacity_4 = phase_capacity(4)
    results["threshold_reasonable"] = 0.5 < threshold_4 < capacity_4

    return results


if __name__ == "__main__":
    # Quick verification
    print("PHASE DYNAMICS VERIFICATION")
    print("=" * 50)

    verification = verify_phase_dynamics()
    for test, passed in verification.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:25}: {status}")

    print(
        f"\nOverall: {'✅ ALL TESTS PASSED' if all(verification.values()) else '❌ SOME TESTS FAILED'}"
    )

    # Quick simulation demo
    print("\nQUICK SIMULATION DEMO:")
    print("-" * 30)
    results = run_emergence_simulation(max_time=5.0, max_dimensions=6)
    state = results["final_state"]

    print(f"Final time: {state['time']:.2f}")
    print(f"Emerged dimensions: {state['emerged_dimensions']}")
    print(f"Effective dimension: {state['effective_dimension']:.3f}")
    print(f"Total energy: {state['total_energy']:.6f}")
    print(f"Coherence: {state['coherence']:.6f}")
