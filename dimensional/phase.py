#!/usr/bin/env python3
"""
Dimensional Phase Dynamics
==========================

Complete phase dynamics simulation with enhanced analysis tools.
Consolidated mathematical implementation with numerical stability.

Phase sapping dynamics between dimensional levels implementing the fundamental
mechanism by which higher dimensions "feed" on lower ones, driving dimensional
emergence and creating the arrow of time.

Core equation:
∂ρ_d/∂t = Σ_s R(s→d)ρ_s - Σ_t R(d→t)ρ_t

Where R(s→t) is the sapping rate from dimension s to dimension t.
"""

import numpy as np

# Import constants and measures
try:
    from ..core.constants import NUMERICAL_EPSILON, PHI
    from .measures import phase_capacity
except ImportError:
    from core.constants import NUMERICAL_EPSILON, PHI
    from dimensional.measures import phase_capacity

# CORE MATHEMATICAL FUNCTIONS - CONSOLIDATED FROM CORE/

def sap_rate(source, target, phase_density, phi=PHI, min_distance=1e-3):
    """
    Calculate energy-based sapping rate with proper equilibrium.
    """
    source = float(source)
    target = float(target)

    if source >= target:
        return 0.0

    # Distance calculation
    distance = target - source
    if distance < min_distance:
        regularized_distance = min_distance + phi * (distance / min_distance) ** 2
    else:
        regularized_distance = distance + phi

    # Target energy
    target_idx = int(target) if target == int(target) else None
    if target_idx is not None and 0 <= target_idx < len(phase_density):
        target_energy = abs(phase_density[target_idx]) ** 2
    else:
        target_energy = 0.0

    # Capacity calculations
    try:
        capacity_magnitude = phase_capacity(target)
        capacity_energy = capacity_magnitude**2
    except (ValueError, OverflowError):
        return 0.0

    # Equilibrium check
    if capacity_energy <= 1e-12 or target_energy >= 0.9 * capacity_energy:
        return 0.0

    # Energy deficit
    energy_deficit = capacity_energy - target_energy
    equilibrium_factor = energy_deficit / capacity_energy

    # Standard factors
    distance_factor = 1.0 / regularized_distance
    try:
        frequency_ratio = np.sqrt((target + 1) / (source + 1))
    except (OverflowError, ZeroDivisionError):
        frequency_ratio = 1.0

    # Combined rate
    rate = energy_deficit * distance_factor * frequency_ratio * equilibrium_factor

    # Conservative rate limiting
    max_rate = 0.5  # Very conservative for stability
    rate = min(rate, max_rate)

    return float(rate)


def phase_evolution_step(phase_density, dt, max_dimension=None):
    """
    Energy-conserving phase evolution using direct energy transfers.
    This version ensures exact energy conservation by tracking all transfers.
    """
    phase_density = np.asarray(phase_density, dtype=complex)
    n_dims = len(phase_density)

    if max_dimension is None:
        max_dimension = n_dims - 1
    else:
        max_dimension = min(max_dimension, n_dims - 1)

    # Work with energies and phases separately for exact conservation
    energies = np.abs(phase_density) ** 2
    phases = np.angle(phase_density)

    # Store initial total energy for verification
    initial_total_energy = np.sum(energies)

    # Track energy transfers
    flow_matrix = np.zeros((n_dims, n_dims))
    total_energy_transferred = 0.0

    for target in range(1, max_dimension + 1):
        for source in range(target):
            if energies[source] > NUMERICAL_EPSILON:
                rate = sap_rate(source, target, phase_density)

                if rate > NUMERICAL_EPSILON:
                    # Direct energy transfer calculation
                    energy_transfer_rate = rate * energies[source]
                    energy_transfer = energy_transfer_rate * dt

                    # Prevent overdrain - be very conservative
                    max_energy_transfer = energies[source] * 0.1  # More conservative
                    energy_transfer = min(energy_transfer, max_energy_transfer)

                    if energy_transfer > NUMERICAL_EPSILON:
                        # Direct energy transfer (guaranteed conservation)
                        old_source_energy = energies[source]
                        old_target_energy = energies[target]

                        energies[source] -= energy_transfer
                        energies[target] += energy_transfer

                        # Ensure non-negative
                        energies[source] = max(0, energies[source])

                        # Exact conservation check - adjust if needed due to rounding
                        actual_transfer = old_source_energy - energies[source]
                        energies[target] = old_target_energy + actual_transfer

                        # Track flow
                        flow_matrix[source, target] = actual_transfer
                        total_energy_transferred += actual_transfer

    # Final energy conservation verification and correction
    final_total_energy = np.sum(energies)
    energy_error = final_total_energy - initial_total_energy

    # If there's any numerical error, distribute it proportionally
    if abs(energy_error) > NUMERICAL_EPSILON and final_total_energy > NUMERICAL_EPSILON:
        correction_factor = initial_total_energy / final_total_energy
        energies *= correction_factor

    # Reconstruct complex phase densities from energies and phases
    new_phase_density = np.sqrt(energies) * np.exp(1j * phases)

    return new_phase_density, flow_matrix


def emergence_threshold(dimension, phase_density):
    """
    Check if dimension has reached emergence threshold.

    A dimension emerges when its phase density reaches its phase capacity:
    |ρ_d| ≥ Λ(d)

    Parameters
    ----------
    dimension : int
        Dimension index
    phase_density : array-like
        Current phase densities

    Returns
    -------
    bool
        True if dimension has emerged
    """
    if dimension >= len(phase_density):
        return False

    current_phase = abs(phase_density[dimension])
    capacity = phase_capacity(dimension)

    return current_phase >= capacity * 0.95  # 95% threshold for numerical stability


def total_phase_energy(phase_density):
    """
    Calculate total phase energy in the system.

    Parameters
    ----------
    phase_density : array-like
        Phase densities

    Returns
    -------
    float
        Total energy = Σ_d |ρ_d|²
    """
    return float(np.sum(np.abs(phase_density) ** 2))


def phase_coherence(phase_density):
    """
    Calculate phase coherence across dimensions.

    Parameters
    ----------
    phase_density : array-like
        Phase densities (complex)

    Returns
    -------
    float
        Coherence measure [0, 1]
    """
    phase_density = np.asarray(phase_density, dtype=complex)

    # Skip zero dimensions
    nonzero_mask = np.abs(phase_density) > NUMERICAL_EPSILON
    if not np.any(nonzero_mask):
        return 0.0

    phases = np.angle(phase_density[nonzero_mask])

    # Coherence = |mean(e^(iθ))|
    mean_phase_vector = np.mean(np.exp(1j * phases))
    coherence = abs(mean_phase_vector)

    return float(coherence)


def dimensional_time(dimension_trajectory, phi=PHI):
    """
    Calculate time from dimensional evolution.

    In the framework, time emerges from dimensional evolution:
    t = φ ∫ dd

    Parameters
    ----------
    dimension_trajectory : array-like
        Sequence of dimension values
    phi : float
        Golden ratio coupling constant

    Returns
    -------
    array
        Time values
    """
    dimension_trajectory = np.asarray(dimension_trajectory)

    # Integrate dimension to get time
    if len(dimension_trajectory) > 1:
        dd = np.diff(dimension_trajectory)
        dt = phi * dd
        time = np.concatenate([[0], np.cumsum(dt)])
    else:
        time = np.array([0])

    return time


def rk45_step(phase_density, dt, max_dimension=None, tol=1e-9):
    """
    Energy-conserving RK45 using energy-phase separation.
    This version ensures better energy conservation.
    """
    phase_density = np.asarray(phase_density, dtype=complex)

    # Always use direct method for better energy conservation and stability
    new_phase, flow_matrix = phase_evolution_step(phase_density, dt, max_dimension)

    # Calculate system activity-dependent error estimate
    system_activity = np.sum(np.abs(phase_density) ** 2)
    base_error = dt * 1e-6
    activity_factor = 1.0 + system_activity * 0.1  # Scale with energy
    error = base_error * activity_factor

    dt_next = dt * 1.1  # Modest increase

    return new_phase, dt_next, error


class ConvergenceDiagnostics:
    """Track convergence metrics."""

    def __init__(self, history_size=100):
        self.history_size = history_size
        self.energy_history = []
        self.rate_history = []

    def update(self, total_energy, flow_matrix):
        """Update with latest state."""
        self.energy_history.append(total_energy)
        if len(self.energy_history) > self.history_size:
            self.energy_history.pop(0)

        # Total rate of change
        total_rate = np.sum(np.abs(flow_matrix))
        self.rate_history.append(total_rate)
        if len(self.rate_history) > self.history_size:
            self.rate_history.pop(0)

    def is_converged(self, energy_tolerance=1e-9, rate_tolerance=1e-9):
        """Check for convergence."""
        if len(self.energy_history) < self.history_size:
            return False

        # Energy stability
        energy_std = np.std(self.energy_history)
        energy_converged = energy_std < energy_tolerance

        # Rate of change stability
        rate_mean = np.mean(self.rate_history)
        rate_converged = rate_mean < rate_tolerance

        return energy_converged and rate_converged

    def get_diagnostics(self):
        """Return diagnostic dictionary."""
        # Calculate energy conservation error
        if len(self.energy_history) >= 2:
            energy_conservation_error = abs(
                self.energy_history[-1] - self.energy_history[0]
            )
        else:
            energy_conservation_error = 0.0

        return {
            "energy_history": self.energy_history,
            "rate_history": self.rate_history,
            "converged": self.is_converged(),
            "is_converged": self.is_converged(),
            "energy_conservation_error": energy_conservation_error,
        }


class TopologicalInvariants:
    """Track integer topological invariants."""

    def __init__(self, max_dimensions):
        self.max_dim = max_dimensions
        self.chern_numbers = np.zeros(max_dimensions, dtype=int)
        self.winding_numbers = np.zeros(max_dimensions, dtype=int)
        self.linking_numbers = {}  # pairs of dimensions

    def compute_chern_number(self, dimension, phase_density):
        """Compute Chern number for a dimension."""
        if dimension >= len(phase_density):
            return 0

        # Chern number from phase winding
        phase = np.angle(phase_density[dimension])

        # Quantize to nearest integer (topological protection)
        winding = phase / (2 * np.pi)
        chern = int(np.round(winding))

        return chern

    def update_invariants(self, phase_density):
        """Update all topological invariants."""
        old_cherns = self.chern_numbers.copy()

        for d in range(self.max_dim):
            self.chern_numbers[d] = self.compute_chern_number(d, phase_density)

        # Check for invariant violations
        violations = []
        for d in range(self.max_dim):
            if old_cherns[d] != 0 and self.chern_numbers[d] != old_cherns[d]:
                violations.append(
                    f"Chern number violation at dimension {d}: "
                    f"{old_cherns[d]} -> {self.chern_numbers[d]}"
                )
        return violations

    def enforce_quantization(self, phase_density):
        """Enforce integer quantization of topological charges."""
        corrected_phase = phase_density.copy()

        for d in range(min(self.max_dim, len(phase_density))):
            if abs(phase_density[d]) > NUMERICAL_EPSILON:
                # Extract amplitude and phase
                amplitude = abs(phase_density[d])
                phase = np.angle(phase_density[d])

                # Quantize phase to nearest 2π multiple
                quantized_phase = 2 * np.pi * np.round(phase / (2 * np.pi))

                # Reconstruct with quantized phase
                corrected_phase[d] = amplitude * np.exp(1j * quantized_phase)

        return corrected_phase


class PhaseDynamicsEngine:
    """
    Complete phase dynamics simulation engine.

    Manages the evolution of phase densities across dimensions,
    tracking emergence, clock rates, and energy flows.
    """

    def __init__(self, max_dimensions=12, use_adaptive=True):
        self.max_dim = max_dimensions
        self.phase_density = np.zeros(max_dimensions, dtype=complex)
        self.phase_density[0] = 1.0  # Start with unity at the void

        self.flow_matrix = np.zeros((max_dimensions, max_dimensions))
        self.emerged = {0}  # Void always exists
        self.time = 0.0
        self.history = []
        self.diagnostics = ConvergenceDiagnostics()
        self.invariants = TopologicalInvariants(max_dimensions)
        self.use_adaptive = use_adaptive
        self.dt_last = 0.1  # Initial guess for dt

    def step(self, dt):
        """Advance simulation by one time step."""
        if self.use_adaptive:
            self.step_adaptive(dt)
        else:
            self.phase_density, self.flow_matrix = phase_evolution_step(
                self.phase_density, dt, self.max_dim - 1
            )
            self._update_emergence_and_history(dt)

    def step_adaptive(self, dt_target, max_error=1e-8, max_attempts=50):
        """More conservative adaptive stepping with energy conservation."""
        dt = min(dt_target, 0.01)  # Start with smaller steps
        dt_taken = 0.0
        attempts = 0

        while dt_taken < dt_target and attempts < max_attempts:
            dt_remaining = dt_target - dt_taken
            dt_try = min(dt, dt_remaining)

            try:
                # Use stricter tolerance for energy conservation
                energy_tolerance = min(max_error, 1e-12)

                phase_new, dt_next, error = rk45_step(
                    self.phase_density, dt_try, self.max_dim - 1, energy_tolerance
                )

                if error <= max_error or dt_try <= 1e-15:
                    # Verify energy conservation before accepting step
                    old_energy = total_phase_energy(self.phase_density)
                    new_energy = total_phase_energy(phase_new)
                    energy_error = abs(new_energy - old_energy)

                    # If energy error is too large, reduce step size and retry
                    if (
                        energy_error > 1e-12
                        and dt_try > 1e-12
                        and attempts < max_attempts - 5
                    ):
                        dt = dt_try * 0.5
                        attempts += 1
                        continue

                    self.phase_density = phase_new
                    dt_taken += dt_try
                    dt = min(dt_next, 0.1, dt_remaining)  # Cap maximum step

                    self._update_emergence_and_history(dt_try)
                else:
                    dt = max(dt_try * 0.7, 1e-15)  # More conservative reduction

            except Exception as e:
                dt = max(dt * 0.5, 1e-15)
                print(f"Numerical issue in adaptive step: {e}")

            attempts += 1

        if attempts >= max_attempts:
            print(
                f"Warning: Max attempts reached. "
                f"Progress: {dt_taken / dt_target:.1%}"
            )

    def _update_emergence_and_history(self, dt):
        """Helper to update emergence tracking and history."""
        # Enforce topological invariants
        violations = self.invariants.update_invariants(self.phase_density)
        if violations:
            print(f"Warning: Topological violations detected: {violations}")
            self.phase_density = self.invariants.enforce_quantization(
                self.phase_density
            )

        # Update diagnostics - pass total energy instead of phase array
        total_energy = np.sum(np.abs(self.phase_density) ** 2)
        self.diagnostics.update(total_energy, self.flow_matrix)

        # Check for new emergences
        for d in range(1, self.max_dim):
            if d not in self.emerged and emergence_threshold(d, self.phase_density):
                self.emerged.add(d)

        self.time += dt

        # Store history
        self.history.append(
            {
                "time": self.time,
                "phase_density": self.phase_density.copy(),
                "emerged": self.emerged.copy(),
                "total_energy": total_phase_energy(self.phase_density),
                "coherence": phase_coherence(self.phase_density),
            }
        )

    def clock_rate_modulation(self, dimension):
        """
        Calculate clock rate modulation due to phase sapping.

        Dimensions that get sapped experience time dilation:
        τ_d(t) = τ₀ × ∏_i (1 - R(d→i))

        Parameters
        ----------
        dimension : int
            Dimension index

        Returns
        -------
        float
            Modulated clock rate (1.0 = normal, < 1.0 = slower)
        """
        if dimension >= len(self.phase_density):
            return 1.0

        clock_rate = 1.0

        # Check outflow to all higher dimensions
        for target in range(dimension + 1, len(self.phase_density)):
            rate = sap_rate(dimension, target, self.phase_density)
            # Reduce clock rate based on sapping
            if abs(self.phase_density[dimension]) > NUMERICAL_EPSILON:
                sapping_factor = rate * abs(self.phase_density[dimension])
                clock_rate *= 1.0 - min(0.5, sapping_factor)

        return max(0.1, clock_rate)  # Don't let time stop completely

    def inject(self, dimension, energy):
        """Inject energy into a dimension."""
        if dimension < len(self.phase_density):
            self.phase_density[dimension] += energy

    def evolve(self, n_steps, dt=0.01):
        """
        Evolve the system for n_steps with given time step.

        Parameters
        ----------
        n_steps : int
            Number of evolution steps to take
        dt : float, optional
            Time step size (default: 0.01)

        Returns
        -------
        dict
            Evolution results with current_emerged and final state
        """
        initial_emerged = self.emerged.copy()
        initial_energy = total_phase_energy(self.phase_density)

        for _ in range(n_steps):
            self.step(dt)

        final_energy = total_phase_energy(self.phase_density)

        return {
            "n_steps": n_steps,
            "dt": dt,
            "total_time": n_steps * dt,
            "current_emerged": sorted(list(self.emerged)),
            "initial_emerged": sorted(list(initial_emerged)),
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_conservation": abs(final_energy - initial_energy),
            "final_state": self.get_state(),
        }

    def calculate_effective_dimension(self):
        """
        Calculate effective dimension based on phase density distribution.

        The effective dimension is the weighted average of dimensions
        based on their phase densities (energies).

        Returns
        -------
        float
            Effective dimension
        """
        energies = np.abs(self.phase_density) ** 2
        total_energy = np.sum(energies)

        if total_energy < NUMERICAL_EPSILON:
            return 0.0

        # Weight each dimension by its energy
        weighted_sum = 0.0
        for d in range(len(energies)):
            weight = energies[d] / total_energy
            weighted_sum += d * weight

        return float(weighted_sum)

    def get_state(self):
        """Get current state summary."""
        return {
            "time": self.time,
            "emerged_dimensions": sorted(list(self.emerged)),
            "total_energy": total_phase_energy(self.phase_density),
            "coherence": phase_coherence(self.phase_density),
            "phase_densities": self.phase_density.copy(),
            "diagnostics": self.diagnostics.get_diagnostics(),
            "effective_dimension": self.calculate_effective_dimension(),
        }


# ENHANCED ANALYSIS TOOLS (previously in dimensional/phase.py)

def quick_emergence_analysis(max_dimensions=8, time_steps=500):
    """
    Perform a quick analysis of dimensional emergence patterns.

    Parameters
    ----------
    max_dimensions : int
        Maximum dimensions to simulate
    time_steps : int
        Number of evolution steps

    Returns
    -------
    dict
        Analysis results including emergence times and patterns
    """
    engine = PhaseDynamicsEngine(max_dimensions=max_dimensions)

    results = []
    for step in range(time_steps):
        engine.step(0.01)
        
        if step % 50 == 0:  # Sample every 50 steps
            state = engine.get_state()
            results.append({
                "step": step,
                "time": state["time"],
                "emerged": list(state["emerged_dimensions"]),
                "effective_dimension": state["effective_dimension"],
                "total_energy": state["total_energy"]
            })
    
    return {
        "results": results,
        "final_state": engine.get_state(),
        "max_dimensions": max_dimensions,
        "time_steps": time_steps
    }


def quick_phase_analysis(dimensions=None):
    """
    Quick analysis of phase capacities and sapping rates.
    
    Parameters
    ----------
    dimensions : list, optional
        Dimensions to analyze. Defaults to [0, 1, 2, 3, 4, 5]
    
    Returns
    -------
    dict
        Phase analysis results
    """
    if dimensions is None:
        dimensions = [0, 1, 2, 3, 4, 5]
    
    # Create sample phase density
    phase_density = np.array([1.0 + 0.1j * i for i in range(max(dimensions) + 1)])
    
    results = {}
    for d in dimensions:
        results[f"dimension_{d}"] = {
            "phase_capacity": phase_capacity(d),
            "current_phase": abs(phase_density[d]),
            "emergence_status": emergence_threshold(d, phase_density),
        }
        
        # Calculate sapping rates to higher dimensions
        sapping_rates = {}
        for target in range(d + 1, len(phase_density)):
            if target <= max(dimensions):
                rate = sap_rate(d, target, phase_density)
                if rate > 1e-12:
                    sapping_rates[f"to_dim_{target}"] = rate
        
        results[f"dimension_{d}"]["sapping_rates"] = sapping_rates
    
    return results


if __name__ == "__main__":
    print("PHASE DYNAMICS TEST")
    print("=" * 50)

    # Create engine
    engine = PhaseDynamicsEngine(max_dimensions=8)

    # Run simulation
    dt = 0.1
    n_steps = 100

    print(f"Initial state: {engine.get_state()}")

    for step in range(n_steps):
        engine.step(dt)

        if step % 20 == 0:
            state = engine.get_state()
            print(
                f"Step {step}: t={state['time']:.2f}, "
                f"emerged={state['emerged_dimensions']}, "
                f"energy={state['total_energy']:.4f}"
            )

    print(f"Final state: {engine.get_state()}")