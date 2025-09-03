"""
Phase Dynamics Core Functions
==============================

Core phase evolution and energy transfer functions extracted from monolithic phase.py.
"""

import numpy as np

from ..mathematics import (
    NUMERICAL_EPSILON,
    PHI,
    phase_capacity,
)


def sap_rate(source, target, phase_density, phi=PHI, min_distance=1e-3):
    """
    Calculate energy-based sapping rate with proper equilibrium.

    Parameters
    ----------
    source : float
        Source dimension
    target : float
        Target dimension
    phase_density : array-like
        Current phase densities
    phi : float
        Golden ratio constant
    min_distance : float
        Minimum distance for regularization

    Returns
    -------
    float
        Sapping rate from source to target
    """
    source = float(source)
    target = float(target)

    if source >= target:
        return 0.0

    # Distance calculation
    distance = target - source
    if distance < min_distance:
        regularized_distance = (
            min_distance + phi * (distance / min_distance) ** 2
        )
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
    rate = (
        energy_deficit * distance_factor * frequency_ratio * equilibrium_factor
    )

    # Conservative rate limiting
    max_rate = 0.5  # Very conservative for stability
    rate = min(rate, max_rate)

    return float(rate)


def phase_evolution_step(phase_density, dt, max_dimension=None):
    """
    Energy-conserving phase evolution using direct energy transfers.
    This version ensures exact energy conservation by tracking all transfers.

    Parameters
    ----------
    phase_density : array-like
        Current phase densities
    dt : float
        Time step
    max_dimension : int, optional
        Maximum dimension to evolve

    Returns
    -------
    tuple
        (new_phase_density, flow_matrix)
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
                    max_energy_transfer = energies[source] * 0.1
                    energy_transfer = min(energy_transfer, max_energy_transfer)

                    if energy_transfer > NUMERICAL_EPSILON:
                        # Direct energy transfer (guaranteed conservation)
                        old_source_energy = energies[source]
                        old_target_energy = energies[target]

                        energies[source] -= energy_transfer
                        energies[target] += energy_transfer

                        # Ensure non-negative
                        energies[source] = max(0, energies[source])

                        # Exact conservation check
                        actual_transfer = old_source_energy - energies[source]
                        energies[target] = old_target_energy + actual_transfer

                        # Track flow
                        flow_matrix[source, target] = actual_transfer
                        total_energy_transferred += actual_transfer

    # Final energy conservation verification and correction
    final_total_energy = np.sum(energies)
    energy_error = final_total_energy - initial_total_energy

    # If there's any numerical error, distribute it proportionally
    if (
        abs(energy_error) > NUMERICAL_EPSILON
        and final_total_energy > NUMERICAL_EPSILON
    ):
        correction_factor = initial_total_energy / final_total_energy
        energies *= correction_factor

    # Reconstruct complex phase densities from energies and phases
    new_phase_density = np.sqrt(energies) * np.exp(1j * phases)

    return new_phase_density, flow_matrix


def total_phase_energy(phase_density):
    """
    Calculate total energy in the phase density system.

    Parameters
    ----------
    phase_density : array-like
        Current phase densities

    Returns
    -------
    float
        Total energy |ρ|²
    """
    return float(np.sum(np.abs(phase_density) ** 2))


def phase_coherence(phase_density):
    """
    Calculate phase coherence across dimensions.

    Parameters
    ----------
    phase_density : array-like
        Complex phase densities

    Returns
    -------
    float
        Coherence measure in [0, 1]
    """
    phase_density = np.asarray(phase_density, dtype=complex)

    # Remove zero entries
    non_zero = phase_density[np.abs(phase_density) > NUMERICAL_EPSILON]

    if len(non_zero) < 2:
        return 1.0  # Perfect coherence for single dimension

    # Calculate phase angles
    phases = np.angle(non_zero)

    # Circular variance as coherence measure
    mean_vector = np.mean(np.exp(1j * phases))
    coherence = np.abs(mean_vector)

    return float(coherence)


def dimensional_time(dimension_trajectory, phi=PHI):
    """
    Calculate dimensional time from emergence trajectory.

    Maps the sequence of dimensional emergence to a time coordinate
    using the golden ratio as the fundamental time constant.

    Parameters
    ----------
    dimension_trajectory : array-like
        Sequence of emerged dimensions
    phi : float
        Golden ratio constant

    Returns
    -------
    array
        Time values for each emergence event
    """
    trajectory = np.asarray(dimension_trajectory)
    n_events = len(trajectory)

    if n_events == 0:
        return np.array([])

    # Time accumulates geometrically with emergence events
    times = np.zeros(n_events)
    for i in range(1, n_events):
        # Time interval depends on dimensional jump
        delta_d = trajectory[i] - trajectory[i-1]
        times[i] = times[i-1] + phi ** delta_d

    return times
