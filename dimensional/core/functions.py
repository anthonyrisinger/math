"""
Core functions module for phase evolution and other operations.
"""

import numpy as np


def phase_evolution_step(phase_state, dt=0.01, coupling=0.1):
    """Evolve phase state by one time step.

    Args:
        phase_state: Current phase state (array or scalar)
        dt: Time step
        coupling: Coupling strength

    Returns:
        Updated phase state
    """
    phase_state = np.asarray(phase_state, dtype=np.float64)

    # Simple evolution: diffusion with coupling
    if phase_state.ndim > 0:
        # Array case: coupled oscillators
        laplacian = np.zeros_like(phase_state)
        laplacian[1:-1] = phase_state[:-2] - 2*phase_state[1:-1] + phase_state[2:]
        laplacian[0] = phase_state[1] - phase_state[0]
        laplacian[-1] = phase_state[-2] - phase_state[-1]

        # Evolution equation
        d_phase = coupling * laplacian
        new_state = phase_state + dt * d_phase
    else:
        # Scalar case: simple decay
        new_state = phase_state * (1 - dt * coupling)

    return new_state


def phase_coupling_matrix(n_dims, coupling_type='nearest'):
    """Generate phase coupling matrix.

    Args:
        n_dims: Number of dimensions
        coupling_type: Type of coupling ('nearest', 'all-to-all', 'power-law')

    Returns:
        Coupling matrix
    """
    if coupling_type == 'nearest':
        # Nearest-neighbor coupling
        matrix = np.zeros((n_dims, n_dims))
        for i in range(n_dims - 1):
            matrix[i, i+1] = 1
            matrix[i+1, i] = 1
    elif coupling_type == 'all-to-all':
        # All-to-all coupling
        matrix = np.ones((n_dims, n_dims)) - np.eye(n_dims)
    elif coupling_type == 'power-law':
        # Power-law coupling
        matrix = np.zeros((n_dims, n_dims))
        for i in range(n_dims):
            for j in range(n_dims):
                if i != j:
                    matrix[i, j] = 1.0 / (1 + abs(i - j))
    else:
        raise ValueError(f"Unknown coupling type: {coupling_type}")

    # Normalize
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix = matrix / row_sums

    return matrix


def phase_synchronization_order(phases):
    """Calculate phase synchronization order parameter.

    Args:
        phases: Array of phase values

    Returns:
        Order parameter (0 = incoherent, 1 = fully synchronized)
    """
    phases = np.asarray(phases, dtype=np.float64)

    # Kuramoto order parameter
    complex_phases = np.exp(1j * phases)
    order = np.abs(np.mean(complex_phases))

    return float(order)


def phase_entropy(phase_density):
    """Calculate entropy of phase density distribution.

    Args:
        phase_density: Phase density array

    Returns:
        Shannon entropy
    """
    phase_density = np.asarray(phase_density, dtype=np.float64)

    # Normalize
    phase_density = phase_density / np.sum(phase_density)

    # Shannon entropy
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    entropy = -np.sum(phase_density * np.log(phase_density + epsilon))

    return float(entropy)


def phase_coherence_length(phase_field):
    """Calculate coherence length of phase field.

    Args:
        phase_field: Spatial phase field

    Returns:
        Coherence length
    """
    phase_field = np.asarray(phase_field, dtype=np.float64)

    # Compute autocorrelation
    mean = np.mean(phase_field)
    var = np.var(phase_field)

    if var < 1e-10:
        # Fully coherent
        return float('inf')

    # Normalized autocorrelation
    n = len(phase_field)
    autocorr = np.correlate(phase_field - mean, phase_field - mean, mode='full')[n-1:] / (var * n)

    # Find where autocorrelation drops to 1/e
    threshold = 1.0 / np.e
    below_threshold = autocorr < threshold

    if np.any(below_threshold):
        coherence_length = float(np.argmax(below_threshold))
    else:
        coherence_length = float(len(autocorr))

    return coherence_length


# Export all
__all__ = [
    'phase_evolution_step',
    'phase_coupling_matrix',
    'phase_synchronization_order',
    'phase_entropy',
    'phase_coherence_length',
]
