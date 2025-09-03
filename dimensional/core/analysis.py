"""
Phase Dynamics Analysis Functions
==================================

Analysis and emergence detection functions extracted from phase.py.
"""

import numpy as np
from scipy.fft import fft

from .constants import NUMERICAL_EPSILON
from .core import phase_capacity, phase_coherence


def analytical_continuation(func, z):
    """Analytical continuation of a function."""
    # Simple placeholder implementation
    return func(z)


def pole_structure(func, z_range):
    """Analyze pole structure of a function."""
    # Simple placeholder implementation
    poles = []
    for z in z_range:
        try:
            val = func(z)
            if abs(val) > 1e10:
                poles.append(z)
        except:
            poles.append(z)
    return poles


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
    dimension = int(dimension)
    phase_density = np.asarray(phase_density, dtype=complex)

    if dimension < 0 or dimension >= len(phase_density):
        return False

    current_magnitude = abs(phase_density[dimension])

    try:
        threshold = phase_capacity(dimension)
    except (ValueError, OverflowError):
        return False

    return current_magnitude >= threshold


def advanced_emergence_detection(phase_density, previous_states=None, spectral_threshold=0.1):
    """
    Advanced emergence detection using multiple indicators.

    Combines energy thresholds, spectral analysis, and temporal patterns
    to detect dimensional emergence more accurately.

    Parameters
    ----------
    phase_density : array-like
        Current phase densities
    previous_states : list, optional
        Previous phase density states for temporal analysis
    spectral_threshold : float
        Threshold for spectral emergence detection

    Returns
    -------
    dict
        Emergence analysis results
    """
    phase_density = np.asarray(phase_density, dtype=complex)
    n_dims = len(phase_density)

    results = {
        'emerged_dimensions': set(),
        'emergence_candidates': set(),
        'spectral_peaks': [],
        'coherence_map': {},
        'temporal_emergence_rate': 0.0,
        'critical_transitions': []
    }

    # 1. Energy-based emergence detection
    for dim in range(n_dims):
        if emergence_threshold(dim, phase_density):
            results['emerged_dimensions'].add(dim)

    # 2. Spectral emergence detection
    energies = np.abs(phase_density) ** 2
    if len(energies) > 4:  # Need sufficient data for FFT
        spectrum = np.abs(fft(energies))[:n_dims//2]

        # Find spectral peaks
        mean_spectrum = np.mean(spectrum)
        std_spectrum = np.std(spectrum)
        peak_threshold = mean_spectrum + 2 * std_spectrum

        peaks = np.where(spectrum > peak_threshold)[0]
        results['spectral_peaks'] = list(peaks)

        # Spectral emergence candidates
        for peak in peaks:
            if spectrum[peak] > spectral_threshold * np.max(spectrum):
                if peak < n_dims:
                    results['emergence_candidates'].add(int(peak))

    # 3. Coherence analysis for neighboring dimensions
    for dim in range(n_dims - 1):
        if energies[dim] > NUMERICAL_EPSILON and energies[dim + 1] > NUMERICAL_EPSILON:
            local_coherence = phase_coherence(phase_density[dim:dim+2])
            results['coherence_map'][dim] = local_coherence

            # High coherence suggests coupled emergence
            if local_coherence > 0.8 and dim not in results['emerged_dimensions']:
                results['emergence_candidates'].add(dim)

    # 4. Temporal analysis (if history provided)
    if previous_states and len(previous_states) > 1:
        # Calculate emergence rate over time
        current_emerged = len(results['emerged_dimensions'])

        # Look at previous states
        emergence_history = []
        for state in previous_states[-10:]:  # Last 10 states
            prev_emerged = set()
            for dim in range(min(len(state), n_dims)):
                if emergence_threshold(dim, state):
                    prev_emerged.add(dim)
            emergence_history.append(len(prev_emerged))

        if emergence_history:
            # Emergence rate (dimensions per timestep)
            recent_rate = (current_emerged - emergence_history[0]) / len(emergence_history)
            results['temporal_emergence_rate'] = recent_rate

            # Detect critical transitions
            for i in range(1, len(emergence_history)):
                if emergence_history[i] > emergence_history[i-1]:
                    # New dimension emerged
                    results['critical_transitions'].append({
                        'timestep': len(previous_states) - len(emergence_history) + i,
                        'type': 'emergence',
                        'count_change': emergence_history[i] - emergence_history[i-1]
                    })

    return results


def quick_phase_analysis(dimensions=None, enable_advanced=True):
    """
    Quick analysis of phase capacities, sapping rates, and emergence patterns.

    Parameters
    ----------
    dimensions : list or float, optional
        Dimensions to analyze. Defaults to [0, 1, 2, 3, 4, 5]
    enable_advanced : bool
        Whether to include advanced emergence detection

    Returns
    -------
    dict
        Phase analysis results with enhanced detection
    """
    import numpy as np

    from ..mathematics import phase_capacity
    from .core import sap_rate

    if dimensions is None:
        dimensions = [0, 1, 2, 3, 4, 5]
    elif isinstance(dimensions, (int, float)):
        dimensions = [dimensions]

    # Create sample phase density
    max_dim = max(int(d) for d in dimensions) + 1
    phase_density = np.array([1.0 + 0.1j * i for i in range(max_dim)])

    results = {}
    for d in dimensions:
        d_int = int(d)
        if d_int < len(phase_density):
            results[f"dimension_{d}"] = {
                "phase_capacity": phase_capacity(d),
                "current_phase": abs(phase_density[d_int]),
                "emergence_status": emergence_threshold(d_int, phase_density),
            }
        else:
            results[f"dimension_{d}"] = {
                "phase_capacity": phase_capacity(d),
                "current_phase": 0.0,
                "emergence_status": False,
            }

        # Calculate sapping rates to higher dimensions
        sapping_rates = {}
        for target in range(d_int + 1, min(len(phase_density), max(dimensions) + 1)):
            rate = sap_rate(d, target, phase_density)
            if rate > 1e-12:
                sapping_rates[f"to_dim_{target}"] = rate

        results[f"dimension_{d}"]["sapping_rates"] = sapping_rates

    # Add advanced detection if enabled
    if enable_advanced:
        results["advanced_analysis"] = advanced_emergence_detection(phase_density)

    return results
