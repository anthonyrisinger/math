#!/usr/bin/env python3
"""
Advanced Spectral Analysis Module
=================================

Spectral theory extensions for dimensional mathematics, moving FAR BEYOND
basic V/S/C measures into advanced mathematical physics and signal analysis.

Provides:
- Eigenvalue decomposition of phase evolution operators
- Spectral density functions for dimensional resonances
- Harmonic analysis on fractal dimension sets
- Fourier transforms of emergence patterns
- Wavelet analysis in dimension-time space
- Resonance detection in dimensional emergence
- Frequency domain analysis of phase dynamics
- Spectral signatures of critical points
"""

import warnings
from typing import Any, Optional

import numpy as np
from scipy import linalg
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert

# Import from consolidated mathematics
from .mathematics import (
    NUMERICAL_EPSILON,
    PI,
    ball_volume,
    complexity_measure,
)

# Import phase dynamics
from .phase import PhaseDynamicsEngine, phase_evolution_step, total_phase_energy

# =============================================================================
# OPERATOR SPECTRAL ANALYSIS
# =============================================================================

class DimensionalOperator:
    """
    Advanced spectral analysis of dimensional evolution operators.

    Provides eigenvalue decomposition, spectral density analysis,
    and operator function calculations for dimensional systems.
    """

    def __init__(self, max_dimensions: int = 16):
        self.max_dimensions = max_dimensions
        self.dimension_space = np.arange(max_dimensions)
        self._operator_matrix = None
        self._eigenvalues = None
        self._eigenvectors = None

    def construct_evolution_operator(self, dt: float = 0.01) -> np.ndarray:
        """
        Construct the discrete evolution operator matrix from phase dynamics.

        The evolution operator U satisfies: ψ(t+dt) = U ψ(t)
        where ψ is the phase density vector across dimensions.
        """
        n = self.max_dimensions
        operator = np.zeros((n, n), dtype=complex)

        # Construct operator from phase evolution step function
        for i in range(n):
            # Create unit vector for dimension i
            test_state = np.zeros(n, dtype=complex)
            test_state[i] = 1.0

            # Apply one evolution step
            evolved_state, _ = phase_evolution_step(test_state, dt, n)

            # Store result as column i of the operator matrix
            operator[:, i] = evolved_state

        self._operator_matrix = operator
        return operator

    def spectral_decomposition(self, dt: float = 0.01) -> dict[str, Any]:
        """
        Complete eigenvalue decomposition of the evolution operator.

        Returns eigenvalues, eigenvectors, and spectral properties.
        """
        if self._operator_matrix is None:
            self.construct_evolution_operator(dt)

        # Eigenvalue decomposition
        eigenvals, eigenvecs = linalg.eig(self._operator_matrix)

        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(np.abs(eigenvals))[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        self._eigenvalues = eigenvals
        self._eigenvectors = eigenvecs

        # Spectral properties
        spectral_radius = np.max(np.abs(eigenvals))
        condition_number = np.max(np.abs(eigenvals)) / np.min(np.abs(eigenvals[eigenvals != 0]))

        # Classify eigenvalues
        stable_modes = eigenvals[np.abs(eigenvals) < 1.0]
        unstable_modes = eigenvals[np.abs(eigenvals) > 1.0]
        critical_modes = eigenvals[np.abs(np.abs(eigenvals) - 1.0) < NUMERICAL_EPSILON]

        return {
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'spectral_radius': spectral_radius,
            'condition_number': condition_number,
            'stable_modes': stable_modes,
            'unstable_modes': unstable_modes,
            'critical_modes': critical_modes,
            'n_stable': len(stable_modes),
            'n_unstable': len(unstable_modes),
            'n_critical': len(critical_modes),
        }

    def operator_function(self, func_name: str, **kwargs) -> np.ndarray:
        """
        Calculate operator functions: exp(A), sin(A), cos(A), etc.

        Uses spectral decomposition for numerical stability.
        """
        if self._eigenvalues is None:
            self.spectral_decomposition()

        eigenvals = self._eigenvalues
        eigenvecs = self._eigenvectors

        # Apply function to eigenvalues
        if func_name == 'exp':
            func_eigenvals = np.exp(eigenvals)
        elif func_name == 'sin':
            func_eigenvals = np.sin(eigenvals)
        elif func_name == 'cos':
            func_eigenvals = np.cos(eigenvals)
        elif func_name == 'log':
            # Handle log carefully for complex eigenvalues
            func_eigenvals = np.log(eigenvals + NUMERICAL_EPSILON)
        elif func_name == 'sqrt':
            func_eigenvals = np.sqrt(eigenvals)
        elif func_name == 'power':
            power = kwargs.get('power', 2.0)
            func_eigenvals = eigenvals ** power
        else:
            raise ValueError(f"Unknown operator function: {func_name}")

        # Reconstruct operator function: f(A) = V f(Λ) V^(-1)
        try:
            func_operator = eigenvecs @ np.diag(func_eigenvals) @ linalg.inv(eigenvecs)
            return func_operator
        except linalg.LinAlgError:
            warnings.warn("Eigenvector matrix is singular, using pseudoinverse")
            func_operator = eigenvecs @ np.diag(func_eigenvals) @ linalg.pinv(eigenvecs)
            return func_operator


# =============================================================================
# SPECTRAL DENSITY ANALYSIS
# =============================================================================

def dimensional_spectral_density(dimensions: np.ndarray, measure_func=None) -> dict[str, Any]:
    """
    Calculate spectral density of dimensional measures.

    Analyzes the frequency content of dimensional measure functions
    to identify resonances and characteristic scales.
    """
    if measure_func is None:
        measure_func = complexity_measure

    # Calculate measure values
    measure_values = np.array([measure_func(d) for d in dimensions])

    # Remove DC component and apply window
    measure_values = measure_values - np.mean(measure_values)
    window = np.hanning(len(measure_values))
    windowed_signal = measure_values * window

    # Fourier transform
    fft_values = fft(windowed_signal)
    freqs = fftfreq(len(dimensions), d=dimensions[1] - dimensions[0])

    # Power spectral density
    psd = np.abs(fft_values) ** 2

    # Find peaks in spectrum
    peak_indices = []
    for i in range(1, len(psd) - 1):
        if psd[i] > psd[i-1] and psd[i] > psd[i+1] and psd[i] > 0.1 * np.max(psd):
            peak_indices.append(i)

    peak_frequencies = freqs[peak_indices]
    peak_powers = psd[peak_indices]

    # Spectral centroid (frequency weighted by power)
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)

    # Spectral bandwidth
    spectral_bandwidth = np.sqrt(np.sum((freqs - spectral_centroid)**2 * psd) / np.sum(psd))

    return {
        'frequencies': freqs,
        'power_spectral_density': psd,
        'peak_frequencies': peak_frequencies,
        'peak_powers': peak_powers,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'total_power': np.sum(psd),
        'signal_values': measure_values,
        'windowed_signal': windowed_signal,
    }


# =============================================================================
# HARMONIC ANALYSIS ON FRACTAL DIMENSION SETS
# =============================================================================

def fractal_harmonic_analysis(
    base_dimension: float,
    fractal_exponent: float,
    n_harmonics: int = 10,
    n_points: int = 1000
) -> dict[str, Any]:
    """
    Harmonic analysis on fractal dimensional sets.

    Analyzes harmonic content of measures evaluated on fractal
    dimensional sequences like Cantor sets, Julia sets, etc.
    """
    # Generate fractal dimensional sequence
    # Using a generalized Cantor-like construction
    dimensions = []
    current_d = base_dimension

    for n in range(n_points):
        dimensions.append(current_d)
        # Fractal iteration: d_{n+1} = d_n + (fractal_exponent - d_n) / (n + 1)
        current_d = current_d + (fractal_exponent - current_d) / (n + 2)

    dimensions = np.array(dimensions)

    # Calculate measure on fractal set
    measure_values = np.array([complexity_measure(d) for d in dimensions])

    # Harmonic decomposition using least squares
    harmonics = {}
    freqs = []
    amplitudes = []
    phases = []

    # Time parameter for harmonic analysis
    t = np.linspace(0, 2*PI, len(dimensions))

    # Fit harmonics
    for k in range(1, n_harmonics + 1):
        freq = k * 2 * PI / (t[-1] - t[0])
        freqs.append(freq)

        # Fit A_k cos(kωt) + B_k sin(kωt)
        cos_component = np.cos(k * t)
        sin_component = np.sin(k * t)

        # Least squares fit
        design_matrix = np.column_stack([cos_component, sin_component])
        coeffs, residuals, rank, s = linalg.lstsq(design_matrix, measure_values)

        A_k, B_k = coeffs
        amplitude = np.sqrt(A_k**2 + B_k**2)
        phase = np.arctan2(B_k, A_k)

        amplitudes.append(amplitude)
        phases.append(phase)

        harmonics[f'harmonic_{k}'] = {
            'frequency': freq,
            'amplitude': amplitude,
            'phase': phase,
            'A_coefficient': A_k,
            'B_coefficient': B_k,
        }

    # Harmonic reconstruction
    reconstructed = np.zeros_like(measure_values)
    for k in range(1, n_harmonics + 1):
        harmonic = harmonics[f'harmonic_{k}']
        reconstructed += harmonic['amplitude'] * np.cos(k * t + harmonic['phase'])

    # Reconstruction error
    reconstruction_error = np.mean((measure_values - reconstructed)**2)

    return {
        'fractal_dimensions': dimensions,
        'measure_values': measure_values,
        'harmonics': harmonics,
        'frequencies': np.array(freqs),
        'amplitudes': np.array(amplitudes),
        'phases': np.array(phases),
        'reconstructed': reconstructed,
        'reconstruction_error': reconstruction_error,
        'time_parameter': t,
        'base_dimension': base_dimension,
        'fractal_exponent': fractal_exponent,
    }


# =============================================================================
# WAVELET ANALYSIS IN DIMENSION-TIME SPACE
# =============================================================================

def dimensional_wavelet_analysis(
    time_series: np.ndarray,
    dimension_series: np.ndarray,
    wavelet_type: str = 'morlet',
    scales: Optional[np.ndarray] = None
) -> dict[str, Any]:
    """
    Wavelet analysis of dimensional evolution in time.

    Provides time-frequency analysis of dimensional phase dynamics
    to identify emergent patterns and critical transitions.
    """
    if scales is None:
        scales = np.logspace(0, 2, 50)  # 50 scales from 1 to 100

    n_points = len(time_series)
    n_scales = len(scales)

    # Wavelet transform implementation (simplified Morlet)
    if wavelet_type == 'morlet':
        # Morlet wavelet: ψ(t) = π^(-1/4) exp(iω₀t) exp(-t²/2)
        omega0 = 6.0  # Central frequency
        wavelet_transform = np.zeros((n_scales, n_points), dtype=complex)

        for i, scale in enumerate(scales):
            for j, t in enumerate(time_series):
                # Evaluate wavelet at all time points for this scale
                wavelet_values = np.zeros(n_points, dtype=complex)
                for k, tau in enumerate(time_series):
                    t_scaled = (tau - t) / scale
                    if abs(t_scaled) < 5:  # Limit support
                        wavelet_values[k] = (PI**(-0.25) / np.sqrt(scale) *
                                           np.exp(1j * omega0 * t_scaled) *
                                           np.exp(-t_scaled**2 / 2))

                # Convolution with dimension series
                wavelet_transform[i, j] = np.sum(dimension_series * wavelet_values)

    else:
        raise ValueError(f"Wavelet type '{wavelet_type}' not implemented")

    # Wavelet power (magnitude squared)
    wavelet_power = np.abs(wavelet_transform)**2

    # Global wavelet spectrum (average over time)
    global_spectrum = np.mean(wavelet_power, axis=1)

    # Time-averaged scale spectrum
    scale_spectrum = np.mean(wavelet_power, axis=0)

    # Find dominant scales and times
    max_power_idx = np.unravel_index(np.argmax(wavelet_power), wavelet_power.shape)
    dominant_scale = scales[max_power_idx[0]]
    dominant_time = time_series[max_power_idx[1]]

    # Ridge analysis (follow maximum power across scales)
    ridge_indices = []
    for j in range(n_points):
        ridge_indices.append(np.argmax(wavelet_power[:, j]))
    ridge_scales = scales[ridge_indices]

    return {
        'wavelet_transform': wavelet_transform,
        'wavelet_power': wavelet_power,
        'scales': scales,
        'time_series': time_series,
        'dimension_series': dimension_series,
        'global_spectrum': global_spectrum,
        'scale_spectrum': scale_spectrum,
        'dominant_scale': dominant_scale,
        'dominant_time': dominant_time,
        'ridge_scales': ridge_scales,
        'ridge_indices': ridge_indices,
        'total_energy': np.sum(wavelet_power),
    }


# =============================================================================
# RESONANCE DETECTION IN DIMENSIONAL EMERGENCE
# =============================================================================

def detect_dimensional_resonances(
    phase_engine: PhaseDynamicsEngine,
    n_steps: int = 1000,
    dt: float = 0.01,
    frequency_resolution: int = 500
) -> dict[str, Any]:
    """
    Detect resonances in dimensional emergence patterns.

    Analyzes phase dynamics evolution to identify resonant frequencies
    and characteristic oscillation modes in dimensional emergence.
    """
    # Run phase dynamics simulation
    evolution_data = []
    time_points = []

    for step in range(n_steps):
        state = phase_engine.get_state()
        evolution_data.append(state['phase_densities'].copy())
        time_points.append(step * dt)
        phase_engine.step(dt)

    evolution_data = np.array(evolution_data)  # Shape: (n_steps, n_dimensions)
    time_points = np.array(time_points)

    # Analyze each dimension for resonances
    resonances = {}

    for dim in range(evolution_data.shape[1]):
        dimension_evolution = evolution_data[:, dim]

        # Extract amplitude and phase evolution
        analytic_signal = hilbert(np.real(dimension_evolution))
        amplitude_evolution = np.abs(analytic_signal)
        phase_evolution = np.angle(analytic_signal)

        # Instantaneous frequency
        phase_diff = np.diff(np.unwrap(phase_evolution))
        instantaneous_freq = phase_diff / (2 * PI * dt)

        # Spectral analysis of amplitude evolution
        amplitude_fft = fft(amplitude_evolution - np.mean(amplitude_evolution))
        freq_axis = fftfreq(len(amplitude_evolution), dt)
        amplitude_spectrum = np.abs(amplitude_fft)

        # Find resonance peaks
        positive_freqs = freq_axis[freq_axis > 0]
        positive_spectrum = amplitude_spectrum[freq_axis > 0]

        # Peak detection
        peak_indices = []
        for i in range(1, len(positive_spectrum) - 1):
            if (positive_spectrum[i] > positive_spectrum[i-1] and
                positive_spectrum[i] > positive_spectrum[i+1] and
                positive_spectrum[i] > 0.1 * np.max(positive_spectrum)):
                peak_indices.append(i)

        resonant_frequencies = positive_freqs[peak_indices]
        resonant_powers = positive_spectrum[peak_indices]

        # Quality factors (Q = f / Δf)
        q_factors = []
        for peak_idx in peak_indices:
            peak_freq = positive_freqs[peak_idx]
            peak_power = positive_spectrum[peak_idx]

            # Find half-power bandwidth
            half_power = peak_power / 2
            left_idx = peak_idx
            right_idx = peak_idx

            # Search left
            while left_idx > 0 and positive_spectrum[left_idx] > half_power:
                left_idx -= 1

            # Search right
            while right_idx < len(positive_spectrum) - 1 and positive_spectrum[right_idx] > half_power:
                right_idx += 1

            if right_idx > left_idx:
                bandwidth = positive_freqs[right_idx] - positive_freqs[left_idx]
                q_factor = peak_freq / bandwidth if bandwidth > 0 else np.inf
            else:
                q_factor = np.inf

            q_factors.append(q_factor)

        resonances[f'dimension_{dim}'] = {
            'dimension_evolution': dimension_evolution,
            'amplitude_evolution': amplitude_evolution,
            'phase_evolution': phase_evolution,
            'instantaneous_frequency': instantaneous_freq,
            'resonant_frequencies': resonant_frequencies,
            'resonant_powers': resonant_powers,
            'q_factors': np.array(q_factors),
            'frequency_axis': positive_freqs,
            'amplitude_spectrum': positive_spectrum,
        }

    # Global resonance analysis
    total_energy_evolution = np.array([total_phase_energy(state) for state in evolution_data])
    global_resonance = detect_global_resonances(total_energy_evolution, time_points)

    return {
        'resonances': resonances,
        'global_resonance': global_resonance,
        'evolution_data': evolution_data,
        'time_points': time_points,
        'n_dimensions': evolution_data.shape[1],
        'simulation_parameters': {
            'n_steps': n_steps,
            'dt': dt,
            'total_time': n_steps * dt,
        }
    }


def detect_global_resonances(energy_evolution: np.ndarray, time_points: np.ndarray) -> dict[str, Any]:
    """Helper function for global energy resonance analysis."""
    dt = time_points[1] - time_points[0]

    # Remove trend
    detrended_energy = energy_evolution - np.mean(energy_evolution)

    # Spectral analysis
    energy_fft = fft(detrended_energy)
    freq_axis = fftfreq(len(energy_evolution), dt)
    energy_spectrum = np.abs(energy_fft)

    # Find global resonance peaks
    positive_freqs = freq_axis[freq_axis > 0]
    positive_spectrum = energy_spectrum[freq_axis > 0]

    peak_indices = []
    for i in range(1, len(positive_spectrum) - 1):
        if (positive_spectrum[i] > positive_spectrum[i-1] and
            positive_spectrum[i] > positive_spectrum[i+1] and
            positive_spectrum[i] > 0.05 * np.max(positive_spectrum)):
            peak_indices.append(i)

    global_resonant_frequencies = positive_freqs[peak_indices] if peak_indices else np.array([])
    global_resonant_powers = positive_spectrum[peak_indices] if peak_indices else np.array([])

    return {
        'energy_evolution': energy_evolution,
        'detrended_energy': detrended_energy,
        'global_resonant_frequencies': global_resonant_frequencies,
        'global_resonant_powers': global_resonant_powers,
        'frequency_axis': positive_freqs,
        'energy_spectrum': positive_spectrum,
    }


# =============================================================================
# SPECTRAL SIGNATURES OF CRITICAL POINTS
# =============================================================================

def analyze_critical_point_spectrum(
    measure_func=None,
    dimension_range: tuple[float, float] = (0.1, 15.0),
    n_points: int = 2000,
    critical_tolerance: float = 1e-6
) -> dict[str, Any]:
    """
    Analyze spectral signatures around critical points of dimensional measures.

    Identifies how spectral properties change near critical points
    like volume peak, surface peak, complexity peak, etc.
    """
    if measure_func is None:
        measure_func = complexity_measure

    # Generate high-resolution dimension grid
    dimensions = np.linspace(dimension_range[0], dimension_range[1], n_points)
    measure_values = np.array([measure_func(d) for d in dimensions])

    # Find critical points (peaks, troughs, inflections)
    from scipy.signal import find_peaks

    # First derivative (finite difference)
    grad = np.gradient(measure_values, dimensions)

    # Second derivative
    hessian = np.gradient(grad, dimensions)

    # Find peaks and troughs
    peaks, peak_properties = find_peaks(measure_values, height=0.01 * np.max(measure_values))
    troughs, trough_properties = find_peaks(-measure_values, height=0.01 * np.max(-measure_values))

    # Find inflection points (zero crossings of second derivative)
    inflection_points = []
    for i in range(len(hessian) - 1):
        if hessian[i] * hessian[i+1] < 0:  # Sign change
            # Linear interpolation to find zero crossing
            d_inflection = dimensions[i] - hessian[i] * (dimensions[i+1] - dimensions[i]) / (hessian[i+1] - hessian[i])
            inflection_points.append(d_inflection)

    inflection_points = np.array(inflection_points)

    # Analyze spectral properties around each critical point
    critical_analyses = {}

    all_critical_points = np.concatenate([
        dimensions[peaks],
        dimensions[troughs],
        inflection_points
    ])

    critical_types = (['peak'] * len(peaks) +
                     ['trough'] * len(troughs) +
                     ['inflection'] * len(inflection_points))

    for i, (critical_d, cp_type) in enumerate(zip(all_critical_points, critical_types)):
        # Define analysis window around critical point
        window_size = 1.0  # Analysis window ±1 dimension unit
        window_mask = np.abs(dimensions - critical_d) <= window_size

        if np.sum(window_mask) < 10:  # Need sufficient points
            continue

        window_dims = dimensions[window_mask]
        window_measures = measure_values[window_mask]

        # Local spectral analysis
        local_spectrum = dimensional_spectral_density(window_dims, lambda d: measure_func(d))

        # Critical point properties
        critical_value = measure_func(critical_d)
        local_grad = np.interp(critical_d, dimensions, grad)
        local_hess = np.interp(critical_d, dimensions, hessian)

        # Curvature analysis
        curvature = local_hess / (1 + local_grad**2)**(3/2)

        # Spectral width around critical point
        spectral_width = local_spectrum['spectral_bandwidth']
        spectral_centroid = local_spectrum['spectral_centroid']

        critical_analyses[f'{cp_type}_{i}'] = {
            'critical_dimension': critical_d,
            'critical_value': critical_value,
            'type': cp_type,
            'gradient': local_grad,
            'hessian': local_hess,
            'curvature': curvature,
            'spectral_width': spectral_width,
            'spectral_centroid': spectral_centroid,
            'local_spectrum': local_spectrum,
            'window_dimensions': window_dims,
            'window_measures': window_measures,
        }

    return {
        'dimensions': dimensions,
        'measure_values': measure_values,
        'gradient': grad,
        'hessian': hessian,
        'peaks': dimensions[peaks],
        'troughs': dimensions[troughs],
        'inflection_points': inflection_points,
        'critical_analyses': critical_analyses,
        'global_spectrum': dimensional_spectral_density(dimensions, measure_func),
    }


def analyze_dimensional_transitions(
    phase_engine=None,
    transition_threshold: float = 0.1,
    n_analysis_steps: int = 100
) -> dict[str, Any]:
    """
    Analyze spectral signatures of dimensional transitions in phase dynamics.

    Identifies critical transitions where dimensions emerge/collapse and
    analyzes their spectral signatures for invariant preservation.

    Parameters
    ----------
    phase_engine : PhaseDynamicsEngine, optional
        Phase dynamics engine. If None, creates new one.
    transition_threshold : float
        Minimum change in emergence strength to classify as transition
    n_analysis_steps : int
        Number of evolution steps for analysis

    Returns
    -------
    dict
        Comprehensive transition analysis with spectral signatures
    """
    if phase_engine is None:
        from .phase import PhaseDynamicsEngine
        phase_engine = PhaseDynamicsEngine(max_dimensions=8, enable_advanced_detection=True)

    # Collect phase evolution data
    evolution_data = []
    emergence_data = []
    time_points = []

    for step in range(n_analysis_steps):
        # Record current state
        current_state = phase_engine.get_state()
        evolution_data.append(current_state['phase_densities'].copy())
        time_points.append(current_state['time'])

        # Record emergence analysis if available
        if 'current_emergence_analysis' in current_state and current_state['current_emergence_analysis']:
            emergence_data.append(current_state['current_emergence_analysis'])

        # Evolve system
        phase_engine.step(0.01)

    evolution_data = np.array(evolution_data)  # Shape: (n_steps, n_dimensions)
    time_points = np.array(time_points)

    # Identify transition points
    transition_events = []
    if emergence_data:
        for i in range(1, len(emergence_data)):
            prev_strengths = emergence_data[i-1].get('emergence_strengths', {})
            curr_strengths = emergence_data[i].get('emergence_strengths', {})

            # Check for significant changes in emergence strengths
            for dim in set(prev_strengths.keys()).union(curr_strengths.keys()):
                prev_strength = prev_strengths.get(dim, 0.0)
                curr_strength = curr_strengths.get(dim, 0.0)
                strength_change = abs(curr_strength - prev_strength)

                if strength_change > transition_threshold:
                    transition_events.append({
                        'step': i,
                        'time': time_points[i],
                        'dimension': dim,
                        'strength_change': strength_change,
                        'prev_strength': prev_strength,
                        'curr_strength': curr_strength,
                        'transition_type': 'emergence' if curr_strength > prev_strength else 'collapse'
                    })

    # Spectral analysis of transitions
    transition_spectra = {}
    for event in transition_events:
        event_step = event['step']
        dimension = event['dimension']

        # Extract time series around transition
        window_size = 20  # ±20 steps around transition
        start_idx = max(0, event_step - window_size)
        end_idx = min(len(evolution_data), event_step + window_size + 1)

        if end_idx - start_idx > 10:  # Need sufficient data
            window_evolution = evolution_data[start_idx:end_idx, dimension]
            window_times = time_points[start_idx:end_idx]

            # Spectral analysis of the transition
            transition_spectrum = analyze_transition_spectral_signature(
                window_evolution, window_times, event
            )

            event_key = f"transition_{event['step']}_dim_{dimension}"
            transition_spectra[event_key] = {
                'event_info': event,
                'spectral_analysis': transition_spectrum,
                'window_data': {
                    'evolution': window_evolution,
                    'times': window_times,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                }
            }

    # Global spectral evolution analysis
    global_spectral_evolution = analyze_global_spectral_evolution(
        evolution_data, time_points
    )

    return {
        'evolution_data': evolution_data,
        'time_points': time_points,
        'emergence_data': emergence_data,
        'transition_events': transition_events,
        'transition_spectra': transition_spectra,
        'global_spectral_evolution': global_spectral_evolution,
        'n_transitions': len(transition_events),
        'analysis_parameters': {
            'n_analysis_steps': n_analysis_steps,
            'transition_threshold': transition_threshold,
            'max_dimensions': phase_engine.max_dim
        }
    }


def analyze_transition_spectral_signature(
    phase_evolution: np.ndarray,
    time_points: np.ndarray,
    transition_event: dict
) -> dict[str, Any]:
    """
    Analyze spectral signature of a specific dimensional transition.

    Parameters
    ----------
    phase_evolution : array
        Phase density evolution for a single dimension
    time_points : array
        Time points corresponding to evolution
    transition_event : dict
        Information about the transition event

    Returns
    -------
    dict
        Spectral signature analysis
    """
    # Extract amplitude and phase components
    amplitudes = np.abs(phase_evolution)
    phases = np.angle(phase_evolution)

    # Time derivatives
    dt = time_points[1] - time_points[0] if len(time_points) > 1 else 0.01
    amplitude_rate = np.gradient(amplitudes, dt)
    phase_rate = np.gradient(np.unwrap(phases), dt)

    # Spectral analysis of amplitude evolution
    from scipy.fft import fft, fftfreq

    # Detrend amplitude signal
    amplitude_detrended = amplitudes - np.mean(amplitudes)

    # Apply window to reduce spectral leakage
    window = np.hanning(len(amplitude_detrended))
    windowed_amplitude = amplitude_detrended * window

    # FFT analysis
    amplitude_fft = fft(windowed_amplitude)
    freq_axis = fftfreq(len(amplitudes), dt)
    amplitude_spectrum = np.abs(amplitude_fft)

    # Find dominant frequencies
    positive_freqs = freq_axis[freq_axis > 0]
    positive_spectrum = amplitude_spectrum[freq_axis > 0]

    # Peak detection in spectrum
    spectral_peaks = []
    for i in range(1, len(positive_spectrum) - 1):
        if (positive_spectrum[i] > positive_spectrum[i-1] and
            positive_spectrum[i] > positive_spectrum[i+1] and
            positive_spectrum[i] > 0.1 * np.max(positive_spectrum)):
            spectral_peaks.append({
                'frequency': positive_freqs[i],
                'power': positive_spectrum[i],
                'relative_power': positive_spectrum[i] / np.max(positive_spectrum)
            })

    # Spectral moments
    spectral_centroid = np.sum(positive_freqs * positive_spectrum) / np.sum(positive_spectrum) if np.sum(positive_spectrum) > 0 else 0.0
    spectral_spread = np.sqrt(np.sum((positive_freqs - spectral_centroid)**2 * positive_spectrum) / np.sum(positive_spectrum)) if np.sum(positive_spectrum) > 0 else 0.0

    # Phase coherence analysis
    phase_coherence = np.abs(np.mean(np.exp(1j * phases))) if len(phases) > 0 else 0.0

    # Transition sharpness metrics
    max_amplitude_rate = np.max(np.abs(amplitude_rate))
    max_phase_rate = np.max(np.abs(phase_rate))

    return {
        'amplitude_evolution': amplitudes,
        'phase_evolution': phases,
        'amplitude_rate': amplitude_rate,
        'phase_rate': phase_rate,
        'spectral_peaks': spectral_peaks,
        'spectral_centroid': spectral_centroid,
        'spectral_spread': spectral_spread,
        'phase_coherence': phase_coherence,
        'max_amplitude_rate': max_amplitude_rate,
        'max_phase_rate': max_phase_rate,
        'frequency_axis': positive_freqs,
        'amplitude_spectrum': positive_spectrum,
        'total_spectral_power': np.sum(positive_spectrum),
        'transition_sharpness': max_amplitude_rate * max_phase_rate  # Combined sharpness metric
    }


def analyze_global_spectral_evolution(
    evolution_data: np.ndarray,
    time_points: np.ndarray
) -> dict[str, Any]:
    """
    Analyze global spectral evolution across all dimensions.

    Parameters
    ----------
    evolution_data : array, shape (n_steps, n_dimensions)
        Phase density evolution across all dimensions
    time_points : array
        Time points for evolution

    Returns
    -------
    dict
        Global spectral evolution analysis
    """
    n_steps, n_dimensions = evolution_data.shape

    # Compute total system energy evolution
    total_energies = np.sum(np.abs(evolution_data)**2, axis=1)

    # Compute dimensional entropy (measure of energy distribution)
    dimensional_entropies = []
    for step in range(n_steps):
        energies = np.abs(evolution_data[step, :])**2
        total_energy = np.sum(energies)
        if total_energy > NUMERICAL_EPSILON:
            probabilities = energies / total_energy
            # Shannon entropy
            entropy = -np.sum(probabilities * np.log(probabilities + NUMERICAL_EPSILON))
            dimensional_entropies.append(entropy)
        else:
            dimensional_entropies.append(0.0)

    dimensional_entropies = np.array(dimensional_entropies)

    # Spectral analysis of global quantities
    dt = time_points[1] - time_points[0] if len(time_points) > 1 else 0.01

    from scipy.fft import fft, fftfreq

    # Energy evolution spectrum
    energy_detrended = total_energies - np.mean(total_energies)
    energy_fft = fft(energy_detrended)
    freq_axis = fftfreq(len(total_energies), dt)
    energy_spectrum = np.abs(energy_fft)

    # Entropy evolution spectrum
    entropy_detrended = dimensional_entropies - np.mean(dimensional_entropies)
    entropy_fft = fft(entropy_detrended)
    entropy_spectrum = np.abs(entropy_fft)

    # Cross-dimensional coherence matrix
    coherence_matrix = np.zeros((n_dimensions, n_dimensions))
    for i in range(n_dimensions):
        for j in range(n_dimensions):
            if i != j:
                # Cross-correlation in frequency domain
                signal_i = evolution_data[:, i]
                signal_j = evolution_data[:, j]

                # Compute coherence
                fft_i = fft(signal_i - np.mean(signal_i))
                fft_j = fft(signal_j - np.mean(signal_j))

                cross_spectrum = fft_i * np.conj(fft_j)
                auto_spectrum_i = fft_i * np.conj(fft_i)
                auto_spectrum_j = fft_j * np.conj(fft_j)

                coherence = np.abs(cross_spectrum)**2 / (auto_spectrum_i * auto_spectrum_j + NUMERICAL_EPSILON)
                coherence_matrix[i, j] = np.mean(coherence.real)

    # Global spectral features
    positive_freqs = freq_axis[freq_axis > 0]
    positive_energy_spectrum = energy_spectrum[freq_axis > 0]
    positive_entropy_spectrum = entropy_spectrum[freq_axis > 0]

    return {
        'total_energy_evolution': total_energies,
        'dimensional_entropy_evolution': dimensional_entropies,
        'energy_spectrum': positive_energy_spectrum,
        'entropy_spectrum': positive_entropy_spectrum,
        'frequency_axis': positive_freqs,
        'coherence_matrix': coherence_matrix,
        'max_coherence': np.max(coherence_matrix),
        'mean_coherence': np.mean(coherence_matrix[coherence_matrix > 0]),
        'energy_spectral_centroid': np.sum(positive_freqs * positive_energy_spectrum) / np.sum(positive_energy_spectrum) if np.sum(positive_energy_spectrum) > 0 else 0.0,
        'entropy_spectral_centroid': np.sum(positive_freqs * positive_entropy_spectrum) / np.sum(positive_entropy_spectrum) if np.sum(positive_entropy_spectrum) > 0 else 0.0
    }


# =============================================================================
# CONVENIENCE FUNCTIONS FOR RESEARCH
# =============================================================================

def quick_spectral_analysis(max_dimensions: int = 12, dt: float = 0.01) -> dict[str, Any]:
    """Quick spectral analysis of phase dynamics system."""
    operator = DimensionalOperator(max_dimensions)
    spectral_data = operator.spectral_decomposition(dt)

    # Add operator functions
    exp_operator = operator.operator_function('exp')
    sin_operator = operator.operator_function('sin')

    return {
        'spectral_decomposition': spectral_data,
        'operator_exponential': exp_operator,
        'operator_sine': sin_operator,
        'max_dimensions': max_dimensions,
        'time_step': dt,
    }


def analyze_emergence_spectrum(
    n_steps: int = 500,
    dt: float = 0.01,
    max_dimensions: int = 8,
    include_transitions: bool = True
) -> dict[str, Any]:
    """Comprehensive spectral analysis of dimensional emergence with transition detection."""

    # Create phase engine with advanced detection
    from .phase import PhaseDynamicsEngine
    engine = PhaseDynamicsEngine(max_dimensions, enable_advanced_detection=True)

    # Resonance detection
    resonance_data = detect_dimensional_resonances(engine, n_steps, dt)

    # Dimensional transition analysis
    transition_analysis = None
    if include_transitions:
        # Reset engine for transition analysis
        engine = PhaseDynamicsEngine(max_dimensions, enable_advanced_detection=True)
        transition_analysis = analyze_dimensional_transitions(
            phase_engine=engine,
            n_analysis_steps=n_steps
        )

    # Spectral signatures of measures
    complexity_spectrum = analyze_critical_point_spectrum(
        complexity_measure, (0.1, 12.0), 1000
    )

    volume_spectrum = analyze_critical_point_spectrum(
        ball_volume, (0.1, 12.0), 1000
    )

    result = {
        'resonance_analysis': resonance_data,
        'complexity_spectrum': complexity_spectrum,
        'volume_spectrum': volume_spectrum,
        'simulation_parameters': {
            'n_steps': n_steps,
            'dt': dt,
            'max_dimensions': max_dimensions,
        }
    }

    if transition_analysis:
        result['transition_analysis'] = transition_analysis
        result['n_detected_transitions'] = transition_analysis.get('n_transitions', 0)

        # Add transition summary statistics
        if transition_analysis['transition_events']:
            transition_times = [e['time'] for e in transition_analysis['transition_events']]
            transition_strengths = [e['strength_change'] for e in transition_analysis['transition_events']]

            result['transition_statistics'] = {
                'first_transition_time': min(transition_times),
                'last_transition_time': max(transition_times),
                'mean_transition_strength': np.mean(transition_strengths),
                'max_transition_strength': max(transition_strengths),
                'transition_rate': len(transition_analysis['transition_events']) / (n_steps * dt)
            }

    return result


# =============================================================================
# MODULE TEST
# =============================================================================

def test_spectral_module():
    """Test spectral analysis capabilities."""
    try:
        # Test basic operator analysis
        operator = DimensionalOperator(4)
        spectral_data = operator.spectral_decomposition()

        # Test spectral density
        dims = np.linspace(0.1, 10, 100)
        density_data = dimensional_spectral_density(dims)

        # Test critical point analysis
        critical_data = analyze_critical_point_spectrum(complexity_measure, (1, 8), 200)

        return {
            'module_status': 'operational',
            'operator_eigenvalues': len(spectral_data['eigenvalues']),
            'spectral_peaks': len(density_data['peak_frequencies']),
            'critical_points': len(critical_data['critical_analyses']),
            'tests_passed': True,
        }

    except Exception as e:
        return {
            'module_status': 'error',
            'error_message': str(e),
            'tests_passed': False,
        }


if __name__ == "__main__":
    # Test the spectral module
    test_results = test_spectral_module()
    print("SPECTRAL ANALYSIS MODULE TEST")
    print("=" * 50)
    for key, value in test_results.items():
        print(f"{key}: {value}")
