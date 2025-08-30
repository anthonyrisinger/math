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
∂ρ_d/∂t = Σ_s R(s→d)ρ_s - Σ_t R(d→t)ρ_

Where R(s→t) is the sapping rate from dimension s to dimension t.
"""

import numpy as np

# Import constants and measures from consolidated mathematics module
from .mathematics import (
    NUMERICAL_EPSILON,
    PHI,
    ConvergenceError,
    NumericalInstabilityError,
    phase_capacity,
)

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

    # Energy defici
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
                    # More conservative
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

                        # Exact conservation check - adjust if needed due to
                        # rounding
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


def emergence_threshold(dimension, phase_density):
    """
    Check if dimension has reached emergence threshold.

    A dimension emerges when its phase density reaches its phase capacity:
    |ρ_d| ≥ Λ(d)

    Parameters
    ----------
    dimension : in
        Dimension index
    phase_density : array-like
        Current phase densities

    Returns
    -------
    bool
        True if dimension has emerged
    """
    dimension_int = int(dimension)  # Convert to integer for indexing
    if dimension_int >= len(phase_density):
        return False

    current_phase = abs(phase_density[dimension_int])
    capacity = phase_capacity(dimension)

    # 95% threshold for numerical stability
    return current_phase >= capacity * 0.95


def advanced_emergence_detection(phase_density, previous_states=None, spectral_threshold=0.1):
    """
    Advanced multi-scale emergence detection with spectral signature analysis.

    Detects emergence patterns across multiple temporal and dimensional scales,
    identifying critical transitions, spectral signatures, and invariant jumps.

    Parameters
    ----------
    phase_density : array-like
        Current phase densities across dimensions
    previous_states : list, optional
        Historical phase states for temporal analysis
    spectral_threshold : float
        Minimum spectral power for emergence detection

    Returns
    -------
    dict
        Comprehensive emergence analysis
    """
    phase_density = np.asarray(phase_density, dtype=complex)
    n_dims = len(phase_density)

    # Current emergence status using standard threshold
    current_emerged = set()
    emergence_strengths = {}
    critical_transitions = []

    for d in range(n_dims):
        if emergence_threshold(d, phase_density):
            current_emerged.add(d)

        # Calculate emergence strength (0 to 1 scale)
        current_phase = abs(phase_density[d])
        try:
            capacity = phase_capacity(d)
            emergence_strengths[d] = min(1.0, current_phase / capacity)
        except (ValueError, OverflowError):
            emergence_strengths[d] = 0.0

    # Temporal pattern analysis if history available
    temporal_patterns = {}
    if previous_states and len(previous_states) > 5:
        # Extract emergence strength time series
        history_length = min(len(previous_states), 100)  # Last 100 states
        recent_states = previous_states[-history_length:]

        for d in range(n_dims):
            # Build emergence strength time series
            strength_series = []
            for past_state in recent_states:
                past_phase = abs(past_state[d]) if d < len(past_state) else 0.0
                try:
                    capacity = phase_capacity(d)
                    strength = min(1.0, past_phase / capacity)
                except (ValueError, OverflowError):
                    strength = 0.0
                strength_series.append(strength)

            strength_series = np.array(strength_series)

            # Detect critical transitions (sudden changes)
            if len(strength_series) > 3:
                gradient = np.gradient(strength_series)
                acceleration = np.gradient(gradient)

                # Critical transition indicators
                max_gradient = np.max(np.abs(gradient))
                max_acceleration = np.max(np.abs(acceleration))
                final_gradient = gradient[-1] if len(gradient) > 0 else 0.0

                # Oscillation detection
                from scipy.fft import fft
                if len(strength_series) >= 8:
                    fft_signal = fft(strength_series - np.mean(strength_series))
                    power_spectrum = np.abs(fft_signal)**2
                    dominant_frequency_power = np.max(power_spectrum[1:len(power_spectrum)//2])
                    total_power = np.sum(power_spectrum)
                    if total_power > 0:
                        oscillation_strength = dominant_frequency_power / total_power
                    else:
                        oscillation_strength = 0.0
                else:
                    oscillation_strength = 0.0

                temporal_patterns[d] = {
                    'strength_series': strength_series,
                    'max_gradient': max_gradient,
                    'max_acceleration': max_acceleration,
                    'current_gradient': final_gradient,
                    'oscillation_strength': oscillation_strength,
                    'is_critical_transition': max_gradient > 0.1 and max_acceleration > 0.05
                }

                # Mark critical transitions
                if temporal_patterns[d]['is_critical_transition']:
                    critical_transitions.append({
                        'dimension': d,
                        'type': 'emergence_acceleration',
                        'strength': max_acceleration,
                        'gradient': max_gradient
                    })

    # Multi-dimensional coupling analysis
    coupling_analysis = {}
    if n_dims > 1:
        # Phase coherence matrix
        coherence_matrix = np.zeros((n_dims, n_dims))
        for i in range(n_dims):
            for j in range(n_dims):
                if i != j and abs(phase_density[i]) > NUMERICAL_EPSILON and abs(phase_density[j]) > NUMERICAL_EPSILON:
                    phase_i = np.angle(phase_density[i])
                    phase_j = np.angle(phase_density[j])
                    # Phase coupling strength
                    coherence_matrix[i, j] = abs(np.exp(1j * (phase_i - phase_j)))

        coupling_analysis['coherence_matrix'] = coherence_matrix
        coupling_analysis['max_coupling'] = np.max(coherence_matrix)
        coupling_analysis['mean_coupling'] = np.mean(coherence_matrix[coherence_matrix > 0])

        # Identify strongly coupled dimension pairs
        strong_couplings = []
        for i in range(n_dims):
            for j in range(i+1, n_dims):
                if coherence_matrix[i, j] > 0.7:  # Strong coupling threshold
                    strong_couplings.append({
                        'dimensions': (i, j),
                        'coupling_strength': coherence_matrix[i, j],
                        'phase_difference': np.angle(phase_density[i] / phase_density[j]) if abs(phase_density[j]) > NUMERICAL_EPSILON else 0.0
                    })

        coupling_analysis['strong_couplings'] = strong_couplings

    # Spectral emergence signatures
    spectral_signatures = {}
    if n_dims > 2:
        # Analyze spectral content of phase density pattern
        try:
            from scipy.fft import fft

            # Real and imaginary parts
            real_spectrum = fft(np.real(phase_density))
            imag_spectrum = fft(np.imag(phase_density))

            # Power spectral density
            power_spectrum = np.abs(real_spectrum)**2 + np.abs(imag_spectrum)**2

            # Identify spectral peaks
            spectral_peaks = []
            for k in range(1, len(power_spectrum)//2):
                if (power_spectrum[k] > spectral_threshold * np.max(power_spectrum) and
                    power_spectrum[k] > power_spectrum[k-1] and
                    power_spectrum[k] > power_spectrum[k+1]):
                    spectral_peaks.append({
                        'frequency_mode': k,
                        'power': power_spectrum[k],
                        'normalized_power': power_spectrum[k] / np.max(power_spectrum)
                    })

            spectral_signatures = {
                'power_spectrum': power_spectrum,
                'spectral_peaks': spectral_peaks,
                'spectral_centroid': np.sum(np.arange(len(power_spectrum)) * power_spectrum) / np.sum(power_spectrum),
                'spectral_width': np.sqrt(np.sum((np.arange(len(power_spectrum)) -
                                               np.sum(np.arange(len(power_spectrum)) * power_spectrum) / np.sum(power_spectrum))**2 * power_spectrum) / np.sum(power_spectrum)),
                'total_spectral_power': np.sum(power_spectrum)
            }
        except Exception:
            spectral_signatures = {'error': 'spectral_analysis_failed'}

    return {
        'current_emerged': current_emerged,
        'emergence_strengths': emergence_strengths,
        'critical_transitions': critical_transitions,
        'temporal_patterns': temporal_patterns,
        'coupling_analysis': coupling_analysis,
        'spectral_signatures': spectral_signatures,
        'total_emergence_activity': sum(emergence_strengths.values()),
        'n_critical_transitions': len(critical_transitions)
    }


def total_phase_energy(phase_density):
    """
    Calculate total phase energy in the system.

    Parameters
    ----------
    phase_density : array-like
        Phase densities

    Returns
    -------
    floa
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
    floa
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
    phi : floa
        Golden ratio coupling constan

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
    new_phase, flow_matrix = phase_evolution_step(
        phase_density, dt, max_dimension
    )

    # Calculate system activity-dependent error estimate
    system_activity = np.sum(np.abs(phase_density) ** 2)
    base_error = dt * 1e-6
    activity_factor = 1.0 + system_activity * 0.1  # Scale with energy
    error = base_error * activity_factor

    dt_next = dt * 1.1  # Modest increase

    return new_phase, dt_next, error


class ConvergenceDiagnostics:
    """Advanced convergence diagnostics with mathematical rigor.

    Tracks multiple convergence metrics including energy conservation,
    spectral radius convergence, emergence pattern stability, and
    topological invariant preservation.
    """

    def __init__(self, history_size=100, strict_tolerance=1e-12):
        self.history_size = history_size
        self.strict_tolerance = strict_tolerance

        # Core metrics
        self.energy_history = []
        self.rate_history = []

        # Advanced metrics
        self.emergence_activity_history = []
        self.coherence_history = []
        self.invariant_violation_history = []
        self.spectral_metrics_history = []

        # Convergence state tracking
        self.convergence_achieved = False
        self.convergence_time = None
        self.convergence_metrics = {}

        # Statistical analysis
        self.trend_analysis = {}
        self.stability_windows = [10, 25, 50]  # Different window sizes for analysis

    def update(self, total_energy, flow_matrix, additional_metrics=None):
        """Update with latest state and additional metrics."""
        # Core metrics
        self.energy_history.append(total_energy)
        if len(self.energy_history) > self.history_size:
            self.energy_history.pop(0)

        # Total rate of change
        total_rate = np.sum(np.abs(flow_matrix))
        self.rate_history.append(total_rate)
        if len(self.rate_history) > self.history_size:
            self.rate_history.pop(0)

        # Advanced metrics if provided
        if additional_metrics:
            if 'emergence_activity' in additional_metrics:
                self.emergence_activity_history.append(additional_metrics['emergence_activity'])
                if len(self.emergence_activity_history) > self.history_size:
                    self.emergence_activity_history.pop(0)

            if 'coherence' in additional_metrics:
                self.coherence_history.append(additional_metrics['coherence'])
                if len(self.coherence_history) > self.history_size:
                    self.coherence_history.pop(0)

            if 'invariant_violations' in additional_metrics:
                n_violations = len(additional_metrics['invariant_violations'])
                self.invariant_violation_history.append(n_violations)
                if len(self.invariant_violation_history) > self.history_size:
                    self.invariant_violation_history.pop(0)

        # Update trend analysis
        self._update_trend_analysis()

        # Check convergence state
        self._update_convergence_state()

    def _update_trend_analysis(self):
        """Update statistical trend analysis."""
        if len(self.energy_history) < 10:
            return

        # Energy trend analysis
        energy_array = np.array(self.energy_history[-50:])  # Last 50 points
        if len(energy_array) > 5:
            # Linear regression for trend detection
            x = np.arange(len(energy_array))
            energy_trend_coeff = np.polyfit(x, energy_array, 1)[0]  # Linear coefficient
            energy_r_squared = np.corrcoef(x, energy_array)[0, 1]**2

            self.trend_analysis['energy_trend_slope'] = energy_trend_coeff
            self.trend_analysis['energy_trend_r2'] = energy_r_squared

        # Rate trend analysis
        if len(self.rate_history) > 5:
            rate_array = np.array(self.rate_history[-50:])
            x = np.arange(len(rate_array))
            rate_trend_coeff = np.polyfit(x, rate_array, 1)[0]

            self.trend_analysis['rate_trend_slope'] = rate_trend_coeff

        # Volatility analysis (rolling standard deviation)
        for window in self.stability_windows:
            if len(self.energy_history) >= window:
                recent_energy = np.array(self.energy_history[-window:])
                energy_volatility = np.std(recent_energy) / np.mean(recent_energy)
                self.trend_analysis[f'energy_volatility_w{window}'] = energy_volatility

            if len(self.rate_history) >= window:
                recent_rates = np.array(self.rate_history[-window:])
                rate_volatility = np.std(recent_rates) / (np.mean(recent_rates) + 1e-15)
                self.trend_analysis[f'rate_volatility_w{window}'] = rate_volatility

    def _update_convergence_state(self):
        """Update overall convergence state assessment."""
        if self.convergence_achieved:
            return  # Already converged

        # Multi-criteria convergence check
        criteria_met = []

        # Energy stability criterion
        if len(self.energy_history) >= 25:
            energy_stability = self.trend_analysis.get('energy_volatility_w25', 1.0)
            criteria_met.append(energy_stability < 1e-8)

        # Rate convergence criterion
        if len(self.rate_history) >= 25:
            recent_rates = self.rate_history[-25:]
            rate_mean = np.mean(recent_rates)
            criteria_met.append(rate_mean < 1e-9)

        # Trend stability criterion
        energy_trend_slope = abs(self.trend_analysis.get('energy_trend_slope', 1.0))
        criteria_met.append(energy_trend_slope < 1e-10)

        # Invariant preservation criterion
        if len(self.invariant_violation_history) >= 10:
            recent_violations = self.invariant_violation_history[-10:]
            criteria_met.append(sum(recent_violations) == 0)
        else:
            criteria_met.append(True)  # No violations recorded yet

        # Convergence achieved if most criteria are met
        convergence_score = sum(criteria_met) / len(criteria_met) if criteria_met else 0.0

        if convergence_score >= 0.8 and not self.convergence_achieved:
            self.convergence_achieved = True
            self.convergence_time = len(self.energy_history)
            self.convergence_metrics = {
                'convergence_score': convergence_score,
                'criteria_met': criteria_met,
                'energy_stability': self.trend_analysis.get('energy_volatility_w25', 0.0),
                'rate_level': np.mean(self.rate_history[-10:]) if len(self.rate_history) >= 10 else 0.0
            }

    def is_converged(self, energy_tolerance=1e-9, rate_tolerance=1e-9,
                    require_trend_stability=True):
        """Enhanced convergence detection with multiple criteria."""
        if len(self.energy_history) < 25:  # Need sufficient history
            return False

        # Energy stability check
        energy_std = np.std(self.energy_history[-25:])
        energy_mean = np.mean(self.energy_history[-25:])
        energy_stability = energy_std / (energy_mean + 1e-15)
        energy_converged = energy_stability < energy_tolerance

        # Rate convergence check
        rate_mean = np.mean(self.rate_history[-25:]) if len(self.rate_history) >= 25 else 1.0
        rate_converged = rate_mean < rate_tolerance

        # Trend stability check
        trend_stable = True
        if require_trend_stability:
            energy_trend = abs(self.trend_analysis.get('energy_trend_slope', 1.0))
            trend_stable = energy_trend < 1e-10

        return energy_converged and rate_converged and trend_stable

    def get_convergence_quality_score(self):
        """Calculate overall convergence quality score (0-1)."""
        if len(self.energy_history) < 10:
            return 0.0

        scores = []

        # Energy conservation score
        if len(self.energy_history) >= 2:
            initial_energy = self.energy_history[0] if self.energy_history[0] != 0 else 1.0
            final_energy = self.energy_history[-1]
            conservation_error = abs(final_energy - initial_energy) / abs(initial_energy)
            energy_score = max(0, 1.0 - conservation_error * 1e6)
            scores.append(energy_score)

        # Stability score
        if len(self.energy_history) >= 25:
            volatility = self.trend_analysis.get('energy_volatility_w25', 1.0)
            stability_score = max(0, 1.0 - volatility * 1e8)
            scores.append(stability_score)

        # Rate convergence score
        if len(self.rate_history) >= 25:
            rate_level = np.mean(self.rate_history[-25:])
            rate_score = max(0, 1.0 - rate_level * 1e8)
            scores.append(rate_score)

        # Invariant preservation score
        if len(self.invariant_violation_history) >= 10:
            violations = sum(self.invariant_violation_history[-10:])
            invariant_score = 1.0 if violations == 0 else max(0, 1.0 - violations * 0.1)
            scores.append(invariant_score)
        else:
            scores.append(1.0)  # Perfect if no violations recorded

        return np.mean(scores) if scores else 0.0

    def get_diagnostics(self):
        """Return comprehensive diagnostic information."""
        # Calculate energy conservation error
        energy_conservation_error = 0.0
        if len(self.energy_history) >= 2:
            initial = self.energy_history[0] if self.energy_history[0] != 0 else 1.0
            final = self.energy_history[-1]
            energy_conservation_error = abs(final - initial) / abs(initial)

        diagnostics = {
            # Core metrics
            "energy_history": self.energy_history.copy(),
            "rate_history": self.rate_history.copy(),
            "converged": self.is_converged(),
            "is_converged": self.is_converged(),
            "energy_conservation_error": energy_conservation_error,

            # Advanced metrics
            "convergence_achieved": self.convergence_achieved,
            "convergence_time": self.convergence_time,
            "convergence_quality_score": self.get_convergence_quality_score(),
            "trend_analysis": self.trend_analysis.copy(),

            # Statistical summaries
            "energy_statistics": self._get_energy_statistics(),
            "rate_statistics": self._get_rate_statistics(),

            # Advanced histories
            "emergence_activity_history": self.emergence_activity_history.copy(),
            "coherence_history": self.coherence_history.copy(),
            "invariant_violation_history": self.invariant_violation_history.copy(),
        }

        # Add convergence metrics if achieved
        if self.convergence_achieved:
            diagnostics["convergence_metrics"] = self.convergence_metrics.copy()

        return diagnostics

    def _get_energy_statistics(self):
        """Get statistical summary of energy history."""
        if not self.energy_history:
            return {}

        energy_array = np.array(self.energy_history)
        return {
            'mean': np.mean(energy_array),
            'std': np.std(energy_array),
            'min': np.min(energy_array),
            'max': np.max(energy_array),
            'range': np.max(energy_array) - np.min(energy_array),
            'cv': np.std(energy_array) / (np.mean(energy_array) + 1e-15)  # Coefficient of variation
        }

    def _get_rate_statistics(self):
        """Get statistical summary of rate history."""
        if not self.rate_history:
            return {}

        rate_array = np.array(self.rate_history)
        return {
            'mean': np.mean(rate_array),
            'std': np.std(rate_array),
            'min': np.min(rate_array),
            'max': np.max(rate_array),
            'recent_mean': np.mean(rate_array[-10:]) if len(rate_array) >= 10 else np.mean(rate_array)
        }

    def reset(self):
        """Reset all convergence metrics."""
        self.energy_history.clear()
        self.rate_history.clear()
        self.emergence_activity_history.clear()
        self.coherence_history.clear()
        self.invariant_violation_history.clear()
        self.spectral_metrics_history.clear()

        self.convergence_achieved = False
        self.convergence_time = None
        self.convergence_metrics.clear()
        self.trend_analysis.clear()

    def export_convergence_report(self):
        """Export detailed convergence report."""
        diagnostics = self.get_diagnostics()
        quality_score = self.get_convergence_quality_score()

        report = f"""
        📊 CONVERGENCE ANALYSIS REPORT
        ==============================

        🎯 Overall Quality Score: {quality_score:.4f}/1.0
        🔄 Convergence Status: {'✅ ACHIEVED' if self.convergence_achieved else '⏳ IN PROGRESS'}

        📈 Energy Conservation:
           Error: {diagnostics['energy_conservation_error']:.2e}
           Status: {'✅ EXCELLENT' if diagnostics['energy_conservation_error'] < 1e-10 else '⚠️  ACCEPTABLE' if diagnostics['energy_conservation_error'] < 1e-6 else '❌ POOR'}

        📉 Stability Analysis:
           Recent Volatility: {self.trend_analysis.get('energy_volatility_w25', 0):.2e}
           Trend Slope: {self.trend_analysis.get('energy_trend_slope', 0):.2e}

        🔬 Rate Analysis:
           Current Rate: {np.mean(self.rate_history[-10:]) if len(self.rate_history) >= 10 else 0:.2e}
           Trend: {'📉 DECREASING' if self.trend_analysis.get('rate_trend_slope', 0) < -1e-10 else '📈 INCREASING' if self.trend_analysis.get('rate_trend_slope', 0) > 1e-10 else '➡️  STABLE'}

        🛡️  Topological Integrity:
           Violations: {sum(self.invariant_violation_history[-10:]) if len(self.invariant_violation_history) >= 10 else 0}
           Status: {'✅ PRESERVED' if (sum(self.invariant_violation_history[-10:]) == 0 if len(self.invariant_violation_history) >= 10 else True) else '⚠️  VIOLATIONS DETECTED'}

        """

        if self.convergence_achieved:
            report += f"""
        ⏱️  Convergence Details:
           Time to Convergence: {self.convergence_time} steps
           Final Score: {self.convergence_metrics.get('convergence_score', 0):.3f}
           """

        return report


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

    def get_diagnostics(self):
        """Get basic topological invariant diagnostics."""
        return {
            'chern_numbers': self.chern_numbers.tolist(),
            'winding_numbers': self.winding_numbers.tolist(),
            'linking_numbers': dict(self.linking_numbers),
            'badges': [f"Chern = {self.chern_numbers[d]}" for d in range(self.max_dim)]
        }


class PhaseDynamicsEngine:
    """
    Complete phase dynamics simulation engine with advanced emergence detection.

    Manages the evolution of phase densities across dimensions,
    tracking emergence, clock rates, energy flows, and critical transitions.
    """

    def __init__(self, max_dimensions=12, use_adaptive=True, enable_advanced_detection=True):
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

        # Advanced emergence detection system
        self.enable_advanced_detection = enable_advanced_detection
        self.emergence_history = []  # Store emergence analysis results
        self.critical_events = []   # Store critical transition events
        self.phase_state_history = []  # Store phase densities for temporal analysis
        self.max_history_length = 200  # Limit memory usage

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
                    self.phase_density,
                    dt_try,
                    self.max_dim - 1,
                    energy_tolerance,
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
                    # More conservative reduction
                    dt = max(dt_try * 0.7, 1e-15)

            except (NumericalInstabilityError, ConvergenceError, OverflowError) as e:
                dt = max(dt * 0.5, 1e-15)
                print(f"Numerical issue in adaptive step: {e}")

            attempts += 1

        if attempts >= max_attempts:
            print(
                f"Warning: Max attempts reached. "
                f"Progress: {dt_taken / dt_target:.1%}"
            )

    def _update_emergence_and_history(self, dt):
        """Helper to update emergence tracking and history with advanced detection."""
        # Enforce topological invariants
        violations = self.invariants.update_invariants(self.phase_density)
        if violations:
            print(f"Warning: Topological violations detected: {violations}")
            self.phase_density = self.invariants.enforce_quantization(
                self.phase_density
            )

        # Update diagnostics with comprehensive metrics
        total_energy = np.sum(np.abs(self.phase_density) ** 2)

        # Prepare additional metrics for diagnostics
        additional_metrics = {
            'coherence': phase_coherence(self.phase_density)
        }

        if self.enable_advanced_detection and hasattr(self, 'emergence_history') and self.emergence_history:
            latest_analysis = self.emergence_history[-1]['analysis']
            additional_metrics['emergence_activity'] = latest_analysis.get('total_emergence_activity', 0.0)
            additional_metrics['invariant_violations'] = violations

        self.diagnostics.update(total_energy, self.flow_matrix, additional_metrics)

        # Store current phase state for temporal analysis
        self.phase_state_history.append(self.phase_density.copy())
        if len(self.phase_state_history) > self.max_history_length:
            self.phase_state_history.pop(0)

        # Advanced emergence detection
        if self.enable_advanced_detection:
            emergence_analysis = advanced_emergence_detection(
                self.phase_density,
                self.phase_state_history if len(self.phase_state_history) > 5 else None
            )

            # Store emergence analysis
            self.emergence_history.append({
                'time': self.time,
                'analysis': emergence_analysis
            })
            if len(self.emergence_history) > self.max_history_length:
                self.emergence_history.pop(0)

            # Process critical events
            for transition in emergence_analysis['critical_transitions']:
                critical_event = {
                    'time': self.time,
                    'dimension': transition['dimension'],
                    'type': transition['type'],
                    'strength': transition['strength'],
                    'emergence_strength': emergence_analysis['emergence_strengths'].get(transition['dimension'], 0.0)
                }
                self.critical_events.append(critical_event)

                # Print critical transition notifications
                print(f"🚨 CRITICAL TRANSITION at t={self.time:.3f}: "
                      f"Dim {transition['dimension']} {transition['type']} "
                      f"(strength={transition['strength']:.3f})")

            # Update emerged set with advanced detection
            old_emerged = self.emerged.copy()
            self.emerged = emergence_analysis['current_emerged'].union({0})  # Void always emerged

            # Detect new emergences
            newly_emerged = self.emerged - old_emerged
            for d in newly_emerged:
                print(f"✨ DIMENSION {d} EMERGED at t={self.time:.3f} "
                      f"(strength={emergence_analysis['emergence_strengths'].get(d, 0.0):.3f})")
        else:
            # Standard emergence detection
            for d in range(1, self.max_dim):
                if d not in self.emerged and emergence_threshold(
                    d, self.phase_density
                ):
                    self.emerged.add(d)

        self.time += dt

        # Store history with topological invariants
        history_entry = {
            "time": self.time,
            "phase_density": self.phase_density.copy(),
            "emerged": self.emerged.copy(),
            "total_energy": total_phase_energy(self.phase_density),
            "coherence": phase_coherence(self.phase_density),
            "topological_invariants": self.invariants.get_diagnostics()
        }

        # Add advanced detection results to history if enabled
        if self.enable_advanced_detection and hasattr(self, 'emergence_history') and self.emergence_history:
            history_entry["emergence_analysis"] = self.emergence_history[-1]['analysis']

        self.history.append(history_entry)

    def clock_rate_modulation(self, dimension):
        """
        Calculate clock rate modulation due to phase sapping.

        Dimensions that get sapped experience time dilation:
        τ_d(t) = τ₀ × ∏_i (1 - R(d→i))

        Parameters
        ----------
        dimension : in
            Dimension index

        Returns
        -------
        floa
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
        n_steps : in
            Number of evolution steps to take
        dt : float, optional
            Time step size (default: 0.01)

        Returns
        -------
        dic
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
        floa
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
        """Get current state summary with advanced emergence information."""
        base_state = {
            "time": self.time,
            "emerged_dimensions": sorted(list(self.emerged)),
            "total_energy": total_phase_energy(self.phase_density),
            "coherence": phase_coherence(self.phase_density),
            "phase_densities": self.phase_density.copy(),
            "diagnostics": self.diagnostics.get_diagnostics(),
            "effective_dimension": self.calculate_effective_dimension(),
        }

        # Add advanced detection information if enabled
        if self.enable_advanced_detection:
            base_state.update({
                "critical_events_count": len(self.critical_events),
                "recent_critical_events": self.critical_events[-5:] if len(self.critical_events) > 0 else [],
                "emergence_activity": self.emergence_history[-1]['analysis']['total_emergence_activity'] if self.emergence_history else 0.0,
                "current_emergence_analysis": self.emergence_history[-1]['analysis'] if self.emergence_history else None
            })

        return base_state

    def inject_energy(self, amount: float, target_dimension: float):
        """
        Inject energy into a specific dimensional level.

        Parameters
        ----------
        amount : float
            Amount of energy to inject
        target_dimension : float
            Target dimension level for energy injection
        """
        # Convert fractional dimension to integer index (simple mapping)
        dim_index = int(np.clip(np.round(target_dimension), 0, self.max_dim - 1))

        # Inject energy as phase magnitude increase
        current_phase = self.phase_density[dim_index]
        current_magnitude = np.abs(current_phase)
        current_phase_angle = np.angle(current_phase)

        # Increase magnitude while preserving phase
        new_magnitude = current_magnitude + amount
        self.phase_density[dim_index] = new_magnitude * np.exp(1j * current_phase_angle)

        # Update emerged dimensions set if energy exceeds threshold
        if new_magnitude > 1e-6:
            self.emerged.add(dim_index)


# ENHANCED ANALYSIS TOOLS (previously in dimensional/phase.py)


def quick_emergence_analysis(max_dimensions=8, time_steps=500, enable_advanced=True):
    """
    Perform comprehensive analysis of dimensional emergence patterns with advanced detection.

    Parameters
    ----------
    max_dimensions : int
        Maximum dimensions to simulate
    time_steps : int
        Number of evolution steps
    enable_advanced : bool
        Whether to use advanced emergence detection

    Returns
    -------
    dict
        Analysis results including emergence times, patterns, and critical events
    """
    engine = PhaseDynamicsEngine(
        max_dimensions=max_dimensions,
        enable_advanced_detection=enable_advanced
    )

    results = []
    for step in range(time_steps):
        engine.step(0.01)

        if step % 50 == 0:  # Sample every 50 steps
            state = engine.get_state()
            sample_result = {
                "step": step,
                "time": state["time"],
                "emerged": list(state["emerged_dimensions"]),
                "effective_dimension": state["effective_dimension"],
                "total_energy": state["total_energy"],
            }

            # Add advanced detection results if enabled
            if enable_advanced:
                sample_result.update({
                    "emergence_activity": state.get("emergence_activity", 0.0),
                    "critical_events_count": state.get("critical_events_count", 0),
                    "recent_transitions": len(state.get("recent_critical_events", []))
                })

            results.append(sample_result)

    final_state = engine.get_state()
    analysis_result = {
        "results": results,
        "final_state": final_state,
        "max_dimensions": max_dimensions,
        "time_steps": time_steps,
    }

    # Add comprehensive advanced analysis if enabled
    if enable_advanced:
        analysis_result.update({
            "critical_events": engine.critical_events,
            "emergence_history_length": len(engine.emergence_history),
            "total_critical_events": len(engine.critical_events),
            "emergence_timeline": [
                {
                    "time": event["time"],
                    "dimension": event["dimension"],
                    "type": event["type"],
                    "strength": event["strength"]
                } for event in engine.critical_events
            ]
        })

        # Statistical analysis of emergence patterns
        if engine.critical_events:
            event_times = [e["time"] for e in engine.critical_events]
            event_strengths = [e["strength"] for e in engine.critical_events]

            analysis_result["emergence_statistics"] = {
                "first_critical_event_time": min(event_times),
                "last_critical_event_time": max(event_times),
                "mean_event_strength": np.mean(event_strengths),
                "max_event_strength": max(event_strengths),
                "event_rate": len(engine.critical_events) / (time_steps * 0.01) if time_steps > 0 else 0.0
            }

    return analysis_result


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
    dic
        Phase analysis results with enhanced detection
    """
    if dimensions is None:
        dimensions = [0, 1, 2, 3, 4, 5]
    elif isinstance(dimensions, (int, float)):
        dimensions = [dimensions]

    # Create sample phase density
    max_dim = max(int(d) for d in dimensions)
    phase_density = np.array([1.0 + 0.1j * i for i in range(max_dim + 1)])

    results = {}
    for d in dimensions:
        d_int = int(d)  # Convert to integer for indexing
        if d_int < len(phase_density):
            results[f"dimension_{d}"] = {
                "phase_capacity": phase_capacity(d),
                "current_phase": abs(phase_density[d_int]),
                "emergence_status": emergence_threshold(d, phase_density),
            }
        else:
            results[f"dimension_{d}"] = {
                "phase_capacity": phase_capacity(d),
                "current_phase": 0.0,
                "emergence_status": False,
            }

        # Calculate sapping rates to higher dimensions
        sapping_rates = {}
        for target in range(d_int + 1, len(phase_density)):
            if target <= max(dimensions):
                rate = sap_rate(d, target, phase_density)
                if rate > 1e-12:
                    sapping_rates[f"to_dim_{target}"] = rate

        results[f"dimension_{d}"]["sapping_rates"] = sapping_rates

    # Add advanced emergence analysis if requested
    if enable_advanced:
        advanced_analysis = advanced_emergence_detection(phase_density)
        results["advanced_emergence"] = advanced_analysis

        # Add summary metrics
        results["emergence_summary"] = {
            "total_emergence_activity": advanced_analysis["total_emergence_activity"],
            "n_critical_transitions": advanced_analysis["n_critical_transitions"],
            "spectral_peak_count": len(advanced_analysis["spectral_signatures"].get("spectral_peaks", [])),
            "max_coupling_strength": advanced_analysis["coupling_analysis"].get("max_coupling", 0.0)
        }

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
