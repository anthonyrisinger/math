#!/usr/bin/env python3
"""
Mathematical Functions - Consolidated Core
==========================================

All core mathematical functions consolidated from core/ modules.
Provides gamma functions, dimensional measures, phase dynamics,
and morphic mathematics in a single unified module.
"""

import warnings

import numpy as np
from scipy.special import digamma as scipy_digamma
from scipy.special import gamma as scipy_gamma
from scipy.special import gammaln

from .constants import (
    CRITICAL_DIMENSIONS,
    GAMMA_OVERFLOW_THRESHOLD,
    LOG_SPACE_THRESHOLD,
    NUMERICAL_EPSILON,
    PHI,
    PI,
    PSI,
)

# =============================================================================
# GAMMA FUNCTION FAMILY
# =============================================================================


def gamma_safe(z):
    """Numerically stable gamma function with proper edge case handling."""
    z = np.asarray(z)

    # Handle edge cases
    if np.any(z == 0):
        result = np.full_like(z, np.inf, dtype=float)
        mask = z != 0
        if np.any(mask):
            result[mask] = gamma_safe(z[mask])
        return result if z.ndim > 0 else float(result)

    # Handle negative integers (poles)
    if np.any((z < 0) & (np.abs(z - np.round(z)) < NUMERICAL_EPSILON)):
        result = np.full_like(z, np.inf, dtype=float)
        mask = ~((z < 0) & (np.abs(z - np.round(z)) < NUMERICAL_EPSILON))
        if np.any(mask):
            result[mask] = gamma_safe(z[mask])
        return result if z.ndim > 0 else float(result)

    # Use log-space for large values
    if np.any(np.abs(z) > GAMMA_OVERFLOW_THRESHOLD):
        large_mask = np.abs(z) > GAMMA_OVERFLOW_THRESHOLD
        result = np.zeros_like(z, dtype=float)

        if np.any(~large_mask):
            result[~large_mask] = scipy_gamma(z[~large_mask])

        if np.any(large_mask):
            log_gamma_vals = gammaln(z[large_mask])
            exp_mask = log_gamma_vals < LOG_SPACE_THRESHOLD
            if np.any(exp_mask):
                if large_mask.ndim > 0:
                    large_indices = np.where(large_mask)[0]
                    safe_indices = large_indices[exp_mask]
                    result[safe_indices] = np.exp(log_gamma_vals[exp_mask])
                else:
                    result[()] = np.exp(log_gamma_vals)

            inf_mask = log_gamma_vals >= LOG_SPACE_THRESHOLD
            if np.any(inf_mask):
                if large_mask.ndim > 0:
                    large_indices = np.where(large_mask)[0]
                    inf_indices = large_indices[inf_mask]
                    result[inf_indices] = np.inf
                else:
                    result[()] = np.inf

        return result if z.ndim > 0 else float(result)

    return scipy_gamma(z)


def gammaln_safe(z):
    """Safe log-gamma function."""
    z = np.asarray(z)

    if np.any(z <= 0):
        if np.any(np.abs(z - np.round(z)) < NUMERICAL_EPSILON):
            result = np.full_like(z, -np.inf, dtype=float)
            mask = ~(np.abs(z - np.round(z)) < NUMERICAL_EPSILON)
            if np.any(mask):
                result[mask] = gammaln_safe(z[mask])
            return result if z.ndim > 0 else float(result)

    return gammaln(z)


def digamma_safe(z):
    """Safe digamma function (psi function)."""
    z = np.asarray(z)

    if np.any(z <= 0):
        if np.any(np.abs(z - np.round(z)) < NUMERICAL_EPSILON):
            result = np.full_like(z, -np.inf, dtype=float)
            mask = ~(np.abs(z - np.round(z)) < NUMERICAL_EPSILON)
            if np.any(mask):
                result[mask] = digamma_safe(z[mask])
            return result if z.ndim > 0 else float(result)

    return scipy_digamma(z)


def beta_function(a, b):
    """Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)."""
    # Use log-space for stability
    if np.any(np.abs(np.asarray(a)) > GAMMA_OVERFLOW_THRESHOLD / 2) or np.any(
        np.abs(np.asarray(b)) > GAMMA_OVERFLOW_THRESHOLD / 2
    ):
        log_beta = gammaln_safe(a) + gammaln_safe(b) - gammaln_safe(a + b)
        return np.exp(log_beta)

    return gamma_safe(a) * gamma_safe(b) / gamma_safe(a + b)


def factorial_extension(n):
    """Factorial extension: n! = Γ(n+1)"""
    return gamma_safe(np.asarray(n) + 1)


# Greek letter aliases for gamma functions
gamma = gamma_safe  # Standard name
ln_gamma = gammaln_safe  # Standard name
digamma = digamma_safe  # Standard name


def abs_gamma(z):
    return np.abs(gamma_safe(z))  # Standard name


# Keep Greek letter aliases for advanced users
γ = gamma_safe
ln_γ = gammaln_safe
ψ = digamma_safe
abs_γ = abs_gamma

# =============================================================================
# DIMENSIONAL MEASURES
# =============================================================================


def _validate_dimension(d, function_name="measure"):
    """Validate dimensional input and issue warnings."""
    d_array = np.asarray(d)

    if np.any(d_array < 0):
        negative_values = d_array[d_array < 0]
        if len(negative_values) == 1:
            from . import DimensionalError
            raise DimensionalError(
                f"Negative dimension d={negative_values[0]:.3f} in {function_name}(). "
                f"Mathematical extension not supported."
            )

    if np.any(d_array > 100):
        large_values = d_array[d_array > 100]
        if len(large_values) == 1:
            warnings.warn(
                f"Large dimension d={
                    large_values[0]:.1f} in {function_name}() "
                f"may underflow to zero.",
                UserWarning,
                stacklevel=3,
            )


def ball_volume(d):
    """Volume of unit d-dimensional ball: V_d = π^(d/2) / Γ(d/2 + 1)"""
    _validate_dimension(d, "ball_volume")
    d = np.asarray(d)

    # Handle d = 0 exactly
    if np.any(np.abs(d) < NUMERICAL_EPSILON):
        result = np.ones_like(d, dtype=float)
        mask = np.abs(d) >= NUMERICAL_EPSILON
        if np.any(mask):
            result[mask] = ball_volume(d[mask])
        return result if d.ndim > 0 else float(result)

    # Use log space for large d
    if np.any(d > 170):
        large_mask = d > 170
        result = np.zeros_like(d, dtype=float)

        if np.any(~large_mask):
            d_small = d[~large_mask]
            log_vol = (d_small / 2) * np.log(PI) - gammaln_safe(
                d_small / 2 + 1
            )
            result[~large_mask] = np.exp(log_vol)

        if np.any(large_mask):
            d_large = d[large_mask]
            log_vol = (d_large / 2) * np.log(PI) - gammaln_safe(
                d_large / 2 + 1
            )
            result[large_mask] = np.exp(np.real(log_vol))

        return result if d.ndim > 0 else float(result)

    return PI ** (d / 2) / gamma_safe(d / 2 + 1)


def sphere_surface(d):
    """Surface area of (d-1)-sphere: S_d = 2π^(d/2) / Γ(d/2)"""
    _validate_dimension(d, "sphere_surface")
    d = np.asarray(d)

    # Handle d = 0 and d = 1 exactly
    if np.any(np.abs(d) < NUMERICAL_EPSILON):
        result = np.full_like(d, 2.0, dtype=float)
        mask = np.abs(d) >= NUMERICAL_EPSILON
        if np.any(mask):
            result[mask] = sphere_surface(d[mask])
        return result if d.ndim > 0 else float(result)

    if np.any(np.abs(d - 1) < NUMERICAL_EPSILON):
        result = np.full_like(d, 2.0, dtype=float)
        mask = np.abs(d - 1) >= NUMERICAL_EPSILON
        if np.any(mask):
            result[mask] = sphere_surface(d[mask])
        return result if d.ndim > 0 else float(result)

    # Use log space for large d
    if np.any(d > 170):
        large_mask = d > 170
        result = np.zeros_like(d, dtype=float)

        if np.any(~large_mask):
            d_small = d[~large_mask]
            log_surf = (
                np.log(2)
                + (d_small / 2) * np.log(PI)
                - gammaln_safe(d_small / 2)
            )
            result[~large_mask] = np.exp(log_surf)

        if np.any(large_mask):
            d_large = d[large_mask]
            log_surf = (
                np.log(2)
                + (d_large / 2) * np.log(PI)
                - gammaln_safe(d_large / 2)
            )
            result[large_mask] = np.exp(np.real(log_surf))

        return result if d.ndim > 0 else float(result)

    return 2 * PI ** (d / 2) / gamma_safe(d / 2)


def complexity_measure(d):
    """V×S complexity: C_d = V_d × S_d"""
    _validate_dimension(d, "complexity_measure")
    return ball_volume(d) * sphere_surface(d)


def ratio_measure(d):
    """Surface/Volume ratio: R_d = S_d / V_d"""
    _validate_dimension(d, "ratio_measure")
    v = ball_volume(d)
    s = sphere_surface(d)
    return s / np.maximum(v, NUMERICAL_EPSILON)


def phase_capacity(d):
    """Phase capacity: Λ(d) = V_d (with minimum bound)"""
    _validate_dimension(d, "phase_capacity")
    return ball_volume(np.maximum(d, 0.01))


# Single letter aliases for measures
v = ball_volume
s = sphere_surface
c = complexity_measure
r = ratio_measure

# Uppercase aliases for backward compatibility
V = ball_volume
S = sphere_surface
C = complexity_measure
R = ratio_measure


def find_peak(measure_func, d_min=0.1, d_max=15, num_points=5000):
    """Find peak (maximum) of a measure function."""
    d_range = np.linspace(d_min, d_max, num_points)
    values = np.array([measure_func(d) for d in d_range])

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.nan, np.nan

    finite_values = values[finite_mask]
    finite_d = d_range[finite_mask]
    peak_idx = np.argmax(finite_values)

    return finite_d[peak_idx], finite_values[peak_idx]


def find_all_peaks(d_min=0.1, d_max=15.0, resolution=10000):
    """Find peaks of all standard measures."""
    results = {}

    vol_peak_d, vol_peak_val = find_peak(ball_volume, d_min, d_max, resolution)
    results["volume_peak"] = (vol_peak_d, vol_peak_val)

    surf_peak_d, surf_peak_val = find_peak(
        sphere_surface, d_min, d_max, resolution
    )
    results["surface_peak"] = (surf_peak_d, surf_peak_val)

    comp_peak_d, comp_peak_val = find_peak(
        complexity_measure, d_min, d_max, resolution
    )
    results["complexity_peak"] = (comp_peak_d, comp_peak_val)

    return results


def is_critical_dimension(d, tolerance=1e-6):
    """Check if d is near any critical dimension value."""
    for name, value in CRITICAL_DIMENSIONS.items():
        if abs(d - value) < tolerance:
            return True, name
    return False, None


# =============================================================================
# PHASE DYNAMICS
# =============================================================================


def sap_rate(source, target, phase_density, phi=PHI, min_distance=1e-3):
    """Calculate energy-based sapping rate between dimensions."""
    source = float(source)
    target = float(target)

    if source >= target:
        return 0.0

    # Distance calculation with regularization
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

    # Energy deficit and rate calculation
    energy_deficit = capacity_energy - target_energy
    equilibrium_factor = energy_deficit / capacity_energy
    distance_factor = 1.0 / regularized_distance

    try:
        frequency_ratio = np.sqrt((target + 1) / (source + 1))
    except (OverflowError, ZeroDivisionError):
        frequency_ratio = 1.0

    rate = (
        energy_deficit * distance_factor * frequency_ratio * equilibrium_factor
    )
    return float(min(rate, 0.5))  # Conservative rate limiting


def phase_evolution_step(phase_density, dt, max_dimension=None):
    """Energy-conserving phase evolution step."""
    phase_density = np.asarray(phase_density, dtype=complex)
    n_dims = len(phase_density)

    if max_dimension is None:
        max_dimension = n_dims - 1
    else:
        max_dimension = min(max_dimension, n_dims - 1)

    # Work with energies and phases separately
    energies = np.abs(phase_density) ** 2
    phases = np.angle(phase_density)
    initial_total_energy = np.sum(energies)

    # Track energy transfers
    flow_matrix = np.zeros((n_dims, n_dims))

    for target in range(1, max_dimension + 1):
        for source in range(target):
            if energies[source] > NUMERICAL_EPSILON:
                rate = sap_rate(source, target, phase_density)

                if rate > NUMERICAL_EPSILON:
                    energy_transfer_rate = rate * energies[source]
                    energy_transfer = energy_transfer_rate * dt
                    max_energy_transfer = energies[source] * 0.1
                    energy_transfer = min(energy_transfer, max_energy_transfer)

                    if energy_transfer > NUMERICAL_EPSILON:
                        old_source_energy = energies[source]
                        old_target_energy = energies[target]

                        energies[source] -= energy_transfer
                        energies[target] += energy_transfer
                        energies[source] = max(0, energies[source])

                        # Exact conservation
                        actual_transfer = old_source_energy - energies[source]
                        energies[target] = old_target_energy + actual_transfer
                        flow_matrix[source, target] = actual_transfer

    # Energy conservation correction
    final_total_energy = np.sum(energies)
    energy_error = final_total_energy - initial_total_energy

    if (
        abs(energy_error) > NUMERICAL_EPSILON
        and final_total_energy > NUMERICAL_EPSILON
    ):
        correction_factor = initial_total_energy / final_total_energy
        energies *= correction_factor

    # Reconstruct complex phase densities
    new_phase_density = np.sqrt(energies) * np.exp(1j * phases)
    return new_phase_density, flow_matrix


def total_phase_energy(phase_density):
    """Calculate total phase energy: Σ |ρ_d|²"""
    return float(np.sum(np.abs(phase_density) ** 2))


def phase_coherence(phase_density):
    """Calculate phase coherence across dimensions."""
    phase_density = np.asarray(phase_density, dtype=complex)

    nonzero_mask = np.abs(phase_density) > NUMERICAL_EPSILON
    if not np.any(nonzero_mask):
        return 0.0

    phases = np.angle(phase_density[nonzero_mask])
    mean_phase_vector = np.mean(np.exp(1j * phases))
    return float(abs(mean_phase_vector))


def emergence_threshold(dimension, phase_density):
    """Check if dimension has reached emergence threshold."""
    if dimension >= len(phase_density):
        return False

    current_phase = abs(phase_density[dimension])
    capacity = phase_capacity(dimension)
    return current_phase >= capacity * 0.95


def quick_phase_analysis(max_dimensions=8, n_steps=100, dt=0.01):
    """Quick phase dynamics analysis."""
    from .validation import PhaseDynamicsEngine

    engine = PhaseDynamicsEngine(max_dimensions)
    results = engine.evolve(n_steps, dt)

    return {
        "emerged_dimensions": results["current_emerged"],
        "final_energy": results["final_energy"],
        "energy_conservation_error": results["energy_conservation"],
        "effective_dimension": engine.calculate_effective_dimension(),
    }


# =============================================================================
# MORPHIC MATHEMATICS
# =============================================================================


def morphic_polynomial_roots(k, mode="shifted"):
    """Find real roots of morphic polynomial families."""
    if mode == "shifted":
        coeffs = [1.0, 0.0, -(2.0 - k), -1.0]  # τ³ - (2-k)τ - 1
    elif mode == "simple":
        coeffs = [1.0, 0.0, -k, -1.0]  # τ³ - kτ - 1
    else:
        raise ValueError("mode must be 'shifted' or 'simple'")

    roots = np.roots(coeffs)
    real_roots = roots.real[np.abs(roots.imag) < NUMERICAL_EPSILON]
    return np.sort(real_roots)[::-1]


def discriminant(k, mode="shifted"):
    """Discriminant of morphic polynomial."""
    if mode == "shifted":
        a, c, d = 1.0, -(2.0 - k), -1.0
        return -4 * a * (c**3) - 27 * (a**2) * (d**2)
    elif mode == "simple":
        a, c, d = 1.0, -k, -1.0
        return -4 * a * (c**3) - 27 * (a**2) * (d**2)
    else:
        raise ValueError("mode must be 'shifted' or 'simple'")


def k_perfect_circle(mode="shifted"):
    """Parameter value where τ = 1 is a root (perfect circle case)."""
    if mode == "shifted":
        return 2.0
    elif mode == "simple":
        return 0.0
    else:
        raise ValueError("mode must be 'shifted' or 'simple'")


def k_discriminant_zero(mode="shifted"):
    """Parameter value where discriminant equals zero."""
    if mode == "shifted":
        return 2.0 - (27.0 / 4.0) ** (1.0 / 3.0)
    elif mode == "simple":
        return (27.0 / 4.0) ** (1.0 / 3.0)
    else:
        raise ValueError("mode must be 'shifted' or 'simple'")


def golden_ratio_properties():
    """Verify and return golden ratio properties."""
    phi = PHI
    psi = PSI

    return {
        "phi": phi,
        "psi": psi,
        "phi_squared_equals_phi_plus_one": abs(phi**2 - (phi + 1))
        < NUMERICAL_EPSILON,
        "psi_squared_equals_one_minus_psi": abs(psi**2 - (1 - psi))
        < NUMERICAL_EPSILON,
        "phi_times_psi_equals_one": abs(phi * psi - 1) < NUMERICAL_EPSILON,
        "phi_minus_psi_equals_one": abs((phi - psi) - 1.0) < NUMERICAL_EPSILON,
        "phi_plus_psi_equals_sqrt5": abs((phi + psi) - np.sqrt(5))
        < NUMERICAL_EPSILON,
    }


# =============================================================================
# PHASE DYNAMICS ENGINE CLASS
# =============================================================================


class PhaseDynamicsEngine:
    """Complete phase dynamics simulation engine."""

    def __init__(self, max_dimensions=12, use_adaptive=True):
        self.max_dim = max_dimensions
        self.phase_density = np.zeros(max_dimensions, dtype=complex)
        self.phase_density[0] = 1.0  # Start with unity at the void

        self.flow_matrix = np.zeros((max_dimensions, max_dimensions))
        self.emerged = {0}  # Void always exists
        self.time = 0.0
        self.history = []
        self.use_adaptive = use_adaptive

    def step(self, dt):
        """Advance simulation by one time step."""
        self.phase_density, self.flow_matrix = phase_evolution_step(
            self.phase_density, dt, self.max_dim - 1
        )

        # Check for new emergences
        for d in range(1, self.max_dim):
            if d not in self.emerged and emergence_threshold(
                d, self.phase_density
            ):
                self.emerged.add(d)

        self.time += d

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

    def evolve(self, n_steps, dt=0.01):
        """Evolve the system for n_steps."""
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
        """Calculate effective dimension from phase density distribution."""
        energies = np.abs(self.phase_density) ** 2
        total_energy = np.sum(energies)

        if total_energy < NUMERICAL_EPSILON:
            return 0.0

        weighted_sum = 0.0
        for d in range(len(energies)):
            weight = energies[d] / total_energy
            weighted_sum += d * weight

        return float(weighted_sum)

    def inject(self, dimension, energy):
        """Inject energy into a dimension."""
        if dimension < len(self.phase_density):
            self.phase_density[dimension] += energy

    def get_state(self):
        """Get current state summary."""
        return {
            "time": self.time,
            "emerged_dimensions": sorted(list(self.emerged)),
            "total_energy": total_phase_energy(self.phase_density),
            "coherence": phase_coherence(self.phase_density),
            "phase_densities": self.phase_density.copy(),
            "effective_dimension": self.calculate_effective_dimension(),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_gamma_analysis(z_values=None):
    """Quick gamma function analysis."""
    if z_values is None:
        z_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    results = {}
    for z in z_values:
        results[z] = {
            "gamma": gamma_safe(z),
            "ln_gamma": gammaln_safe(z),
            "digamma": digamma_safe(z),
            "abs_gamma": abs_gamma(z),
        }

    return results


def quick_measure_analysis(dimensions=None):
    """Quick dimensional measures analysis."""
    if dimensions is None:
        dimensions = [0, 1, 2, 3, 4, 5, 6]

    results = {}
    for d in dimensions:
        results[d] = {
            "volume": ball_volume(d),
            "surface": sphere_surface(d),
            "complexity": complexity_measure(d),
            "ratio": ratio_measure(d),
            "is_critical": is_critical_dimension(d),
        }

    return results


if __name__ == "__main__":
    # Validate mathematical properties
    print("MATHEMATICAL FUNCTIONS VALIDATION")
    print("=" * 50)

    # Test gamma functions
    assert abs(gamma_safe(0.5) - np.sqrt(np.pi)) < 1e-10, "Γ(1/2) ≠ √π"
    assert abs(gamma_safe(1.0) - 1.0) < 1e-10, "Γ(1) ≠ 1"
    assert abs(gamma_safe(2.0) - 1.0) < 1e-10, "Γ(2) ≠ 1!"

    # Test measures for standard dimensions
    for d in [0, 1, 2, 3]:
        v_d = ball_volume(d)
        s_d = sphere_surface(d)
        assert v_d > 0, f"Invalid volume for dimension {d}"
        assert s_d > 0, f"Invalid surface for dimension {d}"

    # Test golden ratio properties
    props = golden_ratio_properties()
    assert props[
        "phi_squared_equals_phi_plus_one"
    ], "Golden ratio property failed"

    print("All mathematical validations passed!")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def setup_3d_axis(
    ax, title="", xlim=None, ylim=None, zlim=None, grid=True, grid_alpha=0.3
):
    """
    Set up 3D axis with standard orthographic projection and golden viewing angle.

    Parameters
    ----------
    ax : Axes3D
        DEPRECATED: Matplotlib eliminated. Use modern visualization.
    title : str
        Axis title
    xlim, ylim, zlim : tuple, optional
        Axis limits as (min, max)
    grid : bool
        Whether to show grid
    grid_alpha : floa
        Grid transparency

    Returns
    -------
    Axes3D
        Configured 3D axis
    """
    # Set orthographic projection
    ax.set_proj_type("ortho")

    # Set golden ratio viewing angle (from constants)
    from .constants import BOX_ASPECT, VIEW_AZIM, VIEW_ELEV

    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    # Set box aspect ratio for accurate spatial representation
    ax.set_box_aspect(BOX_ASPECT)

    # Set title
    if title:
        ax.set_title(title, fontsize=12, pad=15)

    # Set limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)

    # Configure grid
    if grid:
        ax.grid(True, alpha=grid_alpha)

    return ax


def create_3d_figure(figsize=(10, 8), dpi=100):
    """
    Create figure with 3D axis using standard settings.

    Parameters
    ----------
    figsize : tuple
        Figure size in inches
    dpi : in
        Dots per inch resolution

    Returns
    -------
    tuple
        DEPRECATED: Returns (None, None). Use modern visualization instead.
    """
    # MATPLOTLIB ELIMINATED - Use modern visualization instead
    print(
        "⚠️  create_figure_3d() deprecated. Use dimensional CLI or modern dashboard for 3D visualization."
    )
    print("    Example: python -m dimensional --plot 3d")
    return None, None


def create_figure_3d(figsize=(12, 9), dpi=100):
    """DEPRECATED: Create 3D figure.

    Use modern dashboard visualization instead.
    """
    print(
        "⚠️  create_figure_3d() deprecated. Use dimensional CLI or modern dashboard for 3D visualization."
    )
    print("    Example: python -m dimensional --plot 3d")
    return None, None
