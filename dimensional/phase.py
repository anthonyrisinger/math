"""
Phase analysis compatibility module for tests.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from .core import c, r, s, v


def phase_transition(d, critical_point=5.25695):
    """Analyze phase transition at critical dimension."""
    d = np.asarray(d, dtype=np.float64)

    # Volume and surface at dimension
    volume = v(d)
    surface = s(d)

    # Phase indicator: below or above critical point
    phase = np.where(d < critical_point, "subcritical", "supercritical")

    # Distance from critical point
    distance = d - critical_point

    return {
        "dimension": float(d) if np.isscalar(d) else d,
        "volume": float(volume) if np.isscalar(volume) else volume,
        "surface": float(surface) if np.isscalar(surface) else surface,
        "phase": phase if not np.isscalar(d) else str(phase),
        "distance_from_critical": float(distance) if np.isscalar(distance) else distance,
        "critical_point": critical_point,
    }


def critical_dimension(measure="volume"):
    """Find critical dimension where measure is maximized."""
    measure_funcs = {
        "volume": v,
        "surface": s,
        "complexity": c,
        "ratio": r,
    }

    func = measure_funcs.get(measure, v)

    # Find maximum
    result = minimize_scalar(lambda d: -func(d), bounds=(0.1, 20), method='bounded')

    return {
        "measure": measure,
        "critical_dimension": result.x,
        "critical_value": -result.fun,
        "success": result.success,
    }


def phase_diagram(d_range=None, n_points=100):
    """Generate phase diagram data."""
    if d_range is None:
        d_range = np.linspace(0.1, 20, n_points)
    else:
        d_range = np.asarray(d_range)

    volumes = v(d_range)
    surfaces = s(d_range)
    complexities = c(d_range)
    ratios = r(d_range)

    # Find peaks
    v_peak_idx = np.argmax(volumes)
    s_peak_idx = np.argmax(surfaces)
    c_peak_idx = np.argmax(complexities)

    return {
        "dimensions": d_range.tolist(),
        "volume": volumes.tolist(),
        "surface": surfaces.tolist(),
        "complexity": complexities.tolist(),
        "ratio": ratios.tolist(),
        "volume_peak": (float(d_range[v_peak_idx]), float(volumes[v_peak_idx])),
        "surface_peak": (float(d_range[s_peak_idx]), float(surfaces[s_peak_idx])),
        "complexity_peak": (float(d_range[c_peak_idx]), float(complexities[c_peak_idx])),
    }


def phase_velocity(d, h=1e-8):
    """Compute phase velocity (rate of change)."""
    d = np.asarray(d, dtype=np.float64)

    # Numerical derivatives
    v_plus = v(d + h)
    v_minus = v(d - h)
    dv_dd = (v_plus - v_minus) / (2 * h)

    s_plus = s(d + h)
    s_minus = s(d - h)
    ds_dd = (s_plus - s_minus) / (2 * h)

    # Phase velocity as magnitude of gradient
    velocity = np.sqrt(dv_dd**2 + ds_dd**2)

    return {
        "dimension": float(d) if np.isscalar(d) else d,
        "volume_velocity": float(dv_dd) if np.isscalar(dv_dd) else dv_dd,
        "surface_velocity": float(ds_dd) if np.isscalar(ds_dd) else ds_dd,
        "total_velocity": float(velocity) if np.isscalar(velocity) else velocity,
    }


def phase_portrait(d_center=5.0, width=2.0, n_points=50):
    """Generate phase portrait around a dimension."""
    d_range = np.linspace(d_center - width, d_center + width, n_points)

    volumes = v(d_range)
    surfaces = s(d_range)

    # Compute derivatives for vector field
    dv = np.gradient(volumes, d_range)
    ds = np.gradient(surfaces, d_range)

    return {
        "center": d_center,
        "width": width,
        "dimensions": d_range.tolist(),
        "volume": volumes.tolist(),
        "surface": surfaces.tolist(),
        "volume_gradient": dv.tolist(),
        "surface_gradient": ds.tolist(),
        "n_points": n_points,
    }


def phase_stability(d, epsilon=1e-6):
    """Analyze stability at dimension d."""
    d = np.asarray(d, dtype=np.float64)

    # Compute second derivatives (curvature)
    v_0 = v(d)
    v_plus = v(d + epsilon)
    v_minus = v(d - epsilon)
    d2v_dd2 = (v_plus - 2*v_0 + v_minus) / (epsilon**2)

    s_0 = s(d)
    s_plus = s(d + epsilon)
    s_minus = s(d - epsilon)
    d2s_dd2 = (s_plus - 2*s_0 + s_minus) / (epsilon**2)

    # Stability based on curvature
    v_stable = d2v_dd2 < 0  # Negative curvature = local maximum = stable
    s_stable = d2s_dd2 < 0

    return {
        "dimension": float(d) if np.isscalar(d) else d,
        "volume_curvature": float(d2v_dd2) if np.isscalar(d2v_dd2) else d2v_dd2,
        "surface_curvature": float(d2s_dd2) if np.isscalar(d2s_dd2) else d2s_dd2,
        "volume_stable": bool(v_stable) if np.isscalar(v_stable) else v_stable,
        "surface_stable": bool(s_stable) if np.isscalar(s_stable) else s_stable,
    }


def phase_bifurcation(alpha_range=None, d_fixed=5.0):
    """Analyze bifurcation with parameter alpha."""
    if alpha_range is None:
        alpha_range = np.linspace(0.5, 2.0, 50)
    else:
        alpha_range = np.asarray(alpha_range)

    # Bifurcation: how measures change with scaling parameter
    scaled_dims = d_fixed * alpha_range
    volumes = v(scaled_dims)
    surfaces = s(scaled_dims)

    return {
        "alpha": alpha_range.tolist(),
        "fixed_dimension": d_fixed,
        "scaled_dimensions": scaled_dims.tolist(),
        "volume": volumes.tolist(),
        "surface": surfaces.tolist(),
    }


def phase_entropy(d, base=2):
    """Compute phase entropy."""
    d = np.asarray(d, dtype=np.float64)

    # Entropy based on measure distributions
    volume = v(d)
    surface = s(d)
    total = volume + surface + 1e-10  # Avoid division by zero

    # Probabilities
    p_v = volume / total
    p_s = surface / total

    # Shannon entropy
    entropy = -(p_v * np.log(p_v + 1e-10) + p_s * np.log(p_s + 1e-10)) / np.log(base)

    return {
        "dimension": float(d) if np.isscalar(d) else d,
        "entropy": float(entropy) if np.isscalar(entropy) else entropy,
        "volume_probability": float(p_v) if np.isscalar(p_v) else p_v,
        "surface_probability": float(p_s) if np.isscalar(p_s) else p_s,
        "base": base,
    }


def phase_correlation(d1, d2):
    """Compute phase correlation between two dimensions."""
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)

    # Measures at both dimensions
    v1, s1 = v(d1), s(d1)
    v2, s2 = v(d2), s(d2)

    # Correlation coefficient (simplified)
    correlation = (v1 * v2 + s1 * s2) / np.sqrt((v1**2 + s1**2) * (v2**2 + s2**2) + 1e-10)

    return {
        "d1": float(d1) if np.isscalar(d1) else d1,
        "d2": float(d2) if np.isscalar(d2) else d2,
        "correlation": float(correlation) if np.isscalar(correlation) else correlation,
        "v1": float(v1) if np.isscalar(v1) else v1,
        "v2": float(v2) if np.isscalar(v2) else v2,
        "s1": float(s1) if np.isscalar(s1) else s1,
        "s2": float(s2) if np.isscalar(s2) else s2,
    }


def phase_spectrum(d, n_modes=10):
    """Compute phase spectrum (frequency decomposition)."""
    d = np.asarray(d, dtype=np.float64)

    modes = []
    frequencies = []

    for n in range(1, n_modes + 1):
        # Mode frequency
        freq = n * np.pi / 10
        frequencies.append(freq)

        # Mode amplitude (combination of measures)
        amplitude = v(d) * np.cos(freq * d) + s(d) * np.sin(freq * d)
        modes.append(float(amplitude) if np.isscalar(amplitude) else amplitude)

    return {
        "dimension": float(d) if np.isscalar(d) else d,
        "frequencies": frequencies,
        "modes": modes,
        "n_modes": n_modes,
        "total_power": float(np.sum(np.abs(modes)**2)),
    }


# Aliases for compatibility
transition = phase_transition
critical = critical_dimension
diagram = phase_diagram
velocity = phase_velocity
portrait = phase_portrait
stability = phase_stability
bifurcation = phase_bifurcation
entropy = phase_entropy
correlation = phase_correlation
spectrum = phase_spectrum

# Export all
__all__ = [
    'phase_transition', 'critical_dimension', 'phase_diagram',
    'phase_velocity', 'phase_portrait', 'phase_stability',
    'phase_bifurcation', 'phase_entropy', 'phase_correlation',
    'phase_spectrum', 'transition', 'critical', 'diagram',
    'velocity', 'portrait', 'stability', 'bifurcation',
    'entropy', 'correlation', 'spectrum',
]
