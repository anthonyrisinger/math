"""
Core.core compatibility module for tests.
"""

import numpy as np

from ..core import PI, r, s, v


def total_phase_energy(phase_density_or_d, beta=1.0):
    """Compute total phase energy.

    Can accept either:
    - phase_density array (for PhaseDynamicsEngine compatibility)
    - dimension value d (for backward compatibility)
    """
    input_val = np.asarray(phase_density_or_d, dtype=np.float64)

    # Check if it's a phase density array (sum should be ~1)
    if input_val.ndim > 0 and len(input_val) > 1:
        # It's a phase density array - return sum (conserved quantity)
        return np.sum(input_val)
    else:
        # It's a dimension value - use original formula
        d = input_val
        beta = np.asarray(beta, dtype=np.float64)
        energy = beta * (v(d) + s(d))
        return float(energy) if np.isscalar(energy) else energy


def phase_transition_temperature(d):
    """Compute phase transition temperature."""
    d = np.asarray(d, dtype=np.float64)

    # Temperature related to ratio measure
    temp = r(d) * np.sqrt(2 * PI)

    return float(temp) if np.isscalar(temp) else temp


def critical_exponent(d):
    """Compute critical exponent."""
    d = np.asarray(d, dtype=np.float64)

    # Critical exponent from dimensional analysis
    exponent = 2.0 / (d + 1)

    return float(exponent) if np.isscalar(exponent) else exponent


def order_parameter(d, t):
    """Compute order parameter."""
    d = np.asarray(d, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # Order parameter
    param = v(d) * np.exp(-t / phase_transition_temperature(d))

    return float(param) if np.isscalar(param) else param


def correlation_length(d, t):
    """Compute correlation length."""
    d = np.asarray(d, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # Correlation length diverges at critical point
    t_c = phase_transition_temperature(d)
    xi = 1.0 / np.abs(t - t_c + 1e-10)

    return float(xi) if np.isscalar(xi) else xi


def susceptibility(d, t):
    """Compute susceptibility."""
    d = np.asarray(d, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # Susceptibility
    chi = correlation_length(d, t) ** (d - 2)

    return float(chi) if np.isscalar(chi) else chi


def free_energy(d, t):
    """Compute free energy."""
    d = np.asarray(d, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # Free energy
    f = -t * np.log(v(d) + s(d) + 1e-10)

    return float(f) if np.isscalar(f) else f


def entropy(d, t):
    """Compute entropy."""
    d = np.asarray(d, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # Entropy as derivative of free energy
    h = 1e-8
    f_plus = free_energy(d, t + h)
    f_minus = free_energy(d, t - h)
    s_entropy = -(f_plus - f_minus) / (2 * h)

    return float(s_entropy) if np.isscalar(s_entropy) else s_entropy


def specific_heat(d, t):
    """Compute specific heat."""
    d = np.asarray(d, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # Specific heat
    c_v = t * entropy(d, t) / t if t != 0 else 0

    return float(c_v) if np.isscalar(c_v) else c_v


def sap_rate(source, target, phase_density=None):
    """Calculate SAP (Surface Area to Power) rate between dimensions.

    The SAP rate measures energy transfer from source to target dimension.
    Only allows transfer from lower to higher dimensions.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # No sapping to lower dimensions or same dimension
    if source >= target:
        return 0.0

    # Basic rate proportional to dimension difference
    base_rate = (target - source) * 0.1

    # If phase density provided, modulate by density
    if phase_density is not None:
        phase_density = np.asarray(phase_density)
        # Higher density = lower rate (saturation effect)
        density_factor = np.abs(np.mean(phase_density))
        # Inverse relationship: as density increases, rate decreases
        base_rate *= np.exp(-density_factor * 2)  # Exponential decay with density

    return float(base_rate)


# Export all
__all__ = [
    'total_phase_energy', 'phase_transition_temperature',
    'critical_exponent', 'order_parameter', 'correlation_length',
    'susceptibility', 'free_energy', 'entropy', 'specific_heat',
    'sap_rate',
]
