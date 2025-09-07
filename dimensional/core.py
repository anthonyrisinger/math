"""
Minimal dimensional mathematics core.
Total implementation in under 200 lines.
"""

import numpy as np
from scipy.special import digamma as scipy_digamma
from scipy.special import gamma as scipy_gamma
from scipy.special import gammaln

# Mathematical constants
PI = np.pi
E = np.e
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio


def gamma(z):
    """Gamma function Γ(z)."""
    z = np.asarray(z, dtype=np.float64)
    result = scipy_gamma(z)
    return float(result) if np.isscalar(z) else result


def digamma(z):
    """Digamma function ψ(z) = d/dz log(Γ(z))."""
    z = np.asarray(z, dtype=np.float64)
    result = scipy_digamma(z)
    return float(result) if np.isscalar(z) else result


def v(d):
    """Volume of unit d-dimensional ball: V_d = π^(d/2) / Γ(d/2 + 1)."""
    d = np.asarray(d, dtype=np.float64)
    scalar = np.isscalar(d)

    # V_d = π^(d/2) / Γ(d/2 + 1)
    log_vol = (d/2) * np.log(PI) - gammaln(d/2 + 1)
    result = np.exp(log_vol)

    # Handle d=0 special case
    result = np.where(np.abs(d) < 1e-15, 1.0, result)

    return float(result) if scalar else result


def s(d):
    """Surface area of unit (d-1)-sphere: S_d = 2π^(d/2) / Γ(d/2)."""
    d = np.asarray(d, dtype=np.float64)
    scalar = np.isscalar(d)

    # S_d = 2π^(d/2) / Γ(d/2)
    log_surf = np.log(2) + (d/2) * np.log(PI) - gammaln(d/2)
    result = np.exp(log_surf)

    # Handle special cases
    result = np.where(np.abs(d) < 1e-15, 2.0, result)
    result = np.where(np.abs(d - 1) < 1e-15, 2.0, result)

    return float(result) if scalar else result


def c(d):
    """Complexity measure: C(d) = V(d) × S(d)."""
    d = np.asarray(d, dtype=np.float64)
    scalar = np.isscalar(d)

    # C(d) = 2π^d / (Γ(d/2) × Γ(d/2 + 1))
    log_complexity = np.log(2) + d * np.log(PI) - gammaln(d/2) - gammaln(d/2 + 1)
    result = np.exp(log_complexity)

    # Handle edge case
    result = np.where(np.abs(d) < 1e-15, 2.0, result)

    return float(result) if scalar else result


def r(d):
    """Ratio measure: R(d) = S(d) / V(d) = d for unit ball."""
    d = np.asarray(d, dtype=np.float64)
    scalar = np.isscalar(d)

    # For unit ball: R(d) = d
    result = np.where(np.abs(d) < 1e-15, 2.0, d)

    return float(result) if scalar else result


# Dataclass for Ball
class Ball:
    """N-dimensional ball (solid sphere)."""

    def __init__(self, dimension, radius=1.0):
        self.dimension = dimension
        self.radius = radius

    @property
    def volume(self):
        """Volume of the ball."""
        return self.radius**self.dimension * v(self.dimension)

    @property
    def surface_area(self):
        """Surface area of the ball's boundary."""
        return self.radius**(self.dimension-1) * s(self.dimension)

    def __repr__(self):
        return f"Ball(dimension={self.dimension}, radius={self.radius})"


# Dataclass for Sphere
class Sphere:
    """(N-1)-dimensional sphere (surface)."""

    def __init__(self, dimension, radius=1.0):
        self.dimension = dimension
        self.radius = radius

    @property
    def surface_area(self):
        """Surface area of the sphere."""
        return self.radius**(self.dimension-1) * s(self.dimension)

    @property
    def enclosed_volume(self):
        """Volume enclosed by the sphere."""
        return self.radius**self.dimension * v(self.dimension)

    def __repr__(self):
        return f"Sphere(dimension={self.dimension}, radius={self.radius})"


# Aliases for backward compatibility
ball_volume = v
sphere_surface = s
complexity_measure = c
ratio_measure = r

# Export all
__all__ = [
    # Functions
    'v', 's', 'c', 'r',
    'gamma', 'digamma',
    # Classes
    'Ball', 'Sphere',
    # Constants
    'PI', 'E', 'PHI',
    # Aliases
    'ball_volume', 'sphere_surface',
    'complexity_measure', 'ratio_measure'
]
