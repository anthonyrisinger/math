"""
n-dimensional ball (solid sphere).
"""

from dataclasses import dataclass

import numpy as np
from scipy import special


@dataclass(frozen=True)
class Ball:
    """
    An n-dimensional ball (filled sphere).

    Attributes:
        dimension: The dimension of the space
        radius: The radius of the ball (default 1.0 for unit ball)
    """
    dimension: float
    radius: float = 1.0

    def __post_init__(self):
        """Validate inputs."""
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")

    @property
    def volume(self) -> float:
        """
        Calculate the volume of the n-dimensional ball.

        Formula: V_n(r) = π^(n/2) / Γ(n/2 + 1) * r^n

        Returns:
            The volume of the n-dimensional ball
        """
        n = self.dimension
        return np.pi**(n/2) / special.gamma(n/2 + 1) * self.radius**n

    @property
    def surface_area(self) -> float:
        """
        Calculate the surface area of the n-dimensional ball.

        Formula: S_n(r) = 2π^(n/2) / Γ(n/2) * r^(n-1)

        Returns:
            The surface area of the boundary (n-1 sphere)
        """
        n = self.dimension
        return 2 * np.pi**(n/2) / special.gamma(n/2) * self.radius**(n-1)

    @property
    def unit_volume(self) -> float:
        """Volume of the unit ball (radius=1)."""
        n = self.dimension
        return np.pi**(n/2) / special.gamma(n/2 + 1)

    @property
    def unit_surface_area(self) -> float:
        """Surface area of the unit ball (radius=1)."""
        n = self.dimension
        return 2 * np.pi**(n/2) / special.gamma(n/2)

    def scale(self, factor: float) -> "Ball":
        """Return a new ball scaled by the given factor."""
        return Ball(self.dimension, self.radius * factor)

    def __repr__(self) -> str:
        return f"Ball(dimension={self.dimension}, radius={self.radius})"

    def __str__(self) -> str:
        return f"{self.dimension}D ball with radius {self.radius}"
