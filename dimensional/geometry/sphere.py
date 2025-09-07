"""
n-dimensional sphere (surface only).
"""

from dataclasses import dataclass

import numpy as np
from scipy import special


@dataclass(frozen=True)
class Sphere:
    """
    An n-dimensional sphere (surface only, not filled).

    The n-sphere is the boundary of the (n+1)-dimensional ball.

    Attributes:
        dimension: The dimension of the sphere (not the embedding space)
        radius: The radius of the sphere (default 1.0 for unit sphere)
    """
    dimension: float
    radius: float = 1.0

    def __post_init__(self):
        """Validate inputs."""
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")

    @property
    def surface_area(self) -> float:
        """
        Calculate the surface area of the n-dimensional sphere.

        Formula: S_n(r) = 2π^((n+1)/2) / Γ((n+1)/2) * r^n

        Note: An n-sphere lives in (n+1)-dimensional space.

        Returns:
            The surface area of the n-sphere
        """
        n = self.dimension + 1  # Embedding dimension
        return 2 * np.pi**(n/2) / special.gamma(n/2) * self.radius**(n-1)

    @property
    def enclosed_volume(self) -> float:
        """
        Calculate the volume enclosed by the n-sphere.

        This is the volume of the (n+1)-dimensional ball bounded by this sphere.

        Formula: V_{n+1}(r) = π^((n+1)/2) / Γ((n+1)/2 + 1) * r^(n+1)

        Returns:
            The volume enclosed by the n-sphere
        """
        n = self.dimension + 1  # Embedding dimension
        return np.pi**(n/2) / special.gamma(n/2 + 1) * self.radius**n

    @property
    def curvature(self) -> float:
        """
        The Gaussian curvature of the sphere.

        For a sphere of radius r, the curvature is 1/r^2.

        Returns:
            The Gaussian curvature
        """
        return 1 / self.radius**2

    @property
    def diameter(self) -> float:
        """The diameter of the sphere."""
        return 2 * self.radius

    @property
    def circumference(self) -> float:
        """
        The circumference of a great circle on the sphere.

        Returns:
            The circumference (2πr for any dimension)
        """
        return 2 * np.pi * self.radius

    def scale(self, factor: float) -> "Sphere":
        """Return a new sphere scaled by the given factor."""
        return Sphere(self.dimension, self.radius * factor)

    def geodesic_distance(self, angle: float) -> float:
        """
        Calculate geodesic distance on the sphere given central angle.

        Args:
            angle: Central angle in radians

        Returns:
            Geodesic distance on the sphere surface
        """
        return self.radius * angle

    def __repr__(self) -> str:
        return f"Sphere(dimension={self.dimension}, radius={self.radius})"

    def __str__(self) -> str:
        return f"{self.dimension}-sphere with radius {self.radius}"
