"""
Octonion Module
===============

Octonions - 8-dimensional alternative algebra for advanced rotations.
"""

import numpy as np


class Octonion:
    """
    Octonion implementation - 8-dimensional non-associative algebra.

    Extends quaternions to 8 dimensions, providing representations
    for exotic transformations in higher dimensions.
    """

    def __init__(self, components: np.ndarray):
        """
        Initialize octonion.

        Parameters
        ----------
        components : np.ndarray
            8-dimensional array of octonion components
        """
        if len(components) != 8:
            raise ValueError("Octonion requires exactly 8 components")
        self.components = np.array(components, dtype=float)

    def norm_squared(self) -> float:
        """Squared norm of octonion."""
        return np.sum(self.components ** 2)

    def norm(self) -> float:
        """Norm of octonion."""
        return np.sqrt(self.norm_squared())

    def conjugate(self) -> 'Octonion':
        """Octonion conjugate."""
        conj_components = self.components.copy()
        conj_components[1:] = -conj_components[1:]  # Negate imaginary parts
        return Octonion(conj_components)


class OctonionAlgebra:
    """
    Octonion algebra O - 8-dimensional alternative algebra.

    Non-associative but alternative division algebra,
    with applications in exceptional Lie groups.
    """

    def __init__(self):
        """Initialize octonion algebra structure."""
        self.dimension = 8
        self.basis_elements = ['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7']
