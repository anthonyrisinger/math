"""
Lie Algebra Module
==================

Lie algebras for continuous symmetry groups and their infinitesimal
generators in dimensional contexts.
"""

from typing import Optional

import numpy as np
from scipy import linalg

from ..core import NUMERICAL_EPSILON, PI


class LieAlgebra:
    """
    Base class for Lie algebra implementations.

    Provides structure for continuous symmetry groups and their
    infinitesimal generators in dimensional contexts.
    """

    def __init__(self, dimension: int, structure_constants: Optional[np.ndarray] = None):
        """
        Initialize Lie algebra.

        Parameters
        ----------
        dimension : int
            Dimension of the Lie algebra
        structure_constants : np.ndarray, optional
            Structure constants defining the algebra
        """
        self.dimension = dimension
        self.structure_constants = structure_constants
        self.generators = None

    def bracket(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Lie bracket [X,Y] = XY - YX for matrix Lie algebras."""
        return X @ Y - Y @ X

    def exp_map(self, X: np.ndarray, t: float = 1.0) -> np.ndarray:
        """Exponential map from algebra to group: exp(tX)."""
        return linalg.expm(t * X)

    def adjoint_representation(self, X: np.ndarray) -> np.ndarray:
        """Adjoint representation ad_X(Y) = [X,Y]."""
        adj_matrix = np.zeros((self.dimension, self.dimension))

        if self.structure_constants is not None:
            # Use structure constants if available
            for i in range(self.dimension):
                for j in range(self.dimension):
                    for k in range(self.dimension):
                        adj_matrix[i, j] += self.structure_constants[i, k, j] * X[k]

        return adj_matrix


class SO3LieAlgebra(LieAlgebra):
    """
    SO(3) Lie algebra - 3D rotation group algebra.

    Generators are skew-symmetric 3×3 matrices representing
    infinitesimal rotations around coordinate axes.
    """

    def __init__(self):
        """Initialize SO(3) Lie algebra."""
        super().__init__(3)
        self._setup_generators()
        self._setup_structure_constants()

    def _setup_generators(self):
        """Setup SO(3) generators (skew-symmetric matrices)."""
        # Standard basis for so(3)
        J1 = np.array([[0, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]], dtype=float)

        J2 = np.array([[0, 0, 1],
                       [0, 0, 0],
                       [-1, 0, 0]], dtype=float)

        J3 = np.array([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype=float)

        self.generators = [J1, J2, J3]

    def _setup_structure_constants(self):
        """Setup structure constants for SO(3): [J_i, J_j] = ε_{ijk} J_k."""
        self.structure_constants = np.zeros((3, 3, 3))

        # Levi-Civita symbol
        epsilon = np.zeros((3, 3, 3))
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1

        self.structure_constants = epsilon

    def rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Generate rotation matrix from axis-angle representation."""
        axis = axis / np.linalg.norm(axis)  # Normalize

        # Generator in direction of axis
        generator = sum(axis[i] * self.generators[i] for i in range(3))

        # Exponential map
        return self.exp_map(generator, angle)

    def logarithm(self, R: np.ndarray) -> tuple[np.ndarray, float]:
        """Extract axis and angle from rotation matrix."""
        # Use Rodrigues' formula in reverse
        trace_R = np.trace(R)
        angle = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))

        if abs(angle) < NUMERICAL_EPSILON:
            return np.array([0, 0, 1]), 0.0

        if abs(angle - PI) < NUMERICAL_EPSILON:
            # Handle 180-degree rotation case
            # Find eigenvector with eigenvalue 1
            eigenvals, eigenvecs = linalg.eig(R)
            idx = np.argmin(np.abs(eigenvals - 1.0))
            axis = np.real(eigenvecs[:, idx])
            return axis / np.linalg.norm(axis), angle

        # Standard case
        skew_part = (R - R.T) / (2 * np.sin(angle))
        axis = np.array([skew_part[2, 1], skew_part[0, 2], skew_part[1, 0]])

        return axis, angle


class SLnLieAlgebra(LieAlgebra):
    """
    SL(n) Lie algebra - special linear group algebra.

    Matrices with trace zero, representing volume-preserving
    linear transformations in n dimensions.
    """

    def __init__(self, n: int):
        """
        Initialize SL(n) Lie algebra.

        Parameters
        ----------
        n : int
            Dimension of the special linear group
        """
        self.n = n
        super().__init__(n*n - 1)  # SL(n) has dimension n²-1
        self._setup_generators()

    def _setup_generators(self):
        """Setup SL(n) generators (traceless matrices)."""
        self.generators = []

        # Elementary matrices E_{ij} - E_{ji} for i ≠ j
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Symmetric generator
                E_sym = np.zeros((self.n, self.n))
                E_sym[i, j] = 1
                E_sym[j, i] = 1
                self.generators.append(E_sym)

                # Antisymmetric generator
                E_antisym = np.zeros((self.n, self.n))
                E_antisym[i, j] = 1
                E_antisym[j, i] = -1
                self.generators.append(E_antisym)

        # Diagonal generators (traceless)
        for k in range(self.n - 1):
            E_diag = np.zeros((self.n, self.n))
            E_diag[k, k] = 1
            E_diag[k + 1, k + 1] = -1
            self.generators.append(E_diag)

    def random_element(self) -> np.ndarray:
        """Generate random element of sl(n)."""
        coeffs = np.random.normal(0, 1, len(self.generators))
        return sum(c * gen for c, gen in zip(coeffs, self.generators))
