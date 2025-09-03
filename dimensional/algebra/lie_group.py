"""
Lie Group Module
================

Lie groups for continuous transformations with smooth manifold structure,
essential for dimensional transformation analysis.
"""

from typing import Optional

import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation

from ..mathematics import NUMERICAL_EPSILON
from .lie_algebra import LieAlgebra, SLnLieAlgebra, SO3LieAlgebra
from .quaternion import Quaternion


class LieGroup:
    """
    Base class for Lie group implementations.

    Provides structure for continuous groups with smooth manifold structure,
    essential for dimensional transformation analysis.
    """

    def __init__(self, dimension: int, algebra: Optional[LieAlgebra] = None):
        """
        Initialize Lie group.

        Parameters
        ----------
        dimension : int
            Dimension of the group
        algebra : LieAlgebra, optional
            Associated Lie algebra
        """
        self.dimension = dimension
        self.algebra = algebra
        self.identity = self._identity_element()

    def _identity_element(self) -> np.ndarray:
        """Return identity element of the group."""
        return np.eye(self.dimension)

    def multiply(self, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
        """Group multiplication (matrix multiplication for matrix groups)."""
        return g1 @ g2

    def inverse(self, g: np.ndarray) -> np.ndarray:
        """Group inverse (matrix inverse for matrix groups)."""
        try:
            return linalg.inv(g)
        except linalg.LinAlgError:
            return linalg.pinv(g)

    def conjugate(self, g: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Conjugation: g h g^(-1)."""
        return self.multiply(self.multiply(g, h), self.inverse(g))

    def commutator(self, g: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Group commutator: g h g^(-1) h^(-1)."""
        return self.multiply(
            self.conjugate(g, h),
            self.inverse(h)
        )

    def exponential_map(self, X: np.ndarray) -> np.ndarray:
        """Exponential map from Lie algebra to group."""
        if self.algebra is not None:
            return self.algebra.exp_map(X)
        return linalg.expm(X)

    def logarithm_map(self, g: np.ndarray) -> np.ndarray:
        """Logarithm map from group to Lie algebra (approximate)."""
        return linalg.logm(g)

    def one_parameter_subgroup(self, X: np.ndarray, t_values: np.ndarray) -> np.ndarray:
        """Generate one-parameter subgroup exp(tX)."""
        subgroup_elements = []
        for t in t_values:
            subgroup_elements.append(self.exponential_map(t * X))
        return np.array(subgroup_elements)

    def left_translate(self, g: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Left translation: L_g(h) = gh."""
        return self.multiply(g, h)

    def right_translate(self, g: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Right translation: R_g(h) = hg."""
        return self.multiply(h, g)

    def adjoint_action(self, g: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Adjoint action: Ad_g(X) = g X g^(-1)."""
        return self.multiply(self.multiply(g, X), self.inverse(g))


class SO3Group(LieGroup):
    """
    SO(3) group - 3D rotation group.

    Special orthogonal group of 3×3 rotation matrices.
    """

    def __init__(self):
        """Initialize SO(3) group."""
        algebra = SO3LieAlgebra()
        super().__init__(3, algebra)

    def random_element(self) -> np.ndarray:
        """Generate random rotation matrix."""
        # Use Haar measure on SO(3)
        quaternion = Quaternion(
            np.random.normal(),
            np.random.normal(),
            np.random.normal(),
            np.random.normal()
        ).normalize()
        return quaternion.to_rotation_matrix()

    def geodesic(self, g1: np.ndarray, g2: np.ndarray, t: float) -> np.ndarray:
        """Geodesic interpolation between rotations."""
        # Convert to quaternions for spherical linear interpolation
        # Note: Direct matrix interpolation used below

        # Create interpolation points
        interp_times = np.array([0, 1])
        interp_rotations = Rotation.from_matrix(np.array([g1, g2]))

        # Use Slerp for interpolation
        slerp = Rotation.Slerp(interp_times, interp_rotations)
        interpolated = slerp(t)

        return interpolated.as_matrix()

    def distance(self, g1: np.ndarray, g2: np.ndarray) -> float:
        """Riemannian distance on SO(3)."""
        relative_rotation = self.multiply(self.inverse(g1), g2)
        axis, angle = self.algebra.logarithm(relative_rotation)
        return abs(angle)


class SLnGroup(LieGroup):
    """
    SL(n) group - special linear group.

    Group of n×n matrices with determinant 1.
    """

    def __init__(self, n: int):
        """
        Initialize SL(n) group.

        Parameters
        ----------
        n : int
            Dimension of matrices
        """
        self.n = n
        algebra = SLnLieAlgebra(n)
        super().__init__(n, algebra)

    def _identity_element(self) -> np.ndarray:
        """Return identity element."""
        return np.eye(self.n)

    def is_valid_element(self, g: np.ndarray) -> bool:
        """Check if matrix is in SL(n)."""
        det = np.linalg.det(g)
        return abs(det - 1.0) < NUMERICAL_EPSILON

    def random_element(self) -> np.ndarray:
        """Generate random SL(n) element."""
        # Generate random element in Lie algebra and exponentiate
        X = self.algebra.random_element()
        return self.exponential_map(X)

    def polar_decomposition(self, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Polar decomposition: g = k p where k ∈ SO(n), p positive definite."""
        U, s, Vt = linalg.svd(g)

        # Ensure det(U) = det(V) = 1 for proper rotations
        if np.linalg.det(U) < 0:
            U[:, -1] *= -1
            s[-1] *= -1
        if np.linalg.det(Vt) < 0:
            Vt[-1, :] *= -1
            s[-1] *= -1

        k = U @ Vt  # Rotation part
        p = Vt.T @ np.diag(s) @ Vt  # Positive definite part

        return k, p
