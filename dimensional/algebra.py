"""
Advanced Algebraic Structures Module
===================================

Algebraic structures for dimensional mathematics, moving beyond basic measures
into sophisticated mathematical frameworks for rotations, symmetries, and transformations.

Provides:
- Clifford Algebras for multidimensional rotations and reflections
- Lie Groups/Algebras for continuous symmetry analysis
- Quaternion/Octonion extensions for higher-dimensional rotations
- Group actions on dimensional spaces
- Representation theory for dimensional transformations
- Homology and cohomology in dimensional contexts
"""

import warnings
from itertools import combinations
from typing import Any, Optional, Union

import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation

# Import from consolidated mathematics
from .mathematics import (
    NUMERICAL_EPSILON,
    PHI,
    PI,
    DimensionalError,
)

# =============================================================================
# CLIFFORD ALGEBRAS
# =============================================================================

class CliffordAlgebra:
    """
    Clifford algebra implementation for multidimensional geometric operations.

    Clifford algebras Cl(p,q) generalize complex numbers and quaternions,
    providing natural framework for rotations and reflections in arbitrary dimensions.
    """

    def __init__(self, p: int, q: int = 0, r: int = 0):
        """
        Initialize Clifford algebra Cl(p,q,r).

        Args:
            p: Number of basis vectors with e_i² = +1
            q: Number of basis vectors with e_i² = -1
            r: Number of basis vectors with e_i² = 0 (degenerate)
        """
        self.p = p
        self.q = q
        self.r = r
        self.n = p + q + r  # Total dimension
        self.algebra_dim = 2**self.n  # Dimension of Clifford algebra

        # Generate basis elements
        self._generate_basis()
        self._compute_multiplication_table()

    def _generate_basis(self):
        """Generate basis elements and their properties."""
        self.basis_elements = []
        self.basis_grades = []  # Grade (number of generators)
        self.basis_signatures = []  # Signature under conjugation

        # Generate all subsets of {0, 1, ..., n-1}
        for grade in range(self.n + 1):
            for subset in combinations(range(self.n), grade):
                self.basis_elements.append(subset)
                self.basis_grades.append(grade)

                # Compute signature
                signature = 1
                for i in subset:
                    if i < self.p:
                        signature *= 1  # e_i² = +1
                    elif i < self.p + self.q:
                        signature *= -1  # e_i² = -1
                    else:
                        signature *= 0  # e_i² = 0 (degenerate)

                self.basis_signatures.append(signature)

    def _compute_multiplication_table(self):
        """Compute multiplication table for basis elements."""
        n_basis = len(self.basis_elements)
        self.mult_table = np.zeros((n_basis, n_basis), dtype=complex)
        self.mult_signs = np.zeros((n_basis, n_basis), dtype=int)
        self.mult_indices = np.zeros((n_basis, n_basis), dtype=int)

        for i, basis_i in enumerate(self.basis_elements):
            for j, basis_j in enumerate(self.basis_elements):
                # Multiply basis elements
                result_basis, sign = self._multiply_basis_elements(basis_i, basis_j)

                # Find index of result
                try:
                    result_idx = self.basis_elements.index(result_basis)
                    self.mult_table[i, j] = sign
                    self.mult_signs[i, j] = sign
                    self.mult_indices[i, j] = result_idx
                except ValueError:
                    # Result is zero (degenerate case)
                    self.mult_table[i, j] = 0
                    self.mult_signs[i, j] = 0
                    self.mult_indices[i, j] = 0

    def _multiply_basis_elements(self, basis_a: tuple, basis_b: tuple) -> tuple[tuple, int]:
        """Multiply two basis elements and return result with sign."""
        # Combine and sort indices
        combined = list(basis_a) + list(basis_b)

        # Remove pairs (e_i * e_i = ±1 or 0)
        result_indices = []
        sign = 1

        i = 0
        while i < len(combined):
            current = combined[i]
            count = combined.count(current)

            if count % 2 == 1:
                result_indices.append(current)

            # Handle signature
            if count >= 2:
                pairs = count // 2
                if current < self.p:
                    # e_i² = +1, no sign change
                    pass
                elif current < self.p + self.q:
                    # e_i² = -1
                    sign *= (-1)**pairs
                else:
                    # e_i² = 0, result is zero
                    return (), 0

            # Remove all instances of current index
            combined = [x for x in combined if x != current]

        # Sort result and compute additional sign from permutation
        result_indices.sort()

        # Compute permutation sign
        original_order = list(basis_a) + list(basis_b)
        perm_sign = self._permutation_sign(original_order, result_indices)

        return tuple(result_indices), sign * perm_sign

    def _permutation_sign(self, original: list, final: list) -> int:
        """
        Compute sign of permutation needed to sort original to final order.
        
        Uses inversion counting algorithm to determine parity.
        """
        if len(original) != len(final) or set(original) != set(final):
            raise ValueError("Lists must contain same elements")
        
        # Create mapping from element to position in final order
        final_positions = {elem: pos for pos, elem in enumerate(final)}
        
        # Map original to position indices in final order
        position_sequence = [final_positions[elem] for elem in original]
        
        # Count inversions in position sequence
        inversions = 0
        n = len(position_sequence)
        
        for i in range(n):
            for j in range(i + 1, n):
                if position_sequence[i] > position_sequence[j]:
                    inversions += 1
        
        # Return -1 for odd number of inversions, +1 for even
        return -1 if inversions % 2 == 1 else 1

    def multivector(self, coefficients: np.ndarray) -> 'CliffordMultivector':
        """Create multivector from coefficients."""
        if len(coefficients) != self.algebra_dim:
            raise ValueError(f"Expected {self.algebra_dim} coefficients, got {len(coefficients)}")

        return CliffordMultivector(self, coefficients)

    def scalar(self, value: float) -> 'CliffordMultivector':
        """Create scalar multivector."""
        coeffs = np.zeros(self.algebra_dim)
        coeffs[0] = value  # Scalar part is first basis element (empty set)
        return CliffordMultivector(self, coeffs)

    def vector(self, components: np.ndarray) -> 'CliffordMultivector':
        """Create vector multivector from n-dimensional components."""
        if len(components) != self.n:
            raise ValueError(f"Expected {self.n} components, got {len(components)}")

        coeffs = np.zeros(self.algebra_dim)
        coeffs[0] = 0  # No scalar part

        # Find basis elements of grade 1 (single generators)
        for i, basis_elem in enumerate(self.basis_elements):
            if len(basis_elem) == 1:
                generator_idx = basis_elem[0]
                coeffs[i] = components[generator_idx]

        return CliffordMultivector(self, coeffs)


class CliffordMultivector:
    """Multivector in Clifford algebra with arithmetic operations."""

    def __init__(self, algebra: CliffordAlgebra, coefficients: np.ndarray):
        self.algebra = algebra
        self.coefficients = coefficients.copy()

    def __add__(self, other: 'CliffordMultivector') -> 'CliffordMultivector':
        """Add two multivectors."""
        if self.algebra != other.algebra:
            raise ValueError("Cannot add multivectors from different algebras")

        return CliffordMultivector(self.algebra, self.coefficients + other.coefficients)

    def __sub__(self, other: 'CliffordMultivector') -> 'CliffordMultivector':
        """Subtract two multivectors."""
        if self.algebra != other.algebra:
            raise ValueError("Cannot subtract multivectors from different algebras")

        return CliffordMultivector(self.algebra, self.coefficients - other.coefficients)

    def __mul__(self, other: Union['CliffordMultivector', float]) -> 'CliffordMultivector':
        """Multiply multivectors or scale by scalar."""
        if isinstance(other, (int, float, complex)):
            return CliffordMultivector(self.algebra, self.coefficients * other)

        if not isinstance(other, CliffordMultivector):
            return NotImplemented

        if self.algebra != other.algebra:
            raise ValueError("Cannot multiply multivectors from different algebras")

        # Geometric product using multiplication table
        result_coeffs = np.zeros(self.algebra.algebra_dim, dtype=complex)

        for i, coeff_a in enumerate(self.coefficients):
            for j, coeff_b in enumerate(other.coefficients):
                if coeff_a != 0 and coeff_b != 0:
                    result_idx = self.algebra.mult_indices[i, j]
                    sign = self.algebra.mult_signs[i, j]
                    result_coeffs[result_idx] += coeff_a * coeff_b * sign

        return CliffordMultivector(self.algebra, result_coeffs)

    def __rmul__(self, scalar: float) -> 'CliffordMultivector':
        """Right multiply by scalar."""
        return CliffordMultivector(self.algebra, scalar * self.coefficients)

    def reverse(self) -> 'CliffordMultivector':
        """Compute reverse (main involution)."""
        # Reverse changes sign of basis elements based on grade
        result_coeffs = self.coefficients.copy()

        for i, grade in enumerate(self.algebra.basis_grades):
            if grade % 4 >= 2:  # Grades 2,3 (mod 4) change sign
                result_coeffs[i] *= -1

        return CliffordMultivector(self.algebra, result_coeffs)

    def conjugate(self) -> 'CliffordMultivector':
        """Compute Clifford conjugate."""
        # Conjugate changes sign of all vector parts
        result_coeffs = self.coefficients.copy()

        for i, grade in enumerate(self.algebra.basis_grades):
            if grade % 2 == 1:  # Odd grades change sign
                result_coeffs[i] *= -1

        return CliffordMultivector(self.algebra, result_coeffs)

    def norm_squared(self) -> float:
        """Compute squared norm |M|²."""
        conjugated = self.conjugate()
        product = self * conjugated
        return np.real(product.coefficients[0])  # Scalar part

    def norm(self) -> float:
        """Compute norm |M|."""
        norm_sq = self.norm_squared()
        return np.sqrt(abs(norm_sq)) if norm_sq >= 0 else np.sqrt(-norm_sq) * 1j

    def inverse(self) -> 'CliffordMultivector':
        """Compute multiplicative inverse."""
        norm_sq = self.norm_squared()
        if abs(norm_sq) < NUMERICAL_EPSILON:
            raise ValueError("Cannot invert zero multivector")

        conjugated = self.conjugate()
        return (1.0 / norm_sq) * conjugated

    def grade_part(self, grade: int) -> 'CliffordMultivector':
        """Extract part of specific grade."""
        result_coeffs = np.zeros_like(self.coefficients)

        for i, g in enumerate(self.algebra.basis_grades):
            if g == grade:
                result_coeffs[i] = self.coefficients[i]

        return CliffordMultivector(self.algebra, result_coeffs)

    def scalar_part(self) -> float:
        """Extract scalar part (grade 0)."""
        return np.real(self.coefficients[0])

    def vector_part(self) -> np.ndarray:
        """Extract vector part (grade 1) as n-dimensional array."""
        vector_coeffs = np.zeros(self.algebra.n)

        for i, basis_elem in enumerate(self.algebra.basis_elements):
            if len(basis_elem) == 1:  # Grade 1
                generator_idx = basis_elem[0]
                vector_coeffs[generator_idx] = np.real(self.coefficients[i])

        return vector_coeffs


# =============================================================================
# LIE GROUPS AND ALGEBRAS
# =============================================================================

class LieAlgebra:
    """
    Base class for Lie algebra implementations.

    Provides structure for continuous symmetry groups and their
    infinitesimal generators in dimensional contexts.
    """

    def __init__(self, dimension: int, structure_constants: Optional[np.ndarray] = None):
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

        J3 = np.array([[-0, -1, 0],
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


# =============================================================================
# QUATERNION AND OCTONION EXTENSIONS
# =============================================================================

class Quaternion:
    """
    Quaternion implementation for 3D/4D rotations.

    Extends complex numbers to provide efficient representation
    of rotations in 3D space without gimbal lock.
    """

    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w  # Real part
        self.x = x  # i component
        self.y = y  # j component
        self.z = z  # k component

    @property
    def components(self) -> tuple:
        """Get quaternion components as (w, x, y, z) tuple."""
        return (self.w, self.x, self.y, self.z)

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> 'Quaternion':
        """Create quaternion from axis-angle representation."""
        axis = axis / np.linalg.norm(axis)  # Normalize
        half_angle = angle / 2
        sin_half = np.sin(half_angle)

        return cls(
            np.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        )

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> 'Quaternion':
        """Create quaternion from Euler angles (ZYX convention)."""
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        return cls(
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr
        )

    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        """Add quaternions."""
        return Quaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __mul__(self, other: Union['Quaternion', float]) -> 'Quaternion':
        """Multiply quaternions or scale by scalar."""
        if isinstance(other, (int, float)):
            return Quaternion(
                self.w * other,
                self.x * other,
                self.y * other,
                self.z * other
            )

        if not isinstance(other, Quaternion):
            return NotImplemented

        # Quaternion multiplication
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        )

    def __rmul__(self, scalar: float) -> 'Quaternion':
        """Right multiply by scalar."""
        return self * scalar

    def conjugate(self) -> 'Quaternion':
        """Quaternion conjugate."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm_squared(self) -> float:
        """Squared norm of quaternion."""
        return self.w**2 + self.x**2 + self.y**2 + self.z**2

    def norm(self) -> float:
        """Norm of quaternion."""
        return np.sqrt(self.norm_squared())

    def normalize(self) -> 'Quaternion':
        """Return normalized unit quaternion."""
        n = self.norm()
        if n < NUMERICAL_EPSILON:
            raise ValueError("Cannot normalize zero quaternion")
        return self * (1.0 / n)

    def inverse(self) -> 'Quaternion':
        """Multiplicative inverse."""
        norm_sq = self.norm_squared()
        if norm_sq < NUMERICAL_EPSILON:
            raise ValueError("Cannot invert zero quaternion")
        return self.conjugate() * (1.0 / norm_sq)

    def to_rotation_matrix(self) -> np.ndarray:
        """Convert to 3×3 rotation matrix."""
        # Normalize first
        q = self.normalize()

        w, x, y, z = q.w, q.x, q.y, q.z

        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """Rotate 3D vector using quaternion."""
        if len(v) != 3:
            raise ValueError("Vector must be 3-dimensional")

        # Convert vector to quaternion
        v_quat = Quaternion(0, v[0], v[1], v[2])

        # Rotation: q * v * q^(-1)
        q_norm = self.normalize()
        rotated = q_norm * v_quat * q_norm.inverse()

        return np.array([rotated.x, rotated.y, rotated.z])

    def to_axis_angle(self) -> tuple[np.ndarray, float]:
        """Convert to axis-angle representation."""
        q = self.normalize()

        angle = 2 * np.arccos(np.clip(abs(q.w), 0, 1))

        if abs(angle) < NUMERICAL_EPSILON:
            return np.array([1, 0, 0]), 0.0

        sin_half_angle = np.sqrt(1 - q.w**2)
        if sin_half_angle < NUMERICAL_EPSILON:
            return np.array([1, 0, 0]), 0.0

        axis = np.array([q.x, q.y, q.z]) / sin_half_angle

        return axis, angle


    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion subtraction."""
        return Quaternion(
            self.w - other.w,
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )


class Octonion:
    """
    Octonion implementation for 8D algebra.

    Non-associative extension of quaternions providing
    representations for exceptional Lie groups.
    """

    def __init__(self, components: np.ndarray):
        """Initialize with 8 components [e₀, e₁, ..., e₇]."""
        if len(components) != 8:
            raise ValueError("Octonion requires exactly 8 components")
        self.components = np.array(components, dtype=float)

    @classmethod
    def from_real(cls, value: float) -> 'Octonion':
        """Create real octonion."""
        components = np.zeros(8)
        components[0] = value
        return cls(components)

    @classmethod
    def from_quaternions(cls, q1: Quaternion, q2: Quaternion) -> 'Octonion':
        """Create octonion from two quaternions using Cayley-Dickson construction."""
        components = np.array([
            q1.w, q1.x, q1.y, q1.z,
            q2.w, q2.x, q2.y, q2.z
        ])
        return cls(components)

    def __add__(self, other: 'Octonion') -> 'Octonion':
        """Add octonions."""
        return Octonion(self.components + other.components)

    def __sub__(self, other: 'Octonion') -> 'Octonion':
        """Subtract octonions."""
        return Octonion(self.components - other.components)

    def __mul__(self, other: Union['Octonion', float]) -> 'Octonion':
        """Multiply octonions (non-associative!)."""
        if isinstance(other, (int, float)):
            return Octonion(self.components * other)

        if not isinstance(other, Octonion):
            return NotImplemented

        # Octonion multiplication using Cayley-Dickson construction
        a, b = self.components[:4], self.components[4:]
        c, d = other.components[:4], other.components[4:]

        # Convert to quaternions for easier calculation
        q_a = Quaternion(a[0], a[1], a[2], a[3])
        q_b = Quaternion(b[0], b[1], b[2], b[3])
        q_c = Quaternion(c[0], c[1], c[2], c[3])
        q_d = Quaternion(d[0], d[1], d[2], d[3])

        # Cayley-Dickson multiplication: (a,b)*(c,d) = (ac-d̄b, da+bc̄)
        result_1 = q_a * q_c - q_d.conjugate() * q_b
        result_2 = q_d * q_a + q_b * q_c.conjugate()

        result_components = np.array([
            result_1.w, result_1.x, result_1.y, result_1.z,
            result_2.w, result_2.x, result_2.y, result_2.z
        ])

        return Octonion(result_components)

    def conjugate(self) -> 'Octonion':
        """Octonion conjugate."""
        conj_components = self.components.copy()
        conj_components[1:] *= -1  # Negate all except real part
        return Octonion(conj_components)

    def norm_squared(self) -> float:
        """Squared norm."""
        return np.sum(self.components**2)

    def norm(self) -> float:
        """Norm."""
        return np.sqrt(self.norm_squared())

    def normalize(self) -> 'Octonion':
        """Normalize to unit octonion."""
        n = self.norm()
        if n < NUMERICAL_EPSILON:
            raise ValueError("Cannot normalize zero octonion")
        return Octonion(self.components / n)

    def inverse(self) -> 'Octonion':
        """Multiplicative inverse."""
        norm_sq = self.norm_squared()
        if norm_sq < NUMERICAL_EPSILON:
            raise ValueError("Cannot invert zero octonion")
        return Octonion(self.conjugate().components / norm_sq)


class LieGroup:
    """
    Base class for Lie group implementations.

    Provides structure for continuous groups with smooth manifold structure,
    essential for dimensional transformation analysis.
    """

    def __init__(self, dimension: int, algebra: Optional[LieAlgebra] = None):
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

        r1 = Rotation.from_matrix(g1)
        r2 = Rotation.from_matrix(g2)

        # SLERP
        interpolated = r1.slerp(r2, t)
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


# =============================================================================
# QUATERNION AND OCTONION ALGEBRAS
# =============================================================================

class QuaternionAlgebra:
    """
    Quaternion algebra H - 4-dimensional associative division algebra.

    Provides structured operations on quaternions with basis {1, i, j, k}
    where i² = j² = k² = ijk = -1.
    """

    def __init__(self):
        self.basis_elements = ['1', 'i', 'j', 'k']
        self.dimension = 4

        # Multiplication table for quaternions
        self.multiplication_table = {
            ('1', '1'): (1, '1'), ('1', 'i'): (1, 'i'), ('1', 'j'): (1, 'j'), ('1', 'k'): (1, 'k'),
            ('i', '1'): (1, 'i'), ('i', 'i'): (-1, '1'), ('i', 'j'): (1, 'k'), ('i', 'k'): (-1, 'j'),
            ('j', '1'): (1, 'j'), ('j', 'i'): (-1, 'k'), ('j', 'j'): (-1, '1'), ('j', 'k'): (1, 'i'),
            ('k', '1'): (1, 'k'), ('k', 'i'): (1, 'j'), ('k', 'j'): (-1, 'i'), ('k', 'k'): (-1, '1'),
        }

    def multiply(self, q1: 'Quaternion', q2: 'Quaternion') -> 'Quaternion':
        """Multiply two quaternions using algebra structure."""
        w1, x1, y1, z1 = q1.components
        w2, x2, y2, z2 = q2.components

        # Hamilton's multiplication formula
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return Quaternion(w, x, y, z)

    def conjugate(self, q: 'Quaternion') -> 'Quaternion':
        """Quaternion conjugate: q* = w - xi - yj - zk."""
        w, x, y, z = q.components
        return Quaternion(w, -x, -y, -z)

    def norm_squared(self, q: 'Quaternion') -> float:
        """Norm squared: |q|² = q * q*."""
        w, x, y, z = q.components
        return w*w + x*x + y*y + z*z

    def inverse(self, q: 'Quaternion') -> 'Quaternion':
        """Multiplicative inverse: q⁻¹ = q* / |q|²."""
        if q.norm() < NUMERICAL_EPSILON:
            raise DimensionalError("Cannot invert zero quaternion")

        conj = self.conjugate(q)
        norm_sq = self.norm_squared(q)

        w, x, y, z = conj.components
        return Quaternion(w/norm_sq, x/norm_sq, y/norm_sq, z/norm_sq)

    def commutator(self, q1: 'Quaternion', q2: 'Quaternion') -> 'Quaternion':
        """Commutator [q1, q2] = q1*q2 - q2*q1."""
        prod1 = self.multiply(q1, q2)
        prod2 = self.multiply(q2, q1)

        # Subtract components
        w = prod1.components[0] - prod2.components[0]
        x = prod1.components[1] - prod2.components[1]
        y = prod1.components[2] - prod2.components[2]
        z = prod1.components[3] - prod2.components[3]

        return Quaternion(w, x, y, z)

    def exp(self, q: 'Quaternion') -> 'Quaternion':
        """Exponential: exp(q) = exp(w) * (cos|v| + sin|v| * v/|v|)."""
        w, x, y, z = q.components
        v = np.array([x, y, z])
        v_norm = np.linalg.norm(v)

        exp_w = np.exp(w)

        if v_norm < NUMERICAL_EPSILON:
            return Quaternion(exp_w, 0, 0, 0)

        cos_v = np.cos(v_norm)
        sin_v = np.sin(v_norm)

        v_unit = v / v_norm

        return Quaternion(
            exp_w * cos_v,
            exp_w * sin_v * v_unit[0],
            exp_w * sin_v * v_unit[1],
            exp_w * sin_v * v_unit[2]
        )

    def log(self, q: 'Quaternion') -> 'Quaternion':
        """Logarithm: log(q) = log|q| + v/|v| * arccos(w/|q|)."""
        w, x, y, z = q.components
        q_norm = q.norm()

        if q_norm < NUMERICAL_EPSILON:
            raise DimensionalError("Cannot take log of zero quaternion")

        v = np.array([x, y, z])
        v_norm = np.linalg.norm(v)

        log_norm = np.log(q_norm)

        if v_norm < NUMERICAL_EPSILON:
            return Quaternion(log_norm, 0, 0, 0)

        theta = np.arccos(np.clip(w / q_norm, -1, 1))
        v_unit = v / v_norm

        return Quaternion(
            log_norm,
            theta * v_unit[0],
            theta * v_unit[1],
            theta * v_unit[2]
        )


class OctonionAlgebra:
    """
    Octonion algebra O - 8-dimensional alternative algebra.

    Non-associative division algebra extending quaternions.
    Basis: {e₀, e₁, e₂, e₃, e₄, e₅, e₆, e₇} where e₀ = 1.
    """

    def __init__(self):
        self.dimension = 8
        self.basis_elements = [f'e{i}' for i in range(8)]

        # Cayley-Dickson construction from quaternions
        self._setup_multiplication_table()

    def _setup_multiplication_table(self):
        """Setup octonion multiplication table using Cayley-Dickson construction."""
        # This is a simplified representation - full table would be 8x8
        # Using the standard Cayley table for octonions
        self.multiplication_signs = np.array([
            [ 1,  1,  1,  1,  1,  1,  1,  1],
            [ 1, -1,  1, -1,  1, -1, -1,  1],
            [ 1, -1, -1,  1,  1,  1, -1, -1],
            [ 1,  1, -1, -1,  1, -1,  1, -1],
            [ 1, -1, -1, -1, -1,  1,  1,  1],
            [ 1,  1, -1,  1, -1, -1,  1, -1],
            [ 1,  1,  1, -1, -1, -1, -1,  1],
            [ 1, -1,  1,  1, -1,  1, -1, -1]
        ])

        # Multiplication structure indices (which unit gives the product)
        self.multiplication_structure = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 0, 3, 2, 5, 4, 7, 6],
            [2, 3, 0, 1, 6, 7, 4, 5],
            [3, 2, 1, 0, 7, 6, 5, 4],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 4, 7, 6, 1, 0, 3, 2],
            [6, 7, 4, 5, 2, 3, 0, 1],
            [7, 6, 5, 4, 3, 2, 1, 0]
        ])

    def multiply(self, oct1: 'Octonion', oct2: 'Octonion') -> 'Octonion':
        """Multiply two octonions using the algebra structure."""
        result_components = np.zeros(8)

        for i in range(8):
            for j in range(8):
                coeff1 = oct1.components[i]
                coeff2 = oct2.components[j]

                if abs(coeff1) > NUMERICAL_EPSILON and abs(coeff2) > NUMERICAL_EPSILON:
                    product_index = self.multiplication_structure[i, j]
                    sign = self.multiplication_signs[i, j]

                    result_components[product_index] += sign * coeff1 * coeff2

        return Octonion(result_components)

    def conjugate(self, oct: 'Octonion') -> 'Octonion':
        """Octonion conjugate: oct* = a₀ - a₁e₁ - ... - a₇e₇."""
        conj_components = oct.components.copy()
        conj_components[1:] *= -1  # Negate all non-real parts
        return Octonion(conj_components)

    def norm_squared(self, oct: 'Octonion') -> float:
        """Norm squared: |oct|² = oct * oct*."""
        return np.sum(oct.components**2)

    def inverse(self, oct: 'Octonion') -> 'Octonion':
        """Multiplicative inverse using conjugate and norm."""
        norm_sq = self.norm_squared(oct)

        if norm_sq < NUMERICAL_EPSILON:
            raise DimensionalError("Cannot invert zero octonion")

        conj = self.conjugate(oct)
        return Octonion(conj.components / norm_sq)

    def associator(self, oct1: 'Octonion', oct2: 'Octonion', oct3: 'Octonion') -> 'Octonion':
        """Associator [oct1, oct2, oct3] = (oct1*oct2)*oct3 - oct1*(oct2*oct3)."""
        # Left association
        left_product = self.multiply(self.multiply(oct1, oct2), oct3)

        # Right association
        right_product = self.multiply(oct1, self.multiply(oct2, oct3))

        # Associator
        result_components = left_product.components - right_product.components
        return Octonion(result_components)

    def is_alternative(self, oct1: 'Octonion', oct2: 'Octonion') -> bool:
        """Check alternative property: (x*x)*y = x*(x*y) and (x*y)*y = x*(y*y)."""
        # Left alternative: (x*x)*y = x*(x*y)
        xx = self.multiply(oct1, oct1)
        left1 = self.multiply(xx, oct2)
        right1 = self.multiply(oct1, self.multiply(oct1, oct2))

        left_alternative = np.allclose(left1.components, right1.components, atol=NUMERICAL_EPSILON)

        # Right alternative: (x*y)*y = x*(y*y)
        yy = self.multiply(oct2, oct2)
        left2 = self.multiply(self.multiply(oct1, oct2), oct2)
        right2 = self.multiply(oct1, yy)

        right_alternative = np.allclose(left2.components, right2.components, atol=NUMERICAL_EPSILON)

        return left_alternative and right_alternative

    def triality_automorphism(self, oct: 'Octonion', triality_type: int = 1) -> 'Octonion':
        """Apply triality automorphism to octonion."""
        if triality_type == 1:
            # Standard triality permutation
            perm_indices = [0, 1, 2, 4, 3, 6, 5, 7]
        elif triality_type == 2:
            # Second triality
            perm_indices = [0, 1, 4, 2, 6, 3, 7, 5]
        else:
            raise ValueError(f"Unknown triality type: {triality_type}")

        new_components = oct.components[perm_indices]
        return Octonion(new_components)


# =============================================================================
# GROUP ACTIONS ON DIMENSIONAL SPACES
# =============================================================================

class DimensionalGroupAction:
    """
    Framework for group actions on dimensional spaces.

    Provides tools for analyzing symmetries and invariants
    in dimensional mathematical structures.
    """

    def __init__(self, group_elements: list[np.ndarray], space_dimension: int):
        self.group_elements = group_elements
        self.space_dimension = space_dimension
        self.group_size = len(group_elements)

        # Verify group properties
        self._verify_group_structure()

    def _verify_group_structure(self):
        """Verify that elements form a group under matrix multiplication."""
        # Check closure (approximately, for finite groups)
        for g1 in self.group_elements:
            for g2 in self.group_elements:
                product = g1 @ g2

                # Check if product is in group (up to numerical precision)
                in_group = False
                for g in self.group_elements:
                    if np.allclose(product, g, atol=NUMERICAL_EPSILON):
                        in_group = True
                        break

                if not in_group:
                    warnings.warn("Group closure not satisfied exactly")
                    break

    def orbit(self, point: np.ndarray) -> np.ndarray:
        """Compute orbit of point under group action."""
        orbit_points = []

        for g in self.group_elements:
            transformed_point = g @ point

            # Check if already in orbit (avoid duplicates)
            is_new = True
            for existing_point in orbit_points:
                if np.allclose(transformed_point, existing_point, atol=NUMERICAL_EPSILON):
                    is_new = False
                    break

            if is_new:
                orbit_points.append(transformed_point)

        return np.array(orbit_points)

    def stabilizer(self, point: np.ndarray) -> list[np.ndarray]:
        """Find stabilizer subgroup of point."""
        stabilizer_elements = []

        for g in self.group_elements:
            transformed_point = g @ point

            if np.allclose(transformed_point, point, atol=NUMERICAL_EPSILON):
                stabilizer_elements.append(g)

        return stabilizer_elements

    def invariant_measure(self, measure_func) -> float:
        """Compute group-invariant measure using averaging."""
        # Sample points in space
        n_samples = 1000
        sample_points = np.random.normal(0, 1, (n_samples, self.space_dimension))

        total_measure = 0.0

        for point in sample_points:
            orbit_points = self.orbit(point)

            # Average measure over orbit
            orbit_measures = [measure_func(p) for p in orbit_points]
            average_measure = np.mean(orbit_measures)

            total_measure += average_measure

        return total_measure / n_samples

    def decompose_representation(self) -> dict[str, Any]:
        """Decompose representation into irreducible components."""
        # Character table computation (simplified)
        character_table = np.zeros((self.group_size, self.group_size), dtype=complex)

        for i, g in enumerate(self.group_elements):
            character_table[i, i] = np.trace(g)  # Character is trace

        # Find multiplicities of irreducible representations
        # This is a simplified version - full implementation would use proper character theory
        eigenvals, eigenvecs = linalg.eig(character_table)

        return {
            'character_table': character_table,
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'representation_dimension': self.space_dimension,
        }


# =============================================================================
# DIMENSIONAL SYMMETRY ANALYSIS
# =============================================================================

def analyze_dimensional_symmetries(
    measure_func,
    dimension_range: tuple[float, float] = (0.1, 10.0),
    n_points: int = 100
) -> dict[str, Any]:
    """
    Analyze symmetries in dimensional measure functions.

    Identifies continuous and discrete symmetries in the behavior
    of dimensional measures across different scales.
    """
    dimensions = np.linspace(dimension_range[0], dimension_range[1], n_points)
    measure_values = np.array([measure_func(d) for d in dimensions])

    # Look for scaling symmetries
    scaling_analysis = {}

    for scale_factor in [2.0, PHI, PI, np.e]:
        scaled_dims = dimensions * scale_factor
        # Interpolate to get values at scaled dimensions
        scaled_measures = np.interp(scaled_dims, dimensions, measure_values)

        # Compare with original (appropriately scaled)
        correlation = np.corrcoef(measure_values, scaled_measures)[0, 1]

        scaling_analysis[f'scale_{scale_factor:.3f}'] = {
            'correlation': correlation,
            'scaling_factor': scale_factor,
            'is_symmetric': abs(correlation) > 0.95,
        }

    # Look for translational symmetries
    translation_analysis = {}

    for shift in [1.0, PHI, PI]:
        shifted_dims = dimensions + shift
        # Only consider overlap region
        overlap_mask = (shifted_dims >= dimension_range[0]) & (shifted_dims <= dimension_range[1])

        if np.sum(overlap_mask) > 10:  # Need sufficient overlap
            original_overlap = measure_values[overlap_mask]
            shifted_overlap = np.interp(shifted_dims[overlap_mask], dimensions, measure_values)

            correlation = np.corrcoef(original_overlap, shifted_overlap)[0, 1]

            translation_analysis[f'shift_{shift:.3f}'] = {
                'correlation': correlation,
                'shift_amount': shift,
                'is_symmetric': abs(correlation) > 0.95,
            }

    # Look for reflection symmetries around special points
    reflection_analysis = {}

    special_points = [PHI, PI, np.e, 3.0]  # Common special dimensions

    for center in special_points:
        if dimension_range[0] < center < dimension_range[1]:
            # Reflect dimensions around center
            reflected_dims = 2 * center - dimensions

            # Only consider points within range
            valid_mask = (reflected_dims >= dimension_range[0]) & (reflected_dims <= dimension_range[1])

            if np.sum(valid_mask) > 10:
                original_valid = measure_values[valid_mask]
                reflected_valid = np.interp(reflected_dims[valid_mask], dimensions, measure_values)

                correlation = np.corrcoef(original_valid, reflected_valid)[0, 1]

                reflection_analysis[f'center_{center:.3f}'] = {
                    'correlation': correlation,
                    'reflection_center': center,
                    'is_symmetric': abs(correlation) > 0.95,
                }

    return {
        'scaling_symmetries': scaling_analysis,
        'translation_symmetries': translation_analysis,
        'reflection_symmetries': reflection_analysis,
        'dimensions': dimensions,
        'measure_values': measure_values,
    }


# =============================================================================
# MODULE TESTS
# =============================================================================

def test_algebra_module():
    """Test algebraic structures implementations."""
    try:
        # Test Clifford algebra
        cl_2_0 = CliffordAlgebra(2, 0)  # Cl(2,0) ≅ ℍ (quaternions)
        v1 = cl_2_0.vector(np.array([1, 0]))
        v2 = cl_2_0.vector(np.array([0, 1]))
        v1 * v2

        # Test quaternions
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion.from_axis_angle(np.array([0, 0, 1]), PI/2)
        q_product = q1 * q2

        # Test quaternion algebra
        q_algebra = QuaternionAlgebra()
        q_test1 = Quaternion(1, 1, 0, 0)
        q_test2 = Quaternion(0, 1, 1, 0)
        q_algebra.multiply(q_test1, q_test2)

        # Test SO(3) Lie algebra
        so3 = SO3LieAlgebra()
        axis = np.array([0, 0, 1])
        R = so3.rotation_matrix(axis, PI/4)

        # Test octonions
        oct1 = Octonion.from_real(1.0)
        oct2 = Octonion(np.array([0, 1, 0, 0, 0, 0, 0, 0]))
        oct_product = oct1 * oct2

        # Test octonion algebra
        oct_algebra = OctonionAlgebra()
        oct_test1 = Octonion(np.array([1, 1, 0, 0, 0, 0, 0, 0]))
        oct_test2 = Octonion(np.array([0, 0, 1, 0, 0, 0, 0, 0]))
        oct_algebra.multiply(oct_test1, oct_test2)

        return {
            'module_status': 'operational',
            'clifford_algebra_dim': cl_2_0.algebra_dim,
            'quaternion_norm': q_product.norm(),
            'quaternion_algebra_dim': q_algebra.dimension,
            'rotation_matrix_det': np.linalg.det(R),
            'octonion_norm': oct_product.norm(),
            'octonion_algebra_dim': oct_algebra.dimension,
            'tests_passed': True,
        }

    except Exception as e:
        return {
            'module_status': 'error',
            'error_message': str(e),
            'tests_passed': False,
        }


if __name__ == "__main__":
    # Test the algebra module
    test_results = test_algebra_module()
    print("ALGEBRAIC STRUCTURES MODULE TEST")
    print("=" * 50)
    for key, value in test_results.items():
        print(f"{key}: {value}")
