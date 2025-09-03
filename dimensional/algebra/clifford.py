"""
Clifford Algebra Module
=======================

Clifford algebras for multidimensional geometric operations, providing
a natural framework for rotations and reflections in arbitrary dimensions.
"""

from itertools import combinations

import numpy as np

from ..mathematics import NUMERICAL_EPSILON


class CliffordAlgebra:
    """
    Clifford algebra implementation for multidimensional geometric operations.

    Clifford algebras Cl(p,q) generalize complex numbers and quaternions,
    providing natural framework for rotations and reflections in arbitrary dimensions.
    """

    def __init__(self, p: int, q: int = 0, r: int = 0):
        """
        Initialize Clifford algebra Cl(p,q,r).

        Parameters
        ----------
        p : int
            Number of basis vectors with e_i² = +1
        q : int
            Number of basis vectors with e_i² = -1
        r : int
            Number of basis vectors with e_i² = 0 (degenerate)
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
        """
        Initialize multivector.

        Parameters
        ----------
        algebra : CliffordAlgebra
            Parent Clifford algebra
        coefficients : np.ndarray
            Coefficients for each basis element
        """
        self.algebra = algebra
        self.coefficients = coefficients.copy()

    def __add__(self, other: 'CliffordMultivector') -> 'CliffordMultivector':
        """Add two multivectors."""
        if self.algebra != other.algebra:
            raise ValueError("Multivectors must belong to same algebra")
        return CliffordMultivector(self.algebra, self.coefficients + other.coefficients)

    def __sub__(self, other: 'CliffordMultivector') -> 'CliffordMultivector':
        """Subtract two multivectors."""
        if self.algebra != other.algebra:
            raise ValueError("Multivectors must belong to same algebra")
        return CliffordMultivector(self.algebra, self.coefficients - other.coefficients)

    def __mul__(self, other: 'CliffordMultivector') -> 'CliffordMultivector':
        """Multiply two multivectors using Clifford product."""
        if self.algebra != other.algebra:
            raise ValueError("Multivectors must belong to same algebra")

        result_coeffs = np.zeros_like(self.coefficients)

        # Use multiplication table
        for i, coeff_i in enumerate(self.coefficients):
            if abs(coeff_i) < NUMERICAL_EPSILON:
                continue
            for j, coeff_j in enumerate(other.coefficients):
                if abs(coeff_j) < NUMERICAL_EPSILON:
                    continue

                # Get result from multiplication table
                sign = self.algebra.mult_signs[i, j]
                result_idx = int(self.algebra.mult_indices[i, j])

                if sign != 0:
                    result_coeffs[result_idx] += coeff_i * coeff_j * sign

        return CliffordMultivector(self.algebra, result_coeffs)

    def grade_projection(self, grade: int) -> 'CliffordMultivector':
        """Project multivector onto specified grade."""
        result_coeffs = np.zeros_like(self.coefficients)

        for i, basis_grade in enumerate(self.algebra.basis_grades):
            if basis_grade == grade:
                result_coeffs[i] = self.coefficients[i]

        return CliffordMultivector(self.algebra, result_coeffs)

    def reverse(self) -> 'CliffordMultivector':
        """Reverse operation (reverses order of basis vectors)."""
        result_coeffs = np.zeros_like(self.coefficients)

        for i, grade in enumerate(self.algebra.basis_grades):
            # Reverse introduces sign (-1)^(k(k-1)/2) for grade k
            sign = (-1)**(grade * (grade - 1) // 2)
            result_coeffs[i] = self.coefficients[i] * sign

        return CliffordMultivector(self.algebra, result_coeffs)

    def conjugate(self) -> 'CliffordMultivector':
        """Clifford conjugation (combination of reverse and grade involution)."""
        result_coeffs = np.zeros_like(self.coefficients)

        for i, grade in enumerate(self.algebra.basis_grades):
            # Conjugation introduces sign (-1)^(k(k+1)/2)
            sign = (-1)**(grade * (grade + 1) // 2)
            result_coeffs[i] = self.coefficients[i] * sign

        return CliffordMultivector(self.algebra, result_coeffs)

    def norm_squared(self) -> float:
        """Compute squared norm of multivector."""
        # Norm squared is scalar part of (self * self.reverse())
        product = self * self.reverse()
        return float(product.coefficients[0].real)

    def norm(self) -> float:
        """Compute norm of multivector."""
        return np.sqrt(max(0, self.norm_squared()))
