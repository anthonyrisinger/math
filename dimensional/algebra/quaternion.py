"""
Quaternion Module
=================

Quaternions for 3D/4D rotations, extending complex numbers to provide
efficient representation of rotations in 3D space without gimbal lock.
"""

from typing import Union

import numpy as np

from ..mathematics import NUMERICAL_EPSILON, DimensionalError


class Quaternion:
    """
    Quaternion implementation for 3D/4D rotations.

    Extends complex numbers to provide efficient representation
    of rotations in 3D space without gimbal lock.
    """

    def __init__(self, w: float, x: float, y: float, z: float):
        """
        Initialize quaternion.

        Parameters
        ----------
        w : float
            Real part
        x : float
            i component
        y : float
            j component
        z : float
            k component
        """
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

    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        """Subtract quaternions."""
        return Quaternion(
            self.w - other.w,
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
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

        # Quaternion multiplication (Hamilton product)
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


class QuaternionAlgebra:
    """
    Quaternion algebra H - 4-dimensional associative division algebra.

    Provides structured operations on quaternions with basis {1, i, j, k}
    where i² = j² = k² = ijk = -1.
    """

    def __init__(self):
        """Initialize quaternion algebra structure."""
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
