"""
Algebra Module
==============

Refactored from monolithic algebra.py (1,412 lines) into focused submodules.
"""

from .clifford import CliffordAlgebra, CliffordMultivector
from .group_actions import DimensionalGroupAction, analyze_dimensional_symmetries
from .lie_algebra import LieAlgebra, SLnLieAlgebra, SO3LieAlgebra
from .lie_group import LieGroup, SLnGroup, SO3Group
from .octonion import Octonion, OctonionAlgebra
from .quaternion import Quaternion, QuaternionAlgebra

__all__ = [
    # Clifford algebra
    'CliffordAlgebra',
    'CliffordMultivector',
    # Group actions
    'DimensionalGroupAction',
    'analyze_dimensional_symmetries',
    # Lie algebras
    'LieAlgebra',
    'SO3LieAlgebra',
    'SLnLieAlgebra',
    # Lie groups
    'LieGroup',
    'SO3Group',
    'SLnGroup',
    # Octonions
    'Octonion',
    'OctonionAlgebra',
    # Quaternions
    'Quaternion',
    'QuaternionAlgebra',
]
