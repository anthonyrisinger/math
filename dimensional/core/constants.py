"""Mathematical constants for dimensional mathematics."""

import math

# Numerical precision constants - more realistic for gamma functions
NUMERICAL_EPSILON = 1e-12

# Mathematical constants
PI = math.pi
E = math.e
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
PSI = 2 / (1 + math.sqrt(5))  # Golden ratio conjugate (1/PHI)
EULER_GAMMA = -0.5772156649  # Euler-Mascheroni constant (digamma(1))
VARPI = 1.311028777  # Special dimensional constant

# Critical dimensions where interesting behavior occurs
CRITICAL_DIMENSIONS = {
    'volume_peak': 5.2569464,    # Volume peak
    'surface_peak': 7.2569464,    # Surface peak
    'complexity_peak': 6.3352,    # Complexity peak
    'integer_2d': 2.0,           # Integer dimension
    'integer_3d': 3.0,           # 3D space
    'integer_4d': 4.0,           # 4D space
    'pi_boundary': PI,           # Pi boundary
    'e_critical': E,             # E critical point
    'leech_limit': 24.0,         # Leech lattice dimension
}
