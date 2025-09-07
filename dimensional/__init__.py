"""
Dimensional Mathematics - Minimal Core.

A lightweight library for dimensional geometry and gamma functions.
"""

__version__ = "3.0.0"
__author__ = "Dimensional Mathematics Team"
__description__ = "A lightweight library for dimensional geometry and gamma functions"

# Import core functions from core.py module (not core/ package)
import importlib.util
import os
import sys

# Load core.py as a module to avoid confusion with core/ package
core_py_path = os.path.join(os.path.dirname(__file__), 'core.py')
spec = importlib.util.spec_from_file_location("dimensional_core", core_py_path)
core_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core_module)

# Import the functions we need
v = core_module.v
s = core_module.s
c = core_module.c
r = core_module.r
gamma = core_module.gamma
digamma = core_module.digamma
PI = core_module.PI
E = core_module.E
PHI = core_module.PHI

# Uppercase aliases for compatibility
V = v
S = s
C = c
R = r

# Import constants for test compatibility
from .constants_pkg.constants import NUMERICAL_EPSILON

# Import additional constants
try:
    from .core.constants import PSI, VARPI
except ImportError:
    # Fallback values
    PSI = 0.6180339887498948
    VARPI = 1.311028777

# Import geometry
from .geometry import Ball, Sphere

# Import exploration features - THE MAIN FEATURES!
try:
    # Try to import rich version if available
    from .explore import explore, instant, lab, peaks
except ImportError:
    # Fall back to simple version
    from .viz import demo, explore, instant, lab, peaks

# Import report generator - THE KILLER FEATURE!
try:
    from .report import generate_report
except ImportError:
    generate_report = None

def quick_start():
    """Quick start demonstration."""
    print("Dimensional Mathematics Framework")
    print("=" * 35)
    print(f"V(4) = {v(4):.6f}")
    print(f"S(4) = {s(4):.6f}")
    print(f"Γ(3.5) = {gamma(3.5):.6f}")
    print("Use explore(d) to explore dimension d")
    print("Use peaks() to find critical dimensions")

# Public API
__all__ = [
    # MAIN FEATURES - Visual exploration
    'explore', 'instant', 'lab', 'peaks',
    # Functions
    'v', 's', 'c', 'r',
    'V', 'S', 'C', 'R',  # Uppercase aliases
    'gamma', 'digamma',
    # Geometry
    'Ball', 'Sphere',
    # Constants
    'PI', 'E', 'PHI', 'PSI', 'VARPI', 'NUMERICAL_EPSILON',
    # Utilities
    'quick_start',
    'generate_report',
]
