"""Core package for backward compatibility."""

# Re-export functions from sibling core.py module
import importlib.util
import os
import sys

# Import constants from the constants module
from .constants import CRITICAL_DIMENSIONS, NUMERICAL_EPSILON, PHI, PI, PSI, VARPI, E


# Error classes
class DimensionalError(Exception):
    """Base exception for dimensional errors."""
    pass

class NumericalInstabilityError(DimensionalError):
    """Exception for numerical instability."""
    pass

# Load sibling core.py module
core_py_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core.py')
if os.path.exists(core_py_path):
    spec = importlib.util.spec_from_file_location("dimensional_core_module", core_py_path)
    core_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core_module)

    # Re-export everything
    v = core_module.v
    s = core_module.s
    c = core_module.c
    r = core_module.r
    gamma = core_module.gamma
    digamma = core_module.digamma
    Ball = core_module.Ball
    Sphere = core_module.Sphere
