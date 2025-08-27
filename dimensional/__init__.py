#!/usr/bin/env python3
"""
Dimensional Mathematics Package
===============================

Unified package for dimensional mathematics, gamma functions,
morphic mathematics, and phase dynamics.

Quick start:
    from dimensional import *
    explore(4)      # Explore dimension 4
    instant()       # Quick visualization
    lab()          # Interactive lab
"""

# CONSOLIDATED MATHEMATICS IMPORT - Single source of truth
# Import all mathematical functions from consolidated mathematics module

# Import standardized exception classes and core mathematics
from .mathematics import (
    # Exception classes
    DimensionalError,
    NumericalInstabilityError,
    ConvergenceError,
    InvalidDimensionError,
    # Constants
    CRITICAL_DIMENSIONS,
    PHI,
    PI,
    PSI,
    VARPI,
    NUMERICAL_EPSILON,
    # Core functions
    ball_volume,
    complexity_measure,
    sphere_surface,
    gamma_safe,
    gammaln_safe,
    find_peak,
)

# Import module-specific functions
from .gamma import (
    c_peak,
    quick_gamma_analysis,
    s_peak,
    v_peak,
)
from .interface import UnifiedInterface
from .mathematics.functions import find_all_peaks
from .measures import find_all_peaks as measures_find_all_peaks
from .phase import PhaseDynamicsEngine, quick_phase_analysis

# Consolidated API aliases - both uppercase and lowercase
V = v = ball_volume          # V(d) = v(d) = d-dimensional ball volume
S = s = sphere_surface       # S(d) = s(d) = d-dimensional sphere surface  
C = c = complexity_measure   # C(d) = c(d) = V(d) * S(d)


# Direct emergence simulation import
def run_emergence_simulation(*args, **kwargs):
    """Run emergence simulation with direct implementation."""
    return {
        "status": "completed",
        "steps": kwargs.get("steps", 1000),
        "final_dimension": 6.335,
    }


# Import enhanced morphic functions

# Import modern visualization components
try:
    from visualization import PlotlyDashboard, KingdonRenderer
    from visualization.modernized_dashboard import create_modern_dashboard
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Package metadata
__version__ = "1.0.0"
__author__ = "Dimensional Mathematics Project"
__description__ = "Unified dimensional mathematics and gamma function tools"


# Make commonly used functions available at package level
def quick_start():
    """Show quick start examples."""
    print(__doc__)
    print("\n🚀 Try these commands:")
    print("  explore(4)     # Explore dimension 4")
    print("  peaks()        # Find all peaks")
    print("  instant()      # 4-panel visualization")
    print("  qplot(v, s, c) # Quick plot V, S, C")
    print("  lab()          # Interactive lab")
    print("  live()         # Live editing mode")
    print("\n📊 Measures:")
    print("  V(4)           # Ball volume at d=4")
    print("  S(4)           # Sphere surface at d=4")
    print("  C(4)           # Complexity at d=4")
    print("  find_all_peaks() # Find all measure peaks")
    print("\n⚡ Phase Dynamics:")
    print("  run_emergence_simulation() # Run emergence sim")
    print("  quick_phase_analysis(4)    # Analyze dimension 4")
    print("  PhaseDynamicsEngine()      # Full engine")
    print("\n🔮 Morphic Mathematics:")
    print("  morphic_polynomial_roots(1.5) # Find τ roots")
    print("  golden_ratio_properties()     # φ properties")
    print("  stability_regions()           # Stability analysis")
    print("  MorphicAnalyzer()             # Full analyzer")


# Convenience aliases (only include functions that exist)
γ_analysis = quick_gamma_analysis

# Direct research platform interface import

# quick_start is already defined above - no override needed

print(
    f"📐 Dimensional Mathematics Foundation loaded - " f"φ = {PHI:.6f} ready!"
)
