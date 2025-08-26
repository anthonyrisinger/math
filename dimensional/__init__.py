#!/usr/bin/env python3
"""
Dimensional Mathematics Package
===============================

Unified package for dimensional emergence theory, gamma functions,
morphic mathematics, and phase dynamics.

Quick start:
    from dimensional import *
    explore(4)      # Explore dimension 4
    instant()       # Quick visualization
    lab()          # Interactive lab
"""

# Import everything from gamma module - prevent numpy gamma conflict
from .gamma import (
    explore, peaks, demo, live, instant, qplot, lab,
    gamma_safe, gammaln_safe, digamma_safe, factorial_extension,
    v, s, c, r, ρ, v_peak, s_peak, c_peak,
    γ, ln_γ, ψ, abs_γ,
    gamma_explorer, gamma_comparison_plot,
    quick_gamma_analysis, peaks_analysis
)

# Import core constants with hybrid imports for flexibility
try:
    from ..core import PI, PHI, PSI, E, VARPI
    from ..core.constants import SQRT_PI, NUMERICAL_EPSILON
except ImportError:
    # Fallback for script execution
    from core import PI, PHI, PSI, E, VARPI
    from core.constants import SQRT_PI, NUMERICAL_EPSILON

# Import specific functions from modules to prevent namespace conflicts
from .measures import (
    measures_explorer, peak_finder, critical_analysis, comparative_plot,
    quick_measure_analysis, is_critical_dimension, volume_ratio, surface_ratio
)

# Import missing functions from core modules that tests expect
try:
    from ..core.measures import find_all_peaks
    from ..core.phase import sap_rate, total_phase_energy
except ImportError:
    from core.measures import find_all_peaks
    from core.phase import sap_rate, total_phase_energy

from .morphic import (
    morphic_polynomial_roots, real_roots, discriminant, 
    k_perfect_circle, k_discriminant_zero, golden_ratio_properties,
    morphic_scaling_factor, generate_morphic_sequence,
    make_rotor, sample_loop_xyz, morphic_circle_transform
)

from .phase import (
    PhaseDynamicsEngine, quick_phase_analysis, quick_emergence_analysis
)

from .pregeometry import (
    PreGeometry, PreGeometryVisualizer
)# Import modern visualization components
# TEMPORARILY DISABLED - BLOCKING GAMMA MODULE IMPORTS
# try:
#     from visualization import PlotlyDashboard, KingdonRenderer
#     from visualization.modernized_dashboard import create_modern_dashboard
#     VISUALIZATION_AVAILABLE = True
# except ImportError:
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

# Uppercase aliases for test compatibility
V = v  # Volume function
S = s  # Surface function  
C = c  # Complexity function

# Import phase analysis functions
from .phase import PhaseDynamicsEngine, quick_phase_analysis, quick_emergence_analysis
