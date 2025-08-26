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

# Import specific functions from modules to avoid star imports
from .gamma import (
    digamma_safe,
    factorial_extension,
    gamma_comparison_plot,
    gamma_explorer,
    gamma_safe,
    gammaln_safe,
    quick_gamma_analysis,
)
from .measures import (
    C,  # Convenience aliases
    S,
    V,
    ball_volume,
    complexity_measure,
    find_all_peaks,
    sphere_surface,
)
from .morphic import (
    MorphicAnalyzer,
    golden_ratio_properties,
    morphic_polynomial_roots,
    stability_regions,
)
from .phase import (
    PhaseDynamicsEngine,
    dimensional_explorer,
    quick_emergence_analysis,
)

# Import modern visualization components
try:
    from visualization import KingdonRenderer, PlotlyDashboard
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


# Convenience aliases
γ_explorer = gamma_explorer
γ_analysis = quick_gamma_analysis

# Import lowercase aliases for CLI compatibility
from .gamma import demo, explore, instant, lab, live, peaks, qplot
from .measures import c, s, v

# ARCHITECTURAL BYPASS: Direct constants export to avoid contamination
PHI = 1.618033988749895  # Golden ratio φ = (1+√5)/2
PSI = 0.618033988749895  # Golden conjugate ψ = 1/φ
PI = 3.141592653589793   # π
E = 2.718281828459045    # Euler's number e

# Critical dimensions for convenience
CRITICAL_D_PI = PI                    # π ≈ 3.14159
CRITICAL_D_E = E                      # e ≈ 2.71828  
CRITICAL_D_PHI_SQ = PHI * PHI         # φ² ≈ 2.618
CRITICAL_D_2PI = 2 * PI               # 2π ≈ 6.283

# ARCHITECTURAL WORKAROUND: Restore stdout (contamination bypass complete)
try:
    sys.stdout = _original_stdout
except:
    pass  # Already restored or not redirected
