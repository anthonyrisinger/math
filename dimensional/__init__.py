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
# Import algebraic structures
from .algebra import (
    CliffordAlgebra,
    CliffordMultivector,
    DimensionalGroupAction,
    LieAlgebra,
    LieGroup,
    Octonion,
    Quaternion,
    SLnGroup,
    SLnLieAlgebra,
    SO3Group,
    SO3LieAlgebra,
    analyze_dimensional_symmetries,
)

# Import module-specific functions
from .gamma import (
    c_peak,
    explore,
    instant,
    lab,
    peaks,
    quick_gamma_analysis,
    s_peak,
    v_peak,
)
from .mathematics import (
    # Constants
    CRITICAL_DIMENSIONS,
    NUMERICAL_EPSILON,
    PHI,
    PI,
    PSI,
    VARPI,
    ConvergenceError,
    # Exception classes
    DimensionalError,
    InvalidDimensionError,
    NumericalInstabilityError,
    # Core functions
    ball_volume,
    complexity_measure,
    find_peak,
    gamma_safe,
    gammaln_safe,
    sphere_surface,
)

# Import spectral analysis capabilities
from .spectral import (
    DimensionalOperator,
    analyze_critical_point_spectrum,
    analyze_emergence_spectrum,
    detect_dimensional_resonances,
    dimensional_spectral_density,
    dimensional_wavelet_analysis,
    fractal_harmonic_analysis,
    quick_spectral_analysis,
)

# Import enhanced research CLI functions (gracefully handle missing deps)
try:
    from .research_cli import (
        InteractiveParameterSweep,
        PublicationExporter,
        ResearchPersistence,
        ResearchSession,
        RichVisualizer,
        enhanced_explore,
        enhanced_instant,
        enhanced_lab,
    )
    ENHANCED_RESEARCH_AVAILABLE = True
except ImportError:
    ENHANCED_RESEARCH_AVAILABLE = False
from .interface import UnifiedInterface
from .mathematics.functions import find_all_peaks
from .measures import find_all_peaks as measures_find_all_peaks
from .phase import PhaseDynamicsEngine, quick_phase_analysis

# Consolidated API aliases - both uppercase and lowercase
V = v = ball_volume          # V(d) = v(d) = d-dimensional ball volume
S = s = sphere_surface       # S(d) = s(d) = d-dimensional sphere surface
C = c = complexity_measure   # C(d) = c(d) = V(d) * S(d)


# Emergence simulation - Production implementation
def run_emergence_simulation(*args, **kwargs):
    """
    Run emergence simulation using phase dynamics engine.
    
    This function provides a production-ready interface to phase dynamics
    emergence patterns across dimensional transitions.
    
    Parameters
    ----------
    steps : int, optional
        Number of evolution steps (default: 1000)
    initial_dimension : float, optional
        Starting dimension (default: 3.0)
    coupling_strength : float, optional
        Phase coupling parameter (default: 0.1)
        
    Returns
    -------
    dict
        Simulation results with final state and emergence metrics
    """
    import numpy as np
    from .phase import PhaseDynamicsEngine
    
    # Extract parameters with defaults
    steps = kwargs.get("steps", 1000)
    initial_dim = kwargs.get("initial_dimension", 3.0)
    coupling = kwargs.get("coupling_strength", 0.1)
    
    # Initialize phase dynamics engine
    engine = PhaseDynamicsEngine(
        max_dimensions=int(initial_dim * 2),  # Use enough dimensions
        use_adaptive=True
    )
    
    # Run emergence simulation
    final_dimension = initial_dim
    for _ in range(steps):
        # Simple evolution step
        engine.step(dt=0.01)
        # Track emergence patterns
        if len(engine.phase_state_history) > 10:
            # Basic convergence check
            recent_phases = engine.phase_state_history[-10:]
            variance = sum(np.mean(np.abs(p - engine.phase_density) ** 2) for p in recent_phases) / 10
            if variance < 1e-6:
                break
    
    # Calculate emerged dimension from phase state
    phase_magnitudes = np.abs(engine.phase_density)
    active_dimensions = np.sum(phase_magnitudes > 1e-6)
    final_dimension = float(initial_dim + active_dimensions * 0.1)
    
    return {
        "status": "completed",
        "steps_executed": len(engine.history),
        "final_dimension": final_dimension,
        "convergence_achieved": len(engine.history) < steps,
        "active_dimensions": int(active_dimensions),
        "phase_magnitudes": phase_magnitudes.tolist(),
        "phase_engine": engine
    }


# Import enhanced morphic functions

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
    """Show quick start examples with enhanced research capabilities."""
    print(__doc__)
    print("\n🚀 Enhanced Research Commands:")
    if ENHANCED_RESEARCH_AVAILABLE:
        print("  🔬 RESEARCH MODE:")
        print("    enhanced_lab(4)     # Full research laboratory")
        print("    enhanced_explore(4) # Guided dimensional discovery")
        print("    enhanced_instant()  # Multi-panel analysis")
        print("\n  📊 PARAMETER SWEEPS:")
        print("    sweeper = InteractiveParameterSweep(visualizer)")
        print("    sweep = sweeper.run_dimension_sweep(2, 8, 50)")
        print("\n  💾 SESSION MANAGEMENT:")
        print("    persistence = ResearchPersistence()")
        print("    session = persistence.load_session('lab_123')")
        print("\n  📈 RICH VISUALIZATION:")
        print("    viz = RichVisualizer()")
        print("    viz.show_dimensional_analysis(point)")
        print("    viz.show_critical_dimensions_tree()")
        print("\n  🎯 PUBLICATION EXPORTS:")
        print("    exporter = PublicationExporter()")
        print("    exporter.export_csv_data(sweep)")
    else:
        print("  [Basic mode - install research dependencies for enhanced features]")

    print("\n🚀 Basic Commands:")
    print("  explore(4)     # Explore dimension 4")
    print("  peaks()        # Find all peaks")
    print("  instant()      # 4-panel visualization")
    print("  lab()          # Interactive lab")
    print("\n📊 Measures:")
    print("  V(4)           # Ball volume at d=4")
    print("  S(4)           # Sphere surface at d=4")
    print("  C(4)           # Complexity at d=4")
    print("  find_all_peaks() # Find all measure peaks")
    print("\n⚡ Phase Dynamics:")
    print("  run_emergence_simulation() # Run emergence sim")
    print("  quick_phase_analysis(4)    # Analyze dimension 4")
    print("  PhaseDynamicsEngine()      # Full engine")


# Convenience aliases (only include functions that exist)
γ_analysis = quick_gamma_analysis

# Direct research platform interface import

# quick_start is already defined above - no override needed

print(
    f"📐 Dimensional Mathematics Foundation loaded - " f"φ = {PHI:.6f} ready!"
)
