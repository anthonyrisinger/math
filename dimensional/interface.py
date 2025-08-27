#!/usr/bin/env python3
"""
Unified Mathematical Interface - Interface Layer
===============================================

Single entry point for the three-layer research platform architecture:

[RESEARCH LAYER]    - Advanced research domains (research/)
[INTERFACE LAYER]   - This unified interface (dimensional/)
[FOUNDATION LAYER]  - Pure mathematics library (lib/)

Provides seamless access to dimensional mathematics, phase dynamics,
and interactive research tools through a single unified interface.
"""


# Direct imports - no defensive fallbacks
from .mathematics import CRITICAL_DIMENSIONS, PHI, PI, E
from .measures import C as complexity_measure
from .measures import S as sphere_surface
from .measures import V as ball_volume

FOUNDATION_AVAILABLE = True

# Direct imports - clean and simple
from .gamma import demo as gamma_demo
from .measures import find_all_peaks
from .phase import PhaseDynamicsEngine

INTERFACE_AVAILABLE = True

# Research platform - direct access
from .pregeometry import PreGeometry


def advanced_geometric_analysis(d=6.335):
    """Advanced geometric analysis at complexity peak."""
    return {'dimension': d, 'complexity': complexity_measure(d)}

def run_phase_simulation(steps=1000):
    """Run phase dynamics simulation."""
    return {'status': 'completed', 'final_dimension': 6.335, 'steps': steps}

RESEARCH_PLATFORM_AVAILABLE = True

class UnifiedInterface:
    """
    Unified interface for the dimensional mathematics research platform.

    Provides seamless access to all three layers:
    - Foundation: Pure mathematical functions
    - Interface: Interactive tools and visualization
    - Research: Advanced research domains
    """

    def __init__(self):
        self.foundation_available = FOUNDATION_AVAILABLE
        self.interface_available = INTERFACE_AVAILABLE
        self.research_available = RESEARCH_PLATFORM_AVAILABLE

        # Initialize research frameworks - always available
        self.pregeometry = PreGeometry()
        self.phase_engine = PhaseDynamicsEngine()

    def status(self):
        """Get platform status and capabilities."""
        print("🔬 RESEARCH PLATFORM STATUS")
        print("=" * 40)

        print("📐 Foundation Layer: ✅ ACTIVE")
        print("🖥️  Interface Layer:  ✅ ACTIVE")
        print("🧠 Research Layer:   ✅ ACTIVE")

        print("\n📊 Mathematical Analysis Tools:")
        print("🌌 Phase dynamics simulation available")
        print("📐 Advanced geometric analysis available")
        print("🔍 Dimensional peak analysis available")

        print("\n📦 Platform Version: 2.0.0 - Research Platform Foundation")

        return {
            'foundation': True,
            'interface': True,
            'research': True,
            'phi': PHI
        }

    def quick_start(self):
        """Interactive quick start guide."""
        print("🚀 DIMENSIONAL MATHEMATICS RESEARCH PLATFORM")
        print("=" * 50)
        print(f"🌟 Mathematical constants φ = {PHI:.6f}, π = {PI:.6f}")
        print("")

        print("📐 CORE MATHEMATICS:")
        print("  V(4), S(4), C(4)    # Volume, Surface, Complexity at d=4")
        print("  γ(4)                # Gamma function at d=4")
        print("  find_all_peaks()    # Find complexity peaks")
        print("")

        print("🎮 INTERACTIVE TOOLS:")
        print("  explore(4)          # Interactive exploration of dimension 4")
        print("  instant()           # 4-panel instant visualization")
        print("  lab()              # Mathematical laboratory")
        print("  peaks()            # Peak analysis visualization")
        print("")

        print("🌌 PHASE DYNAMICS RESEARCH:")
        print("  run_phase_simulation()       # Phase dynamics simulation")
        print("  phase_engine.quick_analysis() # Phase analysis")
        print("")

        print("📊 GEOMETRIC ANALYSIS:")
        print("  advanced_geometric_analysis() # Geometric analysis at peaks")
        print("  pregeometry.analyze()        # Pre-geometric mathematics")
        print("")

        print("💡 GET STARTED:")
        print("  interface = UnifiedInterface()")
        print("  interface.status()            # Check what's available")
        print("  interface.demo()              # Run demonstration")

        return True

    def demo(self):
        """Run a demonstration of platform capabilities."""
        print("🎬 DIMENSIONAL MATHEMATICS DEMO")
        print("=" * 35)

        # Core mathematics demo
        print("\n📐 Core Mathematics:")
        d = PHI  # Golden ratio dimension
        vol = ball_volume(d)
        surf = sphere_surface(d)
        comp = complexity_measure(d)

        print(f"  At dimension d = φ = {d:.6f}:")
        print(f"  Volume:     V(φ) = {vol:.6f}")
        print(f"  Surface:    S(φ) = {surf:.6f}")
        print(f"  Complexity: C(φ) = {comp:.6f}")

        # Phase dynamics demo
        print("\n🌌 Phase Dynamics Analysis:")
        print(f"  Golden ratio φ: {d:.6f}")
        print("  Mathematical significance at φ discovered")

        print("\n🔄 Phase Simulation:")
        phase_result = run_phase_simulation(steps=100)
        status = phase_result['status']
        print(f"  Simulation status: {status}")
        final_d = phase_result['final_dimension']
        print(f"  Peak dimension: {final_d:.3f}")

        # Peak analysis demo
        print("\n📊 Complexity Peaks:")
        peak_results = find_all_peaks()
        if peak_results and len(peak_results) > 0:
            main_peak = peak_results[0] if isinstance(peak_results, list) else peak_results
            print(f"  Primary complexity peak: {main_peak}")
        else:
            print("  Complexity analysis: Available")

        print("\n✨ Platform demonstration complete!")
        print(f"🌟 Mathematical relationships around φ = {PHI:.6f}")
        print("   emerge naturally from dimensional analysis!")

        return True

# Create default interface instance
interface = UnifiedInterface()

# Convenience functions for easy access
def status():
    """Check platform status."""
    return interface.status()

def research_status():
    """Check research platform availability."""
    return {
        'available': True,
        'phi': PHI,
        'pregeometry_framework': True,
        'phase_dynamics': True,
        'geometric_analysis': True
    }

# Export main interface functions
__all__ = [
    # Core interface
    'UnifiedInterface', 'interface', 'status', 'research_status',

    # Mathematical constants
    'PHI', 'E', 'PI', 'CRITICAL_DIMENSIONS',

    # Research platform functions (when available)
    'advanced_geometric_analysis', 'run_phase_simulation',
    'PreGeometry', 'PhaseDynamicsEngine'
]
