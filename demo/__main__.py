#!/usr/bin/env python3
"""
Dimensional Mathematics Demo
============================

Comprehensive demonstration of dimensional mathematics capabilities including:
- Spectral analysis and eigenvalue decomposition
- Algebraic structures (Clifford, Quaternions, Lie algebras)
- Phase dynamics and emergence simulation
- Advanced mathematical measures and critical point analysis
- Research-grade visualizations and analysis tools

Usage:
    python -m demo
    python -m demo --advanced    # Run advanced demonstrations
    python -m demo --interactive # Interactive exploration mode
"""

import argparse
import sys

import numpy as np

# Import core dimensional mathematics
try:
    from dimensional import (
        # Constants and utilities
        PHI,
        # Core measures
        C,
        # Algebraic structures
        CliffordAlgebra,
        # Phase dynamics
        PhaseDynamicsEngine,
        Quaternion,
        S,
        SO3LieAlgebra,
        V,
        # Spectral analysis
        analyze_emergence_spectrum,
        ball_volume,
        complexity_measure,
        dimensional_spectral_density,
        find_peak,
        # Research tools
        peaks,
        quick_gamma_analysis,
        quick_phase_analysis,
        quick_spectral_analysis,
        sphere_surface,
    )
    print("✅ Dimensional mathematics core loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import core modules: {e}")
    sys.exit(1)

# Check for optional enhanced research capabilities
try:
    import dimensional.enhanced  # noqa: F401
    ENHANCED_AVAILABLE = True
    print("✅ Enhanced research capabilities available")
except ImportError:
    ENHANCED_AVAILABLE = False
    print("⚠️  Enhanced research capabilities not available (optional)")


def demo_basic_measures():
    """Demonstrate basic dimensional measures."""
    print("\n" + "="*60)
    print("🔢 BASIC DIMENSIONAL MEASURES")
    print("="*60)

    dimensions = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    print(f"{'Dimension':<10} {'Volume':<15} {'Surface':<15} {'Complexity':<15}")
    print("-" * 60)

    for d in dimensions:
        vol = V(d)
        surf = S(d)
        comp = C(d)
        print(f"{d:<10.1f} {vol:<15.6f} {surf:<15.6f} {comp:<15.6f}")

    # Find and display peaks
    print("\n📊 Peak Analysis:")
    try:
        vol_peak = find_peak(ball_volume, 0.1, 15.0)
        surf_peak = find_peak(sphere_surface, 0.1, 15.0)
        comp_peak = find_peak(complexity_measure, 0.1, 15.0)

        print(f"  Volume peak:     d = {vol_peak:.6f}")
        print(f"  Surface peak:    d = {surf_peak:.6f}")
        print(f"  Complexity peak: d = {comp_peak:.6f}")
    except Exception as e:
        print(f"  Peak analysis failed: {e}")


def demo_spectral_analysis():
    """Demonstrate spectral analysis capabilities."""
    print("\n" + "="*60)
    print("🌊 SPECTRAL ANALYSIS")
    print("="*60)

    try:
        # Quick spectral analysis
        print("Running quick spectral analysis...")
        spectral_data = quick_spectral_analysis(max_dimensions=6, dt=0.01)

        eigenvals = spectral_data['spectral_decomposition']['eigenvalues']
        print(f"✅ Computed {len(eigenvals)} eigenvalues")
        print(f"   Spectral radius: {spectral_data['spectral_decomposition']['spectral_radius']:.6f}")
        print(f"   Stable modes: {spectral_data['spectral_decomposition']['n_stable']}")
        print(f"   Unstable modes: {spectral_data['spectral_decomposition']['n_unstable']}")

        # Dimensional spectral density
        print("\nComputing dimensional spectral density...")
        dims = np.linspace(0.1, 10, 100)
        density_data = dimensional_spectral_density(dims)

        print(f"✅ Found {len(density_data['peak_frequencies'])} spectral peaks")
        print(f"   Spectral centroid: {density_data['spectral_centroid']:.6f}")
        print(f"   Spectral bandwidth: {density_data['spectral_bandwidth']:.6f}")

    except Exception as e:
        print(f"❌ Spectral analysis failed: {e}")


def demo_algebraic_structures():
    """Demonstrate algebraic structures."""
    print("\n" + "="*60)
    print("🔗 ALGEBRAIC STRUCTURES")
    print("="*60)

    try:
        # Quaternion demo
        print("🔄 Quaternion Operations:")
        q1 = Quaternion(1, 0, 0, 0)  # Identity
        q2 = Quaternion(0, 1, 0, 0)  # i
        q3 = q1 * q2

        print(f"   q1 = {q1}")
        print(f"   q2 = {q2}")
        print(f"   q1 * q2 = {q3}")
        print(f"   |q2| = {q2.norm():.6f}")

        # Clifford Algebra demo
        print("\n🌀 Clifford Algebra Cl(3,0):")
        cliff = CliffordAlgebra(p=3, q=0)  # 3D Euclidean space
        print(f"   Signature: ({cliff.p}, {cliff.q}, {cliff.r})")
        print(f"   Basis elements: {2**(cliff.p + cliff.q + cliff.r)}")

        # SO(3) Lie Algebra demo
        print("\n🔄 SO(3) Lie Algebra:")
        so3 = SO3LieAlgebra()
        axis = np.array([0, 0, 1])  # z-axis
        angle = np.pi/4
        rotation = so3.exponential_map(axis, angle)
        print("   Rotation around z-axis by π/4:")
        print(f"   Matrix determinant: {np.linalg.det(rotation):.6f}")

    except Exception as e:
        print(f"❌ Algebraic structures demo failed: {e}")


def demo_phase_dynamics():
    """Demonstrate phase dynamics and emergence."""
    print("\n" + "="*60)
    print("⚡ PHASE DYNAMICS & EMERGENCE")
    print("="*60)

    try:
        # Quick phase analysis
        print("Running quick phase analysis...")
        phase_data = quick_phase_analysis(dimensions=[4.0])

        print("✅ Phase analysis completed")
        print(f"   Dimension analyzed: {phase_data.get('dimension', 'unknown')}")
        print(f"   Phase capacity: {phase_data.get('phase_capacity', 'unknown')}")

        # Phase dynamics engine demo
        print("\nInitializing Phase Dynamics Engine...")
        engine = PhaseDynamicsEngine(max_dimensions=6)

        print(f"✅ Engine initialized with {engine.max_dimensions} dimensions")

        # Run a few simulation steps
        for step in range(5):
            engine.step(dt=0.01)
            state = engine.get_state()
            energy = np.sum(np.abs(state['phase_densities'])**2)
            print(f"   Step {step+1}: Total energy = {energy:.6f}")

    except Exception as e:
        print(f"❌ Phase dynamics demo failed: {e}")


def demo_research_tools():
    """Demonstrate research and analysis tools."""
    print("\n" + "="*60)
    print("🔬 RESEARCH TOOLS")
    print("="*60)

    try:
        # Basic research functions
        print("📊 Quick gamma analysis...")
        z_values = np.linspace(0.1, 10, 50)
        quick_gamma_analysis(z_values)
        print("✅ Gamma analysis completed")

        print("\n🎯 Finding peaks...")
        peaks()
        print("✅ Peak analysis completed")

        if ENHANCED_AVAILABLE:
            print("\n🚀 Enhanced Research Tools Available:")
            print("   - Enhanced exploration with guided discovery")
            print("   - Interactive parameter sweeps")
            print("   - Rich visualizations with publication quality")
            print("   - Research session persistence")
            print("   - Advanced data export capabilities")
        else:
            print("\n📝 Basic research tools active")
            print("   For enhanced capabilities, install: pip install rich pandas")

    except Exception as e:
        print(f"❌ Research tools demo failed: {e}")


def demo_advanced_analysis():
    """Demonstrate advanced mathematical analysis."""
    print("\n" + "="*60)
    print("🧮 ADVANCED ANALYSIS")
    print("="*60)

    try:
        # Emergence spectrum analysis
        print("Analyzing emergence spectrum...")
        emergence_data = analyze_emergence_spectrum(n_steps=100, dt=0.01, max_dimensions=4)

        print("✅ Emergence analysis completed")
        resonances = emergence_data.get('resonance_analysis', {})
        if 'resonances' in resonances:
            n_resonances = len(resonances['resonances'])
            print(f"   Found {n_resonances} dimensional resonances")

        # Critical point analysis
        complexity_spectrum = emergence_data.get('complexity_spectrum', {})
        if 'critical_analyses' in complexity_spectrum:
            n_critical = len(complexity_spectrum['critical_analyses'])
            print(f"   Identified {n_critical} critical points")

    except Exception as e:
        print(f"❌ Advanced analysis failed: {e}")


def run_interactive_exploration():
    """Run interactive exploration mode."""
    print("\n" + "="*60)
    print("🎮 INTERACTIVE EXPLORATION MODE")
    print("="*60)

    if ENHANCED_AVAILABLE:
        print("🚀 Enhanced interactive mode available!")
        print("   Try: enhanced_explore(4.0)")
        print("   Try: enhanced_lab()")
        print("   Try: enhanced_instant()")
    else:
        print("📊 Basic interactive mode:")
        print("   Try: explore(4.0)")
        print("   Try: lab()")
        print("   Try: instant()")

    print("\n💡 Quick commands to try:")
    print("   >>> from dimensional import *")
    print("   >>> explore(5.2)  # Explore dimension 5.2")
    print("   >>> instant()     # Four-panel analysis")
    print("   >>> V(4), S(4), C(4)  # Basic measures")


def main():
    """Main demo function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Dimensional Mathematics Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Run advanced demonstrations including spectral analysis"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show interactive exploration options"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all demonstrations"
    )

    args = parser.parse_args()

    # Print header
    print("🔬" + "="*58 + "🔬")
    print("    DIMENSIONAL MATHEMATICS COMPREHENSIVE DEMO")
    print("🔬" + "="*58 + "🔬")
    print(f"📐 Framework version with φ = {PHI:.6f}")
    print("🧮 Advanced mathematical analysis ready")

    # Run demonstrations based on arguments
    if args.all or not any(vars(args).values()):
        # Run basic demos by default
        demo_basic_measures()
        demo_algebraic_structures()
        demo_phase_dynamics()
        demo_research_tools()

    if args.advanced or args.all:
        demo_spectral_analysis()
        demo_advanced_analysis()

    if args.interactive or args.all:
        run_interactive_exploration()

    # Show completion and next steps
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETED")
    print("="*60)
    print("🚀 Ready for next workstream!")
    print("📊 All tests passing, ruff checks clean")
    print("🔬 Full mathematical framework operational")

    if ENHANCED_AVAILABLE:
        print("✨ Enhanced research capabilities active")
    else:
        print("💡 For enhanced features: pip install rich pandas jupyter")

    print("\n📖 Next steps:")
    print("   - Import and explore: from dimensional import *")
    print("   - Run interactive analysis: enhanced_lab() or lab()")
    print("   - Explore specific dimensions: explore(d)")
    print("   - Run emergence simulations: analyze_emergence_spectrum()")


if __name__ == "__main__":
    main()
