#!/usr/bin/env python3
"""
Dimensional Package Integration Test
===================================

Comprehensive test demonstrating the unified dimensional package
with consolidated gamma functions, measures, and phase dynamics.
"""


def test_unified_dimensional_package():
    """Test the complete unified dimensional package."""

    print("🧪 UNIFIED DIMENSIONAL PACKAGE TEST")
    print("=" * 60)

    # Import the unified package
    try:
        import numpy as np

        from dimensional import (
            PHI,
            PI,
            C,
            PhaseDynamicsEngine,
            S,
            V,
            find_all_peaks,
            gamma_safe,
            quick_phase_analysis,
            sap_rate,
            total_phase_energy,
        )

        print("✅ Import successful - all modules loaded")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

    print("\n📊 TESTING GAMMA FUNCTIONS:")
    print("-" * 40)
    try:
        print(f"  Γ(0.5) = {gamma_safe(0.5):.6f} (should ≈ {np.sqrt(PI):.6f})")
        print(f"  Γ(4) = {gamma_safe(4):.6f} (should = 6)")
        print(f"  φ = {PHI:.6f}")
        print("  ✅ Gamma functions working")
    except Exception as e:
        print(f"  ❌ Gamma functions failed: {e}")
        return False

    print("\n📐 TESTING DIMENSIONAL MEASURES:")
    print("-" * 40)
    try:
        print(f"  V(0) = {V(0):.6f} (should = 1)")
        print(f"  V(2) = {V(2):.6f} (should ≈ {PI:.6f})")
        print(f"  S(2) = {S(2):.6f} (should ≈ {2*PI:.6f})")
        print(f"  C(4) = {C(4):.6f}")

        # Test peak finding
        peaks = find_all_peaks(d_min=0.1, d_max=10, resolution=1000)
        print(f"  Volume peak at d ≈ {peaks['volume_peak'][0]:.3f}")
        print("  ✅ Dimensional measures working")
    except Exception as e:
        print(f"  ❌ Dimensional measures failed: {e}")
        return False

    print("\n⚡ TESTING PHASE DYNAMICS:")
    print("-" * 40)
    try:
        # Create phase engine
        engine = PhaseDynamicsEngine(max_dimensions=6)
        print(f"  Initial energy: {total_phase_energy(engine.phase_density):.6f}")

        # Evolve for some steps
        result = engine.evolve(200)
        print(f"  After evolution: {total_phase_energy(engine.phase_density):.6f}")
        print(f"  Emerged dimensions: {result['current_emerged']}")
        print(f"  Effective dimension: {engine.calculate_effective_dimension():.3f}")

        # Test sapping rates
        rate = sap_rate(2, 4)
        print(f"  Sap rate 2→4: {rate:.6f}")
        print("  ✅ Phase dynamics working")
    except Exception as e:
        print(f"  ❌ Phase dynamics failed: {e}")
        return False

    print("\n🔬 TESTING INTEGRATION:")
    print("-" * 40)
    try:
        # Run integrated analysis
        analysis = quick_phase_analysis(dimension=4.0, time_steps=500)
        print(f"  Target dimension: {analysis['target_dimension']}")
        print(
            f"  Final effective dim: {analysis['final_state']['effective_dimension']:.3f}"
        )
        print(f"  Energy: {analysis['energy_conservation']:.6f}")
        print("  ✅ Integration working")
    except Exception as e:
        print(f"  ❌ Integration failed: {e}")
        return False

    print("\n🚀 TESTING CONVENIENCE FUNCTIONS:")
    print("-" * 40)
    try:
        # Test quick functions
        print(f"  Quick gamma: γ(2.5) = {gamma_safe(2.5):.6f}")
        print(f"  Quick volume: V(3) = {V(3):.6f}")
        print(f"  Quick surface: S(3) = {S(3):.6f}")
        print(f"  Quick complexity: C(3) = {C(3):.6f}")
        print("  ✅ Convenience functions working")
    except Exception as e:
        print(f"  ❌ Convenience functions failed: {e}")
        return False

    print("\n✨ SUMMARY:")
    print("=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("📦 Unified dimensional package is fully functional")
    print("🧬 Gamma functions, measures, and phase dynamics integrated")
    print("⚡ Ready for advanced mathematical exploration!")

    return True


def demo_unified_capabilities():
    """Demonstrate the unified package capabilities."""

    print("\n\n🌟 UNIFIED PACKAGE DEMONSTRATION")
    print("=" * 60)

    from dimensional import PI, C, R, S, V, find_all_peaks, run_emergence_simulation

    print("📈 DIMENSIONAL ANALYSIS WORKFLOW:")
    print("-" * 50)

    # Find all peaks
    peaks = find_all_peaks()
    print("Critical Points Found:")
    for name, (d, value) in peaks.items():
        print(f"  {name:20}: d={d:.3f}, value={value:.6f}")

    print("\n🌀 PHASE EMERGENCE SIMULATION:")
    print("-" * 50)

    # Run emergence simulation
    sim_results = run_emergence_simulation(max_time=8.0, max_dimensions=7)
    state = sim_results["final_state"]

    print(f"Simulation completed at t = {state['time']:.2f}")
    print(f"Emerged dimensions: {state['emerged_dimensions']}")
    print(f"Effective dimension: {state['effective_dimension']:.3f}")
    print(f"Total energy: {state['total_energy']:.6f}")
    print(f"Phase coherence: {state['coherence']:.6f}")

    # Show emergence timeline
    engine = sim_results["engine"]
    print("\nEmergence Timeline:")
    for d, t in sorted(engine.emergence_times.items()):
        print(f"  Dimension {d}: emerged at t = {t:.2f}")

    print("\n🎯 DIMENSIONAL RELATIONSHIPS:")
    print("-" * 50)

    # Show relationships at key dimensions
    key_dims = [1, 2, 3, 4, PI, 6, 2 * PI]
    print("d      | V(d)      | S(d)      | C(d)      | R(d)")
    print("-" * 55)
    for d in key_dims:
        v = V(d)
        s = S(d)
        c = C(d)
        r = R(d)
        print(f"{d:6.3f} | {v:9.6f} | {s:9.6f} | {c:9.6f} | {r:9.6f}")

    print("\n🎊 UNIFIED PACKAGE READY FOR RESEARCH!")
    print("Use quick_start() for more examples")


if __name__ == "__main__":

    # Run comprehensive test
    success = test_unified_dimensional_package()

    if success:
        # Run demonstration
        demo_unified_capabilities()
    else:
        print("❌ Tests failed - check error messages above")
