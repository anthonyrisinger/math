#!/usr/bin/env python3
"""
Analysis Package Test Suite
===========================

Comprehensive tests for the dimensional analysis package.
Tests all modules: geometric_measures, emergence_framework, reality_modeling.
"""

import sys
import traceback

import numpy as np

# Import constants from geometric_measures
from .geometric_measures import PHI


def test_geometric_measures():
    """Test geometric measures module."""
    print("🧮 Testing geometric measures...")

    try:
        from .geometric_measures import (
            PHI,
            PI,
            DimensionalAnalyzer,
            S,
            V,
            find_peaks,
        )

        # Test basic measures
        assert abs(V(0) - 1.0) < 1e-10, "V(0) should be 1"
        assert abs(V(1) - 2.0) < 1e-10, "V(1) should be 2"
        assert abs(V(2) - PI) < 1e-10, "V(2) should be π"

        # Test sphere surface
        assert abs(S(2) - 2 * PI) < 1e-10, "S(2) should be 2π"

        # Test array inputs
        dims = np.array([0, 1, 2, 3])
        volumes = V(dims)
        assert len(volumes) == 4, "Array input should return array output"

        # Test critical dimensions
        peaks = find_peaks()
        assert "volume_peak" in peaks, "Should find volume peak"
        assert "complexity_peak" in peaks, "Should find complexity peak"

        # Test analyzer
        analyzer = DimensionalAnalyzer()
        analysis = analyzer.analyze_dimension(PHI)
        assert "ball_volume" in analysis, "Analysis should contain ball_volume"
        assert "complexity" in analysis, "Analysis should contain complexity"

        print("  ✅ Geometric measures tests passed")
        return True

    except Exception as e:
        print(f"  ❌ Geometric measures test failed: {e}")
        traceback.print_exc()
        return False


def test_emergence_framework():
    """Test emergence framework module."""
    print("🌱 Testing emergence framework...")

    try:
        from .emergence_framework import (
            EmergenceFramework,
            analyze_emergence,
        )

        # Test framework initialization
        framework = EmergenceFramework()
        assert framework.dimension == 0.0, "Should start at dimension 0"
        assert framework.time == 0.0, "Should start at time 0"
        assert len(framework.phase_density) > 0, "Should have phase densities"

        # Test potential calculation
        framework.dimensional_potential(0)
        potential_phi = framework.dimensional_potential(PHI)
        assert potential_phi > 0, "Potential at φ should be positive"

        # Test phase evolution
        framework.phase_density.copy()
        framework.evolve_phase(dt=0.001)
        assert framework.time > 0, "Time should advance"

        # Test emergence simulation (short)
        results = framework.run_emergence_simulation(steps=10, dt=0.01)
        assert "final_dimension" in results, "Should return final dimension"
        assert "emerged_dimensions" in results, "Should track emerged dimensions"

        # Test convenience functions
        analysis = analyze_emergence(PHI)
        assert "potential" in analysis, "Analysis should contain potential"
        assert "evolution_rate" in analysis, "Analysis should contain evolution rate"

        print("  ✅ Emergence framework tests passed")
        return True

    except Exception as e:
        print(f"  ❌ Emergence framework test failed: {e}")
        traceback.print_exc()
        return False


def test_reality_modeling():
    """Test reality modeling module."""
    print("🌌 Testing reality modeling...")

    try:
        from .reality_modeling import (
            RealityModeler,
            consciousness_at,
            life_probability_at,
            reality_stability_at,
        )

        # Test modeler initialization
        modeler = RealityModeler()
        assert hasattr(modeler, "physical_constants"), "Should have physical constants"
        assert hasattr(
            modeler, "consciousness_threshold"
        ), "Should have consciousness threshold"

        # Test reality measures
        stability_3d = modeler.reality_stability(3.0)
        assert stability_3d > 0, "3D reality should be stable"

        consciousness_phi = modeler.consciousness_emergence(PHI)
        assert consciousness_phi > 0, "Consciousness should emerge at φ"

        life_3d = modeler.life_probability(3.0)
        assert life_3d > 0, "Life should be possible in 3D"

        # Test temporal flow
        time_flow = modeler.temporal_flow_rate(3.0)
        assert time_flow > 0, "Time should flow in 3D"

        # Test reality map generation (small)
        reality_map = modeler.generate_reality_map(d_range=(1, 4), resolution=10)
        assert "dimensions" in reality_map, "Should contain dimensions"
        assert "critical_zones" in reality_map, "Should contain critical zones"

        # Test convenience functions
        consciousness_level = consciousness_at(PHI)
        life_prob = life_probability_at(3.0)
        stability = reality_stability_at(3.0)

        assert consciousness_level >= 0, "Consciousness should be non-negative"
        assert life_prob >= 0, "Life probability should be non-negative"
        assert stability > 0, "Stability should be positive"

        print("  ✅ Reality modeling tests passed")
        return True

    except Exception as e:
        print(f"  ❌ Reality modeling test failed: {e}")
        traceback.print_exc()
        return False


def test_package_integration():
    """Test integration between modules."""
    print("🔗 Testing package integration...")

    try:
        from .emergence_framework import EmergenceFramework
        from .geometric_measures import PHI, C
        from .reality_modeling import RealityModeler

        # Test that modules work together
        framework = EmergenceFramework()
        modeler = RealityModeler()

        # Test dimension consistency
        test_d = PHI
        geometric_complexity = C(test_d)
        emergence_potential = framework.dimensional_potential(test_d)
        reality_stability = modeler.reality_stability(test_d)

        # All should be related (complexity drives emergence and stability)
        assert geometric_complexity > 0, "Geometric complexity should be positive"
        assert emergence_potential > 0, "Emergence potential should be positive"
        assert reality_stability > 0, "Reality stability should be positive"

        # Test that measures are consistent across modules
        # The emergence potential should be based on geometric complexity
        potential_from_framework = framework.dimensional_potential(test_d)
        geometric_complexity_direct = geometric_complexity

        # Both should be positive and related
        assert potential_from_framework > 0, "Framework potential should be positive"
        assert geometric_complexity_direct > 0, "Direct complexity should be positive"

        # They should be equal since dimensional_potential uses complexity_measure
        assert (
            abs(potential_from_framework - geometric_complexity_direct) < 1e-10
        ), "Potentials should match complexity"

        print("  ✅ Package integration tests passed")
        return True

    except Exception as e:
        print(f"  ❌ Package integration test failed: {e}")
        traceback.print_exc()
        return False


def test_mathematical_consistency():
    """Test mathematical consistency and properties."""
    print("🔢 Testing mathematical consistency...")

    try:
        from .geometric_measures import PI, C, S, V

        # Test mathematical relationships

        # Volume should be positive and decreasing after peak
        v_peak = 5.257  # Approximate volume peak
        assert V(v_peak - 0.1) < V(v_peak), "Volume should peak around 5.257"
        assert V(v_peak + 0.1) < V(v_peak), "Volume should decrease after peak"

        # Surface should equal d * Volume
        test_d = 3.5
        assert abs(S(test_d) - test_d * V(test_d)) < 1e-10, "S(d) should equal d * V(d)"

        # Complexity should equal V * S
        assert (
            abs(C(test_d) - V(test_d) * S(test_d)) < 1e-10
        ), "C(d) should equal V(d) * S(d)"

        # Test special values
        assert abs(V(2) - PI) < 1e-10, "V(2) should equal π"
        assert abs(S(1) - 2) < 1e-10, "S(1) should equal 2"

        # Test continuity (no sudden jumps)
        d_test = 2.5
        epsilon = 1e-6
        v1 = V(d_test - epsilon)
        v2 = V(d_test + epsilon)
        relative_change = abs(v2 - v1) / v1
        assert relative_change < 1e-3, "Volume should be continuous"

        print("  ✅ Mathematical consistency tests passed")
        return True

    except Exception as e:
        print(f"  ❌ Mathematical consistency test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run complete test suite for analysis package."""
    print("ANALYSIS PACKAGE TEST SUITE")
    print("=" * 60)

    tests = [
        test_geometric_measures,
        test_emergence_framework,
        test_reality_modeling,
        test_package_integration,
        test_mathematical_consistency,
    ]

    results = []
    for test in tests:
        try:
            success = test()
            results.append(success)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Analysis package is ready.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please fix issues.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
