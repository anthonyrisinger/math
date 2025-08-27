#!/usr/bin/env python3
"""
Core Library Test Suite
=======================

Comprehensive tests for the core mathematical library.
Tests mathematical correctness, numerical stability, edge cases,
and usability of the API.
"""

import os
import sys

import numpy as np
import pytest

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

# Test imports - this will immediately reveal import issues
try:
    # Note: core module is being phased out during consolidation
    # import core  # Temporarily commented out during consolidation
    from dimensional.mathematics import (
        CRITICAL_DIMENSIONS,
        PHI,
        PI,
        PSI,
        VARPI,
        PhaseDynamicsEngine,
        ball_volume,
        complexity_measure,
        create_3d_figure,
        gamma_safe,
        morphic_polynomial_roots,
        sap_rate,
        setup_3d_axis,
        sphere_surface,
    )

    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except Exception as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


class TestImports:
    """Test that all imports work correctly."""

    def test_basic_import(self):
        """Test basic core import."""
        assert IMPORT_SUCCESS, f"Failed to import core: {IMPORT_ERROR}"

    def test_constants_available(self):
        """Test that fundamental constants are available."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # Check constants exist and have reasonable values
        assert 1.6 < PHI < 1.7, f"PHI = {PHI} (expected ~1.618)"
        assert 3.1 < PI < 3.2, f"PI = {PI} (expected ~3.14159)"
        assert 0.6 < PSI < 0.7, f"PSI = {PSI} (expected ~0.618)"
        assert 1.3 < VARPI < 1.4, f"VARPI = {VARPI} (expected ~1.311)"

    def test_critical_dimensions(self):
        """Test critical dimensions dictionary."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        assert isinstance(CRITICAL_DIMENSIONS, dict)
        assert "pi_boundary" in CRITICAL_DIMENSIONS
        assert "complexity_peak" in CRITICAL_DIMENSIONS
        assert "leech_limit" in CRITICAL_DIMENSIONS


class TestGammaFunctions:
    """Test gamma function family."""

    def test_known_values(self):
        """Test gamma function against known values."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # Γ(1) = 1
        assert abs(gamma_safe(1.0) - 1.0) < 1e-10

        # Γ(2) = 1! = 1
        assert abs(gamma_safe(2.0) - 1.0) < 1e-10

        # Γ(3) = 2! = 2
        assert abs(gamma_safe(3.0) - 2.0) < 1e-10

        # Γ(1/2) = √π
        assert abs(gamma_safe(0.5) - np.sqrt(PI)) < 1e-10

    def test_edge_cases(self):
        """Test gamma function edge cases."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # Γ(0) should be infinite
        assert np.isinf(gamma_safe(0.0))

        # Γ(-1) should be infinite (pole)
        assert np.isinf(gamma_safe(-1.0))

        # Γ(-2) should be infinite (pole)
        assert np.isinf(gamma_safe(-2.0))

    def test_array_input(self):
        """Test gamma function with array inputs."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        values = np.array([1.0, 2.0, 3.0, 0.5])
        results = gamma_safe(values)

        assert len(results) == 4
        assert abs(results[0] - 1.0) < 1e-10  # Γ(1)
        assert abs(results[1] - 1.0) < 1e-10  # Γ(2)
        assert abs(results[2] - 2.0) < 1e-10  # Γ(3)
        assert abs(results[3] - np.sqrt(PI)) < 1e-10  # Γ(1/2)


class TestDimensionalMeasures:
    """Test dimensional measures."""

    def test_ball_volume_known_values(self):
        """Test ball volume against known values."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # V_0 = 1 (point)
        assert abs(ball_volume(0) - 1.0) < 1e-10

        # V_1 = 2 (line segment)
        assert abs(ball_volume(1) - 2.0) < 1e-10

        # V_2 = π (disk)
        assert abs(ball_volume(2) - PI) < 1e-10

        # V_3 = 4π/3 (ball)
        assert abs(ball_volume(3) - 4 * PI / 3) < 1e-10

    def test_sphere_surface_known_values(self):
        """Test sphere surface against known values."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # S_1 = 2 (two points)
        assert abs(sphere_surface(1) - 2.0) < 1e-10

        # S_2 = 2π (circle)
        assert abs(sphere_surface(2) - 2 * PI) < 1e-10

        # S_3 = 4π (sphere)
        assert abs(sphere_surface(3) - 4 * PI) < 1e-10

    def test_complexity_measure(self):
        """Test complexity measure C = V × S."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # Test that C(d) = V(d) × S(d)
        for d in [1, 2, 3, 4, 5]:
            v = ball_volume(d)
            s = sphere_surface(d)
            c = complexity_measure(d)
            assert abs(c - v * s) < 1e-10, f"C({d}) ≠ V({d}) × S({d})"

    def test_fractional_dimensions(self):
        """Test measures work for fractional dimensions."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # Should work for fractional dimensions
        d = 1.5
        v = ball_volume(d)
        s = sphere_surface(d)
        c = complexity_measure(d)

        assert np.isfinite(v) and v > 0
        assert np.isfinite(s) and s > 0
        assert np.isfinite(c) and c > 0

    def test_peak_finding(self):
        """Test that peaks are found correctly."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        from dimensional.mathematics.functions import find_all_peaks

        peaks = find_all_peaks()

        # Volume should peak around d ≈ 5.26
        vol_peak_d, vol_peak_val = peaks["volume_peak"]
        assert 5.0 < vol_peak_d < 6.0, f"Volume peak at d={vol_peak_d} (expected ~5.26)"

        # Surface should peak around d ≈ 7.26
        surf_peak_d, surf_peak_val = peaks["surface_peak"]
        assert (
            7.0 < surf_peak_d < 8.0
        ), f"Surface peak at d={surf_peak_d} (expected ~7.26)"

        # Complexity should peak around d ≈ 6
        comp_peak_d, comp_peak_val = peaks["complexity_peak"]
        assert (
            5.5 < comp_peak_d < 6.5
        ), f"Complexity peak at d={comp_peak_d} (expected ~6)"


class TestPhaseDynamics:
    """Test phase dynamics."""

    def test_sap_rate_basic(self):
        """Test basic sapping rate calculation."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # Create phase density array
        phase_density = np.array([1.0, 0.5, 0.0, 0.0], dtype=complex)

        # Rate from 0 to 1
        rate = sap_rate(0, 1, phase_density)
        assert rate >= 0, "Sapping rate should be non-negative"

        # No reverse flow
        reverse_rate = sap_rate(1, 0, phase_density)
        assert reverse_rate == 0, "No reverse sapping should occur"

    def test_phase_engine_creation(self):
        """Test phase dynamics engine creation."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        engine = PhaseDynamicsEngine(max_dimensions=6)

        # Check initial state
        state = engine.get_state()
        assert state["time"] == 0.0
        assert 0 in state["emerged_dimensions"]
        assert state["total_energy"] > 0  # Should have energy at void

    def test_phase_evolution(self):
        """Test phase evolution step."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        engine = PhaseDynamicsEngine(max_dimensions=4)
        initial_energy = engine.get_state()["total_energy"]

        # Run a few steps
        for _ in range(10):
            engine.step(0.1)

        final_state = engine.get_state()

        # Time should have advanced
        assert final_state["time"] > 0

        # Energy should be conserved (approximately)
        final_energy = final_state["total_energy"]
        assert abs(final_energy - initial_energy) < 0.1, "Energy not conserved"

    def test_energy_injection(self):
        """Test energy injection mechanism."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        engine = PhaseDynamicsEngine(max_dimensions=4)
        initial_energy = engine.get_state()["total_energy"]

        # Inject energy into dimension 1
        engine.inject(1, 0.5)

        # Total energy should increase
        new_energy = engine.get_state()["total_energy"]
        assert (
            new_energy > initial_energy
        ), "Energy injection should increase total energy"


class TestMorphicMathematics:
    """Test morphic mathematics."""

    def test_golden_ratio_properties(self):
        """Test golden ratio mathematical properties."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        from dimensional.mathematics.functions import golden_ratio_properties

        props = golden_ratio_properties()

        # φ² = φ + 1
        assert props["phi_squared_equals_phi_plus_one"], "φ² ≠ φ + 1"

        # ψ² = 1 - ψ
        assert props["psi_squared_equals_one_minus_psi"], "ψ² ≠ 1 - ψ"

        # φψ = 1
        assert props["phi_times_psi_equals_one"], "φψ ≠ 1"

        # φ - ψ = 1 (correct mathematical relationship)
        assert props["phi_minus_psi_equals_one"], "φ - ψ ≠ 1"

        # φ + ψ = √5 (correct mathematical relationship)
        assert props["phi_plus_psi_equals_sqrt5"], "φ + ψ ≠ √5"

    def test_polynomial_roots(self):
        """Test morphic polynomial root finding."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # Test at k = 2 (perfect circle case)
        roots = morphic_polynomial_roots(2.0, "shifted")

        # Should have τ = 1 as a roo
        assert any(
            abs(r - 1.0) < 1e-10 for r in roots
        ), "τ = 1 should be a root at k = 2"

    def test_discriminant(self):
        """Test discriminant calculation."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        from dimensional.mathematics.functions import discriminant, k_discriminant_zero

        # At critical k, discriminant should be zero
        k_critical = k_discriminant_zero("shifted")
        disc = discriminant(k_critical, "shifted")
        assert abs(disc) < 1e-8, f"Discriminant should be zero at k={k_critical}"


class TestVisualization:
    """Test visualization functions."""

    def test_view_preserving_3d_class(self):
        """Test ViewPreserving3D class functionality."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        from visualization.view_3d import ViewPreserving3D

        # Create instance
        view_3d = ViewPreserving3D()

        # Test initial state
        assert hasattr(view_3d, 'current_elev')
        assert hasattr(view_3d, 'current_azim')
        assert hasattr(view_3d, 'view_changed')
        assert view_3d.view_changed == False

        # Test view saving/restoring (mock object)
        class MockAxis:
            def __init__(self):
                self.elev = 25.0
                self.azim = 45.0

            def view_init(self, elev=None, azim=None):
                if elev is not None:
                    self.elev = elev
                if azim is not None:
                    self.azim = azim

        mock_ax = MockAxis()

        # Test save view
        view_3d.save_view(mock_ax)
        assert view_3d.current_elev == 25.0
        assert view_3d.current_azim == 45.0

        # Test restore view
        mock_ax.elev = 10.0  # Change view
        mock_ax.azim = 20.0
        view_3d.restore_view(mock_ax)
        assert mock_ax.elev == 25.0  # Should be restored
        assert mock_ax.azim == 45.0

    def test_figure_creation(self):
        """Test figure creation."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        try:
            # Test deprecated function now returns None
            fig, ax = create_3d_figure(figsize=(6, 4))

            # MATPLOTLIB ELIMINATED - function now returns None
            assert fig is None
            assert ax is None
            print("✅ Deprecated 3D figure creation handled correctly")

        except ImportError:
            pytest.skip("Matplotlib not available")


class TestAPIUsability:
    """Test API usability and common usage patterns."""

    def test_star_import(self):
        """Test that 'from dimensional.mathematics import *' works."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # This should work without issues
        exec("from dimensional.mathematics import *")

    def test_common_workflow(self):
        """Test a common mathematical workflow."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # Calculate dimensional measures for a range
        dimensions = np.linspace(0.1, 10, 50)

        volumes = [ball_volume(d) for d in dimensions]
        surfaces = [sphere_surface(d) for d in dimensions]
        complexities = [complexity_measure(d) for d in dimensions]

        # All should be finite and positive
        assert all(np.isfinite(v) and v > 0 for v in volumes)
        assert all(np.isfinite(s) and s > 0 for s in surfaces)
        assert all(np.isfinite(c) and c > 0 for c in complexities)

    def test_phase_simulation_workflow(self):
        """Test phase dynamics simulation workflow."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # Create engine
        engine = PhaseDynamicsEngine(max_dimensions=6)

        # Run simulation
        dt = 0.05
        n_steps = 20

        for step in range(n_steps):
            engine.step(dt)

            # Inject energy occasionally
            if step % 10 == 0:
                engine.inject(0, 0.1)

            state = engine.get_state()

            # State should always be valid
            assert state["time"] >= 0
            assert state["total_energy"] >= 0
            assert 0 <= state["coherence"] <= 1

    def test_mathematical_consistency(self):
        """Test mathematical consistency across modules."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # Phase capacity should equal ball volume
        for d in [1, 2, 3, 4, 5]:
            from dimensional.mathematics.functions import phase_capacity

            capacity = phase_capacity(d)
            volume = ball_volume(d)
            assert (
                abs(capacity - volume) < 1e-10
            ), f"Phase capacity ≠ ball volume at d={d}"


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_large_dimensions(self):
        """Test behavior with large dimensions."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # Should handle moderately large dimensions
        d_large = 50.0

        v = ball_volume(d_large)
        s = sphere_surface(d_large)

        # Should be finite (possibly very small)
        assert np.isfinite(v), f"Ball volume not finite at d={d_large}"
        assert np.isfinite(s), f"Sphere surface not finite at d={d_large}"

    def test_very_small_values(self):
        """Test behavior with very small dimensions."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        d_small = 1e-6

        v = ball_volume(d_small)
        s = sphere_surface(d_small)

        assert np.isfinite(v) and v > 0
        assert np.isfinite(s) and s > 0

    def test_array_operations(self):
        """Test operations with numpy arrays."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        # Should handle array inputs smoothly
        dimensions = np.array([0, 1, 2, 3, 4, 5])

        volumes = ball_volume(dimensions)
        surfaces = sphere_surface(dimensions)

        assert len(volumes) == len(dimensions)
        assert len(surfaces) == len(dimensions)
        assert all(np.isfinite(volumes))
        assert all(np.isfinite(surfaces))


def test_library_verification():
    """Test the built-in library verification."""
    if not IMPORT_SUCCESS:
        pytest.skip("Import failed")

    # Test verification functions
    from dimensional.mathematics.validation import validate_mathematical_properties
    verification = validate_mathematical_properties()

    # All tests should pass
    assert verification, "Built-in verification failed"


def run_performance_test():
    """Run performance tests (not part of main test suite)."""
    if not IMPORT_SUCCESS:
        print("Cannot run performance tests - import failed")
        return

    import time

    print("PERFORMANCE TESTS")
    print("=" * 40)

    # Test ball volume calculation speed
    dimensions = np.linspace(0.1, 10, 1000)

    start_time = time.time()
    [ball_volume(d) for d in dimensions]
    end_time = time.time()

    print(f"1000 ball volume calculations: {end_time - start_time:.4f}s")

    # Test phase dynamics step
    engine = PhaseDynamicsEngine(max_dimensions=10)

    start_time = time.time()
    for _ in range(100):
        engine.step(0.01)
    end_time = time.time()

    print(f"100 phase dynamics steps: {end_time - start_time:.4f}s")


if __name__ == "__main__":
    # Run basic tests
    print("CORE LIBRARY TEST RESULTS")
    print("=" * 50)

    if not IMPORT_SUCCESS:
        print(f"❌ IMPORT FAILED: {IMPORT_ERROR}")
        sys.exit(1)

    # Run key tests manually
    try:
        # Test constants
        test_const = TestImports()
        test_const.test_basic_import()
        test_const.test_constants_available()
        print("✅ Constants and imports work")

        # Test gamma functions
        test_gamma = TestGammaFunctions()
        test_gamma.test_known_values()
        print("✅ Gamma functions work")

        # Test dimensional measures
        test_measures = TestDimensionalMeasures()
        test_measures.test_ball_volume_known_values()
        test_measures.test_sphere_surface_known_values()
        print("✅ Dimensional measures work")

        # Test phase dynamics
        test_phase = TestPhaseDynamics()
        test_phase.test_phase_engine_creation()
        print("✅ Phase dynamics work")

        # Test morphic mathematics
        test_morphic = TestMorphicMathematics()
        test_morphic.test_golden_ratio_properties()
        print("✅ Morphic mathematics work")

        # Test verification
        test_library_verification()
        print("✅ Built-in verification passes")

        print("\n🎉 ALL BASIC TESTS PASSED")

        # Show library info
        print("\n" + "=" * 50)
        print("DIMENSIONAL MATHEMATICS LIBRARY")
        print("=" * 50)
        print(f"π (pi): {PI:.6f}")
        print(f"φ (golden ratio): {PHI:.6f}")
        print("Library verification complete ✅")

        # Performance tes
        print("\n" + "=" * 50)
        run_performance_test()

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# ADDITIONAL TESTS CONSOLIDATED FROM UNIFIED GAMMA MODULE
# =============================================================================

class TestQuickTools:
    """Test quick exploration and calculation tools."""

    def test_dimensional_functions(self):
        """Test dimensional measure functions v, s, c, r, ρ."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        try:
            from dimensional.gamma import v, s, c, r, ρ

            # Test known dimensions
            for d in [1, 2, 3, 4]:
                # All functions should return positive finite values
                volume = v(d)
                surface = s(d)
                complexity = c(d)
                ratio = r(d)
                density = ρ(d)

                assert np.isfinite(volume) and volume > 0
                assert np.isfinite(surface) and surface > 0
                assert np.isfinite(complexity) and complexity > 0
                assert np.isfinite(ratio) and ratio > 0
                assert np.isfinite(density) and density > 0

                # Test relationships
                assert abs(complexity - volume * surface) < 1e-10
                assert abs(ratio - surface / volume) < 1e-10
                assert abs(density - 1 / volume) < 1e-10
        except ImportError:
            pytest.skip("Quick tools not available")

    def test_peak_finders(self):
        """Test peak finding functions."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        try:
            from dimensional.gamma import v_peak, s_peak, c_peak

            # Volume peak should be around 5.26
            v_peak_d = v_peak()
            assert 5.0 < v_peak_d < 6.0

            # Surface peak should be around 7.26
            s_peak_d = s_peak()
            assert 7.0 < s_peak_d < 8.0

            # Complexity peak should be around 6.0
            c_peak_d = c_peak()
            assert 5.5 < c_peak_d < 6.5
        except ImportError:
            pytest.skip("Peak finders not available")

    def test_gamma_shortcuts(self):
        """Test gamma shortcut functions."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        try:
            from dimensional.gamma import γ, abs_γ

            # γ should be same as gamma_safe
            assert abs(γ(1.0) - gamma_safe(1.0)) < 1e-10
            assert abs(γ(2.5) - gamma_safe(2.5)) < 1e-10

            # abs_γ should be absolute value
            assert abs(abs_γ(1.0) - abs(gamma_safe(1.0))) < 1e-10
        except ImportError:
            pytest.skip("Gamma shortcuts not available")


class TestVisualizationCompat:
    """Test visualization functions for compatibility."""

    def test_qplot_exists(self):
        """Test qplot function exists and returns data."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        try:
            from dimensional.gamma import qplot, v, s, c

            # Modern implementation returns plot data
            result = qplot(v, s, c, labels=["Volume", "Surface", "Complexity"])
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("Visualization tools not available")

    def test_instant_exists(self):
        """Test instant visualization function."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        try:
            from dimensional.gamma import instant

            result = instant()
            assert isinstance(result, dict)
            assert "panels" in resul
        except ImportError:
            pytest.skip("Instant visualization not available")

    def test_explore_function(self):
        """Test explore function."""
        if not IMPORT_SUCCESS:
            pytest.skip("Import failed")

        try:
            from dimensional.gamma import explore

            result = explore(4.0)
            assert isinstance(result, dict)
            assert "dimension" in resul
            assert result["dimension"] == 4.0
        except ImportError:
            pytest.skip("Explore function not available")
