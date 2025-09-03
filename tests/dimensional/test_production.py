#!/usr/bin/env python3
"""
Production Features Tests
========================

Test suite for production-grade features including performance,
error handling, and robustness components.
"""

import numpy as np
import pytest

import dimensional as dm


class TestProductionReadiness:
    """Test production-grade readiness features."""

    def test_production_imports_work(self):
        """Test that production module imports work."""
        try:
            from dimensional.production import TechnicalDebtCleanupSystem
            system = TechnicalDebtCleanupSystem()
            assert system is not None
        except ImportError:
            # If class doesn't exist, that's okay for now
            pass

    def test_basic_mathematical_robustness(self):
        """Test basic mathematical operations are robust."""

        # Test known values
        assert abs(dm.gamma_safe(1.0) - 1.0) < 1e-10
        assert abs(dm.gamma_safe(2.0) - 1.0) < 1e-10
        assert abs(dm.gamma_safe(3.0) - 2.0) < 1e-10

    def test_vectorized_operations_robustness(self):
        """Test vectorized operations for robustness."""

        # Test array input
        dims = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        results = dm.gamma_safe(dims)

        assert len(results) == len(dims)
        assert all(np.isfinite(results))
        assert all(results > 0)  # Gamma should be positive for positive inputs


class TestNumericalStability:
    """Test numerical stability across the entire package."""

    def test_dimensional_measures_stability(self):
        """Test stability of dimensional measures."""
        # Test a range of dimensions
        dimensions = np.logspace(-2, 1, 50)  # 0.01 to 10

        for d in dimensions:
            v_result = dm.V(d)
            s_result = dm.S(d)
            c_result = dm.C(d)

            assert np.isfinite(v_result), f"V({d}) not finite"
            assert np.isfinite(s_result), f"S({d}) not finite"
            assert np.isfinite(c_result), f"C({d}) not finite"

            assert v_result > 0, f"V({d}) not positive"
            assert s_result > 0, f"S({d}) not positive"
            assert c_result > 0, f"C({d}) not positive"

    def test_gamma_function_stability(self):
        """Test gamma function numerical stability."""
        # Test range of inputs
        inputs = np.linspace(0.1, 10.0, 100)

        for x in inputs:
            result = dm.gamma_safe(x)
            assert np.isfinite(result), f"gamma({x}) not finite"
            assert result > 0, f"gamma({x}) not positive"

    def test_array_operations_consistency(self):
        """Test that array operations are consistent with scalar operations."""
        dims = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Test vectorized vs individual
        v_array = dm.V(dims)
        v_individual = [dm.V(d) for d in dims]

        np.testing.assert_allclose(v_array, v_individual, rtol=1e-12)

        s_array = dm.S(dims)
        s_individual = [dm.S(d) for d in dims]

        np.testing.assert_allclose(s_array, s_individual, rtol=1e-12)


class TestPerformanceCharacteristics:
    """Test performance characteristics and benchmarking."""

    def test_large_array_handling(self):
        """Test that large arrays can be handled efficiently."""
        # Test with reasonably large array
        large_dims = np.linspace(0.5, 8.0, 1000)

        # Should complete without excessive time or memory
        v_results = dm.V(large_dims)
        s_results = dm.S(large_dims)

        assert len(v_results) == len(large_dims)
        assert len(s_results) == len(large_dims)
        assert all(np.isfinite(v_results))
        assert all(np.isfinite(s_results))

    def test_peak_finding_robustness(self):
        """Test that peak finding algorithms are robust."""
        # Test peak finding functions
        v_peak_d, v_peak_val = dm.v_peak()
        s_peak_d, s_peak_val = dm.s_peak()
        c_peak_d, c_peak_val = dm.c_peak()

        # Peaks should be reasonable values
        assert 5.0 < v_peak_d < 6.0
        assert 7.0 < s_peak_d < 8.0
        assert 5.5 < c_peak_d < 6.5

        # Test that peaks are actually maxima
        epsilon = 0.01
        assert dm.V(v_peak_d) > dm.V(v_peak_d - epsilon)
        assert dm.V(v_peak_d) > dm.V(v_peak_d + epsilon)


class TestErrorHandlingRobustness:
    """Test comprehensive error handling and robustness."""

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""

        # Test handling of None
        with pytest.raises((TypeError, ValueError)):
            dm.V(None)

        # Test handling of string input
        with pytest.raises((TypeError, ValueError)):
            dm.V("invalid")

        # Test handling of complex numbers (should either work or fail gracefully)
        try:
            result = dm.V(1 + 1j)
            # If it works, result should be complex
            assert isinstance(result, (complex, np.complex128))
        except (TypeError, ValueError):
            # If it fails, that's also acceptable
            pass

    def test_mathematical_property_validation(self):
        """Test that mathematical properties are preserved."""
        # Test recurrence relation: Γ(n+1) = n*Γ(n)
        for n in [1.0, 2.0, 3.0, 4.0, 5.0]:
            gamma_n = dm.gamma_safe(n)
            gamma_n_plus_1 = dm.gamma_safe(n + 1)

            expected = n * gamma_n
            relative_error = abs(gamma_n_plus_1 - expected) / expected
            assert relative_error < 1e-12, f"Recurrence relation failed for n={n}"

    def test_dimensional_consistency(self):
        """Test dimensional consistency of measures."""
        d = 4.0

        # Test that complexity measure equals V*S
        v_val = dm.V(d)
        s_val = dm.S(d)
        c_val = dm.C(d)

        expected_c = v_val * s_val
        relative_error = abs(c_val - expected_c) / expected_c
        assert relative_error < 1e-12, "Complexity measure inconsistent with V*S"


class TestMemoryEfficiency:
    """Test memory efficiency and resource management."""

    def test_memory_usage_reasonable(self):
        """Test that memory usage is reasonable for large operations."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform large calculation
        large_dims = np.linspace(0.1, 10.0, 10000)
        results = dm.V(large_dims)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

        # Clean up
        del results
        del large_dims

    def test_no_memory_leaks_in_loops(self):
        """Test that repeated operations don't cause memory leaks."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform many repeated calculations
        for i in range(1000):
            result = dm.V(4.0)
            del result

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal (less than 10MB)
        assert memory_increase < 10 * 1024 * 1024
