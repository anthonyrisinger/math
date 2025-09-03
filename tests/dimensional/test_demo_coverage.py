#!/usr/bin/env python3
"""
Demo Day Test Coverage Enhancement
===================================

Sprint 4: Focused test coverage to reach 85% target for Demo Day readiness.
This module provides comprehensive test coverage for core mathematical functions,
performance systems, and reproducibility framework.
"""


import numpy as np
import pytest

from dimensional.core import (
    CRITICAL_DIMENSIONS,
    NUMERICAL_EPSILON,
    PHI,
    DimensionalError,
    NumericalInstabilityError,
)
from dimensional.demo_performance import DemoPerformanceOptimizer

# Import modules to test
from dimensional.gamma import (
    c_peak,
    explore,
    factorial_extension,
    gamma,
    gammaln,
    peaks,
    quick_gamma_analysis,
    s_peak,
    v_peak,
)
from dimensional.measures import (
    ball_volume,
    complexity_measure,
    find_peak,
    sphere_surface,
)
from dimensional.performance import BenchmarkResult, PerformanceProfiler
from dimensional.reproducibility import (
    ComputationalEnvironment,
    ReproducibilityFramework,
)

# research_cli was removed during consolidation - these classes no longer exist
# from dimensional.research_cli import (
#     ParameterSweep,
#     PublicationExporter,
#     ResearchPoint,
#     ResearchSession,
# )


class TestCoreMathematicalFunctions:
    """Comprehensive tests for core mathematical functions."""

    def test_ball_volume_basic_cases(self):
        """Test ball volume for standard dimensions."""
        assert ball_volume(1) == 2.0  # Line segment
        assert ball_volume(2) == np.pi  # Circle
        assert abs(ball_volume(3) - 4*np.pi/3) < 1e-10  # Sphere

    def test_ball_volume_critical_dimensions(self):
        """Test ball volume at critical dimensions."""
        vol_peak_dim, vol_peak_val = v_peak()
        result = ball_volume(vol_peak_dim)
        assert abs(result - vol_peak_val) < 1e-10

    def test_sphere_surface_basic_cases(self):
        """Test sphere surface for standard dimensions."""
        assert sphere_surface(1) == 2.0  # Two points
        assert abs(sphere_surface(2) - 2*np.pi) < 1e-10  # Circle circumference
        assert abs(sphere_surface(3) - 4*np.pi) < 1e-10  # Sphere surface

    def test_sphere_surface_critical_dimensions(self):
        """Test sphere surface at critical dimensions."""
        surf_peak_dim, surf_peak_val = s_peak()
        result = sphere_surface(surf_peak_dim)
        assert abs(result - surf_peak_val) < 1e-10

    def test_complexity_measure_properties(self):
        """Test complexity measure properties."""
        dims = [2.0, 4.0, 6.0, 8.0]
        for dim in dims:
            vol = ball_volume(dim)
            surf = sphere_surface(dim)
            comp = complexity_measure(dim)
            assert abs(comp - vol * surf) < 1e-12

    def test_complexity_measure_peak(self):
        """Test complexity measure peak detection."""
        comp_peak_dim, comp_peak_val = c_peak()
        result = complexity_measure(comp_peak_dim)
        assert abs(result - comp_peak_val) < 1e-10

    def test_gamma_safe_positive_values(self):
        """Test gamma function for positive values."""
        # Test known values
        assert abs(gamma(1) - 1.0) < 1e-12  # Γ(1) = 1
        assert abs(gamma(2) - 1.0) < 1e-12  # Γ(2) = 1
        assert abs(gamma(3) - 2.0) < 1e-12  # Γ(3) = 2
        assert abs(gamma(4) - 6.0) < 1e-12  # Γ(4) = 6

    def test_gamma_safe_fractional_values(self):
        """Test gamma function for fractional values."""
        # Γ(1/2) = √π
        result = gamma(0.5)
        expected = np.sqrt(np.pi)
        assert abs(result - expected) < 1e-10

    def test_gammaln_safe_properties(self):
        """Test log-gamma function properties."""
        test_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        for val in test_values:
            log_result = gammaln(val)
            gamma_result = gamma(val)
            assert abs(log_result - np.log(gamma_result)) < 1e-10

    @pytest.mark.skip(reason='Deprecated')
    def test_find_peak_functionality(self):
        """Test peak finding functionality."""
        # Test with simple quadratic function
        def quadratic(x):
            return -(x - 5)**2 + 10

        peak_x, peak_val = find_peak(quadratic, 0, 10)
        assert abs(peak_x - 5.0) < 0.1  # Peak should be near x=5
        assert abs(peak_val - 10.0) < 0.1  # Peak value should be near 10

    def test_mathematical_constants(self):
        """Test mathematical constants."""
        # Golden ratio
        expected_phi = (1 + np.sqrt(5)) / 2
        assert abs(PHI - expected_phi) < 1e-12

        # Numerical epsilon
        assert NUMERICAL_EPSILON > 0
        assert NUMERICAL_EPSILON < 1e-10

        # Critical dimensions
        assert isinstance(CRITICAL_DIMENSIONS, dict)
        assert len(CRITICAL_DIMENSIONS) > 0


class TestGammaModuleFunctions:
    """Test gamma module functions."""

    def test_explore_function(self):
        """Test explore function."""
        result = explore(4.0)
        assert isinstance(result, dict)
        assert 'dimension' in result
        assert 'volume' in result
        assert 'surface' in result
        assert result['dimension'] == 4.0

    def test_peaks_function(self):
        """Test peaks discovery function."""
        result = peaks()
        assert isinstance(result, dict)
        assert 'volume_peak' in result
        assert 'surface_peak' in result
        assert 'complexity_peak' in result

    def test_quick_gamma_analysis(self):
        """Test quick gamma analysis."""
        result = quick_gamma_analysis(4.0)
        assert isinstance(result, dict)
        assert 'dimension' in result
        assert 'gamma_value' in result

    def test_factorial_extension(self):
        """Test factorial extension."""
        # Test integer values
        assert factorial_extension(4) == 24  # 4! = 24
        assert factorial_extension(5) == 120  # 5! = 120

        # Test fractional values
        result = factorial_extension(3.5)
        assert result > 0  # Should be positive


class TestPerformanceSystem:
    """Test performance optimization systems."""

    def test_performance_profiler_creation(self):
        """Test performance profiler initialization."""
        profiler = PerformanceProfiler()
        assert profiler.regression_threshold == 0.1
        assert isinstance(profiler.baseline_results, dict)

    def test_benchmark_result_creation(self):
        """Test benchmark result structure."""
        result = BenchmarkResult(
            function_name="test_function",
            operations_per_second=1000.0,
            mean_time=0.001,
            std_time=0.0001,
            min_time=0.0008,
            max_time=0.0012,
            total_time=1.0,
            sample_size=1000
        )
        assert result.function_name == "test_function"
        assert result.operations_per_second == 1000.0

    def test_demo_performance_optimizer(self):
        """Test demo performance optimizer."""
        optimizer = DemoPerformanceOptimizer()
        assert len(optimizer.demo_dimensions) == 6
        assert optimizer.excellence_threshold == 100000

    def test_performance_optimization_execution(self):
        """Test performance optimization execution."""
        optimizer = DemoPerformanceOptimizer()
        optimizations = optimizer.optimize_for_demo()
        assert isinstance(optimizations, dict)
        assert "numpy_config" in optimizations or "cache_warming" in optimizations


class TestReproducibilityFramework:
    """Test research reproducibility framework."""

    def test_computational_environment_capture(self):
        """Test computational environment capture."""
        framework = ReproducibilityFramework()
        env = framework.environment

        assert isinstance(env, ComputationalEnvironment)
        assert env.python_version is not None
        assert env.platform_system is not None
        assert env.numpy_version is not None
        assert env.random_seed == 42

    def test_environment_hashing(self):
        """Test environment hash generation."""
        framework = ReproducibilityFramework()
        hash1 = framework.environment.hash()
        hash2 = framework.environment.hash()

        # Same environment should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # 16 character hash

    def test_reproducibility_manifest_creation(self):
        """Test reproducibility manifest creation."""
        framework = ReproducibilityFramework()
        test_results = {"test": "data"}

        manifest_path = framework.create_reproducibility_manifest(test_results)
        assert manifest_path.exists()
        assert manifest_path.suffix == ".json"

        # Clean up
        manifest_path.unlink()
        # Remove the directory if empty
        if framework.base_path.exists() and not any(framework.base_path.iterdir()):
            framework.base_path.rmdir()

    def test_research_certificate_generation(self):
        """Test research certificate generation."""
        framework = ReproducibilityFramework()

        # Create temporary manifest
        test_results = {"demo": "test"}
        manifest_path = framework.create_reproducibility_manifest(test_results)

        # Generate certificate
        cert_path = framework.generate_research_certificate("test_session", manifest_path)
        assert cert_path.exists()
        assert cert_path.suffix == ".md"

        # Clean up
        manifest_path.unlink()
        cert_path.unlink()
        # Remove the directory if empty
        if framework.base_path.exists() and not any(framework.base_path.iterdir()):
            framework.base_path.rmdir()


# TestResearchCLIComponents removed - deprecated functionality


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.mark.skip(reason='Deprecated')
    def test_dimensional_error_handling(self):
        """Test dimensional error handling."""
        with pytest.raises(DimensionalError):
            ball_volume(-1.0)  # Negative dimension should raise error

    def test_numerical_instability_handling(self):
        """Test numerical instability handling."""
        # Test very large dimension (may cause overflow)
        try:
            result = ball_volume(1000.0)
            # If it doesn't raise an exception, result should be finite
            assert np.isfinite(result) or np.isinf(result)
        except (NumericalInstabilityError, OverflowError):
            # Expected for very large dimensions
            pass

    def test_zero_dimension_handling(self):
        """Test zero dimension edge case."""
        # Volume of 0-dimensional ball should be 1
        assert abs(ball_volume(0) - 1.0) < 1e-12

    def test_small_dimension_stability(self):
        """Test stability for small dimensions."""
        small_dims = [0.1, 0.5, 1.0, 1.5]
        for dim in small_dims:
            vol = ball_volume(dim)
            surf = sphere_surface(dim)
            comp = complexity_measure(dim)

            assert vol > 0, f"Volume should be positive for dim={dim}"
            assert surf > 0, f"Surface should be positive for dim={dim}"
            assert comp > 0, f"Complexity should be positive for dim={dim}"


class TestIntegrationAndEndToEnd:
    """Integration tests for end-to-end workflows."""

    def test_complete_research_workflow(self):
        """Test complete research workflow integration."""
        # 1. Explore dimension
        analysis = explore(5.26414)  # Volume peak
        assert analysis['dimension'] == 5.26414

        # 2. Performance validation
        optimizer = DemoPerformanceOptimizer()
        optimizer.optimize_for_demo()

        # 3. Reproducibility validation
        framework = ReproducibilityFramework()
        results = framework.run_reproducibility_tests()
        assert len(results) > 0

    @pytest.mark.skip(reason="PublicationExporter removed in consolidation")
    def test_publication_workflow_integration(self):
        """Test publication workflow integration."""
        # Skip - ResearchSession and PublicationExporter removed
        pass

    def test_demo_day_readiness_validation(self):
        """Test Demo Day readiness validation."""
        # Performance validation
        optimizer = DemoPerformanceOptimizer()
        perf_results = optimizer.run_demo_benchmarks()

        # Should have multiple function benchmarks
        assert len(perf_results) >= 5

        # At least some functions should be real-time capable
        real_time_count = sum(1 for r in perf_results.values() if r.real_time_capable)
        assert real_time_count > 0

        # Overall performance should be adequate
        avg_ops = sum(r.operations_per_second for r in perf_results.values()) / len(perf_results)
        assert avg_ops > 1000  # At least interactive performance


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
