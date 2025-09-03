#!/usr/bin/env python3
"""
Performance Benchmarks
======================

Performance testing and benchmarking for production deployment.
"""

import time

import numpy as np
import pytest

import dimensional as dm


class TestPerformanceBenchmarks:
    """Test performance characteristics for production deployment."""

    def test_single_value_performance(self):
        """Test single value computation performance."""
        start = time.time()

        # Should complete very quickly
        for i in range(1000):
            dm.V(4.0)

        elapsed = time.time() - start
        ops_per_second = 1000 / elapsed

        # Should achieve >10K ops/second
        assert ops_per_second > 10000, f"Only {ops_per_second:.0f} ops/sec, expected >10K"

    def test_vectorized_performance(self):
        """Test vectorized computation performance."""
        dims = np.linspace(0.5, 8.0, 10000)

        start = time.time()
        results = dm.V(dims)
        elapsed = time.time() - start

        ops_per_second = len(dims) / elapsed

        # Should achieve >100K ops/second for vectorized operations
        assert ops_per_second > 100000, f"Only {ops_per_second:.0f} ops/sec, expected >100K"
        assert len(results) == len(dims)
        assert all(np.isfinite(results))

    def test_gamma_function_performance(self):
        """Test gamma function performance."""
        values = np.linspace(0.1, 10.0, 1000)

        start = time.time()
        dm.gamma_safe(values)
        elapsed = time.time() - start

        ops_per_second = len(values) / elapsed

        # Should achieve >50K ops/second
        assert ops_per_second > 50000, f"Only {ops_per_second:.0f} ops/sec, expected >50K"

    def test_peak_finding_performance(self):
        """Test peak finding performance."""
        start = time.time()

        peaks = dm.find_all_peaks()

        elapsed = time.time() - start

        # Should complete in less than 1 second
        assert elapsed < 1.0, f"Peak finding took {elapsed:.2f}s, expected <1s"
        assert isinstance(peaks, dict)
        assert 'volume_peak' in peaks

    def test_phase_analysis_performance(self):
        """Test phase analysis performance."""
        start = time.time()

        analysis = dm.quick_phase_analysis([1, 2, 3, 4, 5])

        elapsed = time.time() - start

        # Should complete quickly
        assert elapsed < 0.1, f"Phase analysis took {elapsed:.3f}s, expected <0.1s"
        assert isinstance(analysis, dict)

    def test_memory_efficiency(self):
        """Test memory efficiency for large arrays."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Process large array
        large_dims = np.linspace(0.1, 10.0, 100000)
        results = dm.V(large_dims)

        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory

        # Clean up
        del results
        del large_dims

        # Memory increase should be reasonable (less than 50MB for 100K elements)
        assert memory_increase < 50 * 1024 * 1024, f"Memory usage {memory_increase/1024/1024:.1f}MB, expected <50MB"

    @pytest.mark.slow
    def test_sustained_performance(self):
        """Test sustained performance over time."""
        times = []

        for i in range(100):
            start = time.time()

            # Mixed operations
            dm.V(np.random.uniform(0.5, 8.0, 100))
            dm.S(np.random.uniform(0.5, 8.0, 100))
            dm.gamma_safe(np.random.uniform(0.1, 5.0, 50))

            elapsed = time.time() - start
            times.append(elapsed)

        # Performance should be consistent (no significant degradation)
        early_avg = np.mean(times[:10])
        late_avg = np.mean(times[-10:])

        degradation = (late_avg - early_avg) / early_avg
        assert degradation < 0.5, f"Performance degraded by {degradation*100:.1f}%, expected <50%"


class TestNumericalAccuracy:
    """Test numerical accuracy under various conditions."""

    def test_high_precision_gamma(self):
        """Test gamma function high precision."""
        # Known high-precision values
        test_cases = [
            (1.0, 1.0),
            (2.0, 1.0),
            (3.0, 2.0),
            (0.5, np.sqrt(np.pi)),
            (1.5, 0.5 * np.sqrt(np.pi))
        ]

        for x, expected in test_cases:
            result = dm.gamma_safe(x)
            relative_error = abs(result - expected) / expected
            assert relative_error < 1e-14, f"γ({x}) error {relative_error}, expected <1e-14"

    def test_dimensional_measures_precision(self):
        """Test dimensional measures precision."""
        # Known exact values
        assert abs(dm.V(0) - 1.0) < 1e-15
        assert abs(dm.V(1) - 2.0) < 1e-15
        assert abs(dm.V(2) - np.pi) < 1e-15

        # Consistency check: C(d) = V(d) * S(d)
        for d in [1.0, 2.0, 3.0, 4.0, 5.0]:
            v = dm.V(d)
            s = dm.S(d)
            c = dm.C(d)

            expected_c = v * s
            relative_error = abs(c - expected_c) / expected_c
            assert relative_error < 1e-14, f"C({d}) consistency error {relative_error}"

    def test_edge_case_stability(self):
        """Test numerical stability at edge cases."""
        # Very small dimensions
        small_dims = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
        for d in small_dims:
            result = dm.V(d)
            assert np.isfinite(result), f"V({d}) not finite"
            assert result > 0, f"V({d}) not positive"

        # Large dimensions (should decay gracefully)
        large_dims = [20.0, 50.0, 100.0]
        for d in large_dims:
            result = dm.V(d)
            assert np.isfinite(result), f"V({d}) not finite"
            assert result > 0, f"V({d}) not positive"
            assert result < 1e10, f"V({d}) unexpectedly large"


class TestProductionReadiness:
    """Test production readiness characteristics."""

    def test_error_handling_robustness(self):
        """Test robust error handling."""
        # Invalid inputs should raise appropriate errors
        with pytest.raises((TypeError, ValueError)):
            dm.V("invalid")

        with pytest.raises((TypeError, ValueError)):
            dm.V(None)

        # Complex inputs should either work or fail gracefully
        try:
            result = dm.V(1+1j)
            assert isinstance(result, complex)
        except (TypeError, ValueError):
            pass  # Acceptable to reject complex inputs

    def test_thread_safety(self):
        """Test thread safety for concurrent access."""
        import queue
        import threading

        results = queue.Queue()

        def worker():
            try:
                # Mixed operations in different threads
                v = dm.V(4.0)
                s = dm.S(4.0)
                g = dm.gamma_safe(3.0)
                results.put((v, s, g))
            except Exception as e:
                results.put(e)

        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check all results are correct
        while not results.empty():
            result = results.get()
            if isinstance(result, Exception):
                pytest.fail(f"Thread error: {result}")
            else:
                v, s, g = result
                assert abs(v - 4.934802200544679) < 1e-12
                assert abs(g - 2.0) < 1e-12

    def test_import_time(self):
        """Test import time is reasonable."""
        import subprocess
        import sys

        start = time.time()
        result = subprocess.run([
            sys.executable, "-c", "import dimensional"
        ], capture_output=True)
        elapsed = time.time() - start

        # Should import in less than 2 seconds
        assert elapsed < 2.0, f"Import took {elapsed:.2f}s, expected <2s"
        assert result.returncode == 0

    def test_api_completeness(self):
        """Test API completeness for production use."""
        # Essential functions should be available
        essential_functions = [
            'V', 'S', 'C',  # Dimensional measures
            'gamma_safe', 'gammaln_safe',  # Gamma functions
            'find_all_peaks', 'v_peak', 's_peak', 'c_peak',  # Peak finding
            'PhaseDynamicsEngine', 'quick_phase_analysis',  # Phase dynamics
            'PHI', 'PSI', 'PI', 'VARPI',  # Constants
        ]

        for func_name in essential_functions:
            assert hasattr(dm, func_name), f"Missing essential function: {func_name}"

    def test_version_and_metadata(self):
        """Test that version and metadata are properly set."""
        assert hasattr(dm, '__version__')
        assert hasattr(dm, '__author__')
        assert hasattr(dm, '__description__')

        # Version should follow semantic versioning
        import re
        version_pattern = r'^\d+\.\d+\.\d+$'
        assert re.match(version_pattern, dm.__version__), f"Invalid version format: {dm.__version__}"
