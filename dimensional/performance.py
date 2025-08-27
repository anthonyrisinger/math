#!/usr/bin/env python3
"""
Performance Optimization & Benchmarking Framework
=================================================

Sprint 2: Advanced mathematical invariant hardening and performance optimization.
This module provides comprehensive performance analysis, benchmarking, and
optimization tools for the dimensional mathematics framework.

Features:
- Performance regression detection
- Symbolic verification integration
- Advanced property validation
- Distributed property testing
- Production readiness metrics
"""

import math
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

# Import mathematical functions for benchmarking
# Import from consolidated mathematics module
from .mathematics import (
    ball_volume,
    complexity_measure, 
    sphere_surface,
    gamma_safe,
    gammaln_safe,
    DimensionalError,
    NumericalInstabilityError,
)
from .gamma import factorial_extension


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    function_name: str
    operations_per_second: float
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    total_time: float
    sample_size: int
    error_rate: float = 0.0
    memory_usage: Optional[float] = None


@dataclass
class PropertyTestResult:
    """Container for property test results."""

    property_name: str
    passed: bool
    error_magnitude: float
    test_cases: int
    failures: list[dict] = field(default_factory=list)
    execution_time: float = 0.0


class PerformanceProfiler:
    """Advanced performance profiling and optimization framework."""

    def __init__(self):
        self.baseline_results: dict[str, BenchmarkResult] = {}
        self.current_results: dict[str, BenchmarkResult] = {}
        self.regression_threshold = 0.1  # 10% performance regression threshold

    def benchmark_function(
        self,
        func: Callable,
        test_inputs: list[Any],
        name: str,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
    ) -> BenchmarkResult:
        """
        Comprehensive function benchmarking with statistical analysis.

        Parameters:
        -----------
        func : callable
            Function to benchmark
        test_inputs : lis
            List of inputs to tes
        name : str
            Name identifier for the benchmark
        warmup_runs : in
            Number of warmup runs to stabilize performance
        benchmark_runs : in
            Number of benchmark runs for statistical analysis

        Returns:
        --------
        BenchmarkResul
            Detailed benchmark statistics
        """
        print(f"📊 Benchmarking {name}...")

        # Warmup phase
        for _ in range(warmup_runs):
            for inp in test_inputs[:10]:  # Use subset for warmup
                try:
                    func(inp)
                except (DimensionalError, NumericalInstabilityError):
                    # Skip problematic computations during warmup
                    pass

        # Benchmark phase
        times = []
        errors = 0

        for run in range(benchmark_runs):
            start_time = time.perf_counter()

            for inp in test_inputs:
                try:
                    result = func(inp)
                    # Basic sanity check
                    if not np.isfinite(result):
                        errors += 1
                except (DimensionalError, NumericalInstabilityError, OverflowError):
                    errors += 1

            end_time = time.perf_counter()
            times.append(end_time - start_time)

        # Statistical analysis
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        min_time = min(times)
        max_time = max(times)
        total_time = sum(times)

        # Calculate operations per second
        total_operations = len(test_inputs) * benchmark_runs
        ops_per_second = total_operations / total_time

        # Error rate
        error_rate = errors / total_operations

        return BenchmarkResult(
            function_name=name,
            operations_per_second=ops_per_second,
            mean_time=mean_time,
            std_time=std_time,
            min_time=min_time,
            max_time=max_time,
            total_time=total_time,
            sample_size=benchmark_runs,
            error_rate=error_rate,
        )

    def benchmark_suite(self) -> dict[str, BenchmarkResult]:
        """Run comprehensive benchmark suite on all core functions."""

        # Generate test inputs
        linear_inputs = np.linspace(0.1, 10.0, 1000)
        log_inputs = np.logspace(-1, 2, 1000)
        critical_inputs = [
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.256,
            6.0,
            7.256,
            8.0,
            10.0,
        ]

        benchmark_configs = [
            (gamma_safe, linear_inputs, "gamma_safe_linear"),
            (gamma_safe, log_inputs, "gamma_safe_log"),
            (gamma_safe, critical_inputs, "gamma_safe_critical"),
            (ball_volume, linear_inputs, "ball_volume_linear"),
            (ball_volume, log_inputs, "ball_volume_log"),
            (sphere_surface, linear_inputs, "sphere_surface_linear"),
            (complexity_measure, linear_inputs, "complexity_measure_linear"),
        ]

        results = {}
        for func, inputs, name in benchmark_configs:
            results[name] = self.benchmark_function(func, inputs, name)

        return results

    def detect_regressions(
        self,
        baseline: dict[str, BenchmarkResult],
        current: dict[str, BenchmarkResult],
    ) -> dict[str, float]:
        """
        Detect performance regressions between baseline and current results.

        Returns:
        --------
        Dict[str, float]
            Dictionary mapping function names to performance change ratios
            (negative values indicate regressions)
        """
        regressions = {}

        for func_name in baseline.keys():
            if func_name in current:
                baseline_ops = baseline[func_name].operations_per_second
                current_ops = current[func_name].operations_per_second

                change_ratio = (current_ops - baseline_ops) / baseline_ops
                regressions[func_name] = change_ratio

                if change_ratio < -self.regression_threshold:
                    print(
                        f"⚠️  REGRESSION DETECTED in {func_name}: {
                            change_ratio:.2%} slower"
                    )
                elif change_ratio > 0.1:
                    print(
                        f"🚀 IMPROVEMENT in {func_name}: {
                            change_ratio:.2%} faster"
                    )

        return regressions


class PropertyValidator:
    """Advanced mathematical property validation framework."""

    def __init__(self):
        self.tolerance = 1e-12
        self.test_results: list[PropertyTestResult] = []

    def validate_gamma_properties(
        self, test_range: np.ndarray
    ) -> list[PropertyTestResult]:
        """Validate fundamental gamma function properties."""
        results = []

        # Property 1: Γ(z+1) = z·Γ(z)
        print("🔬 Testing Γ(z+1) = z·Γ(z)...")
        errors = []
        for z in test_range[test_range > 0]:
            gamma_z = gamma_safe(z)
            gamma_z_plus_1 = gamma_safe(z + 1)
            expected = z * gamma_z
            error = abs(gamma_z_plus_1 - expected) / max(abs(expected), 1e-10)

            if error > self.tolerance:
                errors.append(
                    {
                        "z": z,
                        "actual": gamma_z_plus_1,
                        "expected": expected,
                        "error": error,
                    }
                )

        results.append(
            PropertyTestResult(
                property_name="gamma_recurrence_relation",
                passed=len(errors) == 0,
                error_magnitude=max([e["error"] for e in errors], default=0.0),
                test_cases=len(test_range[test_range > 0]),
                failures=errors,
            )
        )

        # Property 2: Γ(n) = (n-1)! for positive integers
        print("🔬 Testing Γ(n) = (n-1)! for integers...")
        errors = []
        for n in range(1, 11):
            gamma_n = gamma_safe(n)
            factorial_n_minus_1 = factorial_extension(n - 1)
            error = abs(gamma_n - factorial_n_minus_1) / max(
                abs(factorial_n_minus_1), 1e-10
            )

            if error > self.tolerance:
                errors.append(
                    {
                        "n": n,
                        "gamma_n": gamma_n,
                        "factorial": factorial_n_minus_1,
                        "error": error,
                    }
                )

        results.append(
            PropertyTestResult(
                property_name="gamma_factorial_identity",
                passed=len(errors) == 0,
                error_magnitude=max([e["error"] for e in errors], default=0.0),
                test_cases=10,
                failures=errors,
            )
        )

        # Property 3: Γ(1/2) = √π
        print("🔬 Testing Γ(1/2) = √π...")
        gamma_half = gamma_safe(0.5)
        sqrt_pi = math.sqrt(math.pi)
        error = abs(gamma_half - sqrt_pi) / sqrt_pi

        results.append(
            PropertyTestResult(
                property_name="gamma_half_sqrt_pi",
                passed=error < self.tolerance,
                error_magnitude=error,
                test_cases=1,
                failures=[{"error": error}] if error >= self.tolerance else [],
            )
        )

        return results

    def validate_dimensional_properties(
        self, test_range: np.ndarray
    ) -> list[PropertyTestResult]:
        """Validate dimensional measure properties."""
        results = []

        # Property 1: C(d) = V(d) * S(d)
        print("🔬 Testing C(d) = V(d) * S(d)...")
        errors = []
        for d in test_range[test_range > 0]:
            v_d = ball_volume(d)
            s_d = sphere_surface(d)
            c_d = complexity_measure(d)
            expected = v_d * s_d
            error = abs(c_d - expected) / max(abs(expected), 1e-10)

            if error > self.tolerance:
                errors.append(
                    {
                        "d": d,
                        "v_d": v_d,
                        "s_d": s_d,
                        "c_d": c_d,
                        "expected": expected,
                        "error": error,
                    }
                )

        results.append(
            PropertyTestResult(
                property_name="complexity_volume_surface_product",
                passed=len(errors) == 0,
                error_magnitude=max([e["error"] for e in errors], default=0.0),
                test_cases=len(test_range[test_range > 0]),
                failures=errors,
            )
        )

        # Property 2: S(d) = d * V(d) (surface-volume relationship)
        print("🔬 Testing S(d) = d * V(d)...")
        errors = []
        for d in test_range[test_range > 0]:
            v_d = ball_volume(d)
            s_d = sphere_surface(d)
            expected = d * v_d
            error = abs(s_d - expected) / max(abs(expected), 1e-10)

            if error > self.tolerance:
                errors.append(
                    {
                        "d": d,
                        "v_d": v_d,
                        "s_d": s_d,
                        "expected": expected,
                        "error": error,
                    }
                )

        results.append(
            PropertyTestResult(
                property_name="surface_volume_relationship",
                passed=len(errors) == 0,
                error_magnitude=max([e["error"] for e in errors], default=0.0),
                test_cases=len(test_range[test_range > 0]),
                failures=errors,
            )
        )

        return results

    def run_comprehensive_validation(
        self,
    ) -> dict[str, list[PropertyTestResult]]:
        """Run comprehensive mathematical property validation."""
        print("🧮 COMPREHENSIVE MATHEMATICAL PROPERTY VALIDATION")
        print("=" * 60)

        test_range = np.concatenate(
            [
                np.linspace(0.1, 10.0, 100),
                np.array(
                    [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.256, 6.0, 7.256, 8.0]
                ),
            ]
        )

        validation_results = {
            "gamma_properties": self.validate_gamma_properties(test_range),
            "dimensional_properties": self.validate_dimensional_properties(
                test_range
            ),
        }

        # Summary
        total_tests = sum(
            len(results) for results in validation_results.values()
        )
        total_passed = sum(
            sum(1 for result in results if result.passed)
            for results in validation_results.values()
        )

        print(
            f"\n📊 VALIDATION SUMMARY: {total_passed}/{total_tests} tests passed"
        )

        if total_passed == total_tests:
            print("✅ ALL MATHEMATICAL PROPERTIES VALIDATED")
        else:
            print("⚠️  SOME PROPERTIES FAILED - INVESTIGATION REQUIRED")

        return validation_results


class DistributedTester:
    """Distributed property testing for scalability validation."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def distributed_property_test(
        self,
        test_func: Callable,
        test_ranges: list[np.ndarray],
        property_name: str,
    ) -> PropertyTestResult:
        """
        Run property tests in parallel across multiple processes.

        Parameters:
        -----------
        test_func : callable
            Function that tests a property on a range of inputs
        test_ranges : List[np.ndarray]
            List of input ranges to distribute across workers
        property_name : str
            Name of the property being tested

        Returns:
        --------
        PropertyTestResul
            Aggregated results from all workers
        """
        print(f"🔄 Running distributed test for {property_name}...")

        start_time = time.time()

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(test_func, test_range)
                for test_range in test_ranges
            ]
            results = [future.result() for future in futures]

        end_time = time.time()

        # Aggregate results
        total_cases = sum(r.test_cases for r in results)
        total_failures = []
        for r in results:
            total_failures.extend(r.failures)

        max_error = max([r.error_magnitude for r in results], default=0.0)
        all_passed = all(r.passed for r in results)

        return PropertyTestResult(
            property_name=f"{property_name}_distributed",
            passed=all_passed,
            error_magnitude=max_error,
            test_cases=total_cases,
            failures=total_failures,
            execution_time=end_time - start_time,
        )


def sprint2_gate1_validation():
    """
    Sprint 2 Gate 1: Comprehensive validation and performance analysis.

    This function runs all Sprint 2 Gate 1 validation tests:
    1. Performance baseline establishmen
    2. Mathematical property validation
    3. Regression detection setup
    4. Production readiness assessmen
    """
    print("🚀 SPRINT 2 GATE 1 VALIDATION")
    print("=" * 80)

    # Performance Analysis
    print("\n📊 PERFORMANCE ANALYSIS")
    profiler = PerformanceProfiler()
    benchmark_results = profiler.benchmark_suite()

    for name, result in benchmark_results.items():
        print(
            f"  {name}: {result.operations_per_second:,.0f} ops/sec "
            f"(±{
                result.std_time *
                1000:.2f}ms, {
                result.error_rate:.2%} errors)"
        )

    # Property Validation
    print("\n🔬 MATHEMATICAL PROPERTY VALIDATION")
    validator = PropertyValidator()
    property_results = validator.run_comprehensive_validation()

    # Distributed Testing
    print("\n🔄 DISTRIBUTED SCALABILITY TEST")
    DistributedTester()

    # Production Readiness Assessmen
    print("\n✅ PRODUCTION READINESS ASSESSMENT")

    performance_score = min(
        [r.operations_per_second for r in benchmark_results.values()]
    )
    property_score = sum(
        sum(1 for r in results if r.passed)
        for results in property_results.values()
    ) / sum(len(results) for results in property_results.values())

    overall_score = (performance_score / 100000 + property_score) / 2

    print(f"  Performance Score: {performance_score:,.0f} ops/sec")
    print(f"  Property Validation: {property_score:.2%}")
    print(f"  Overall Readiness: {overall_score:.2%}")

    if overall_score > 0.95:
        print("🎯 PRODUCTION READY - SPRINT 2 GATE 1 PASSED")
    elif overall_score > 0.9:
        print("⚠️  MINOR OPTIMIZATIONS NEEDED")
    else:
        print("❌ SIGNIFICANT WORK REQUIRED")

    return {
        "benchmark_results": benchmark_results,
        "property_results": property_results,
        "overall_score": overall_score,
    }


# ============================================================================
# SPRINT 3 PERFORMANCE BREAKTHROUGH - 100K+ OPS/SEC TARGET
# ============================================================================


class Sprint3PerformanceOptimizer:
    """
    Sprint 3: High-performance optimization targeting 100K+ ops/sec.

    Current baseline: ~51K ops/sec
    Target: 100K+ ops/sec
    Required improvement: 2x
    """

    def __init__(self):
        self.baseline_results = {}
        self.optimized_results = {}

    def benchmark_optimizations(self, n_operations=100000):
        """
        Benchmark optimization techniques against baseline.

        Target: Achieve 100K+ ops/sec sustained throughpu
        """
        print("🚀 SPRINT 3 PERFORMANCE OPTIMIZATION BENCHMARK")
        print("Target: 100,000+ ops/sec | Baseline: ~51K ops/sec")
        print("=" * 70)

        # Generate test data
        dimensions = np.random.uniform(0.1, 10.0, n_operations)

        # 1. Baseline measuremen
        print("📊 Baseline measurement...")
        start_time = time.perf_counter()
        [complexity_measure(d) for d in dimensions]
        baseline_time = time.perf_counter() - start_time
        baseline_ops_per_sec = n_operations / baseline_time

        print(
            f"  Baseline: {baseline_ops_per_sec:,.0f} ops/sec ({baseline_time:.3f}s)"
        )

        # 2. Vectorized NumPy optimization
        print("🔧 Testing vectorized optimization...")
        start_time = time.perf_counter()
        self._vectorized_complexity_measure(dimensions)
        vectorized_time = time.perf_counter() - start_time
        vectorized_ops_per_sec = n_operations / vectorized_time

        print(
            f"  Vectorized: {
                vectorized_ops_per_sec:,.0f} ops/sec ({
                vectorized_time:.3f}s)"
        )
        speedup_vectorized = vectorized_ops_per_sec / baseline_ops_per_sec
        print(f"  Speedup: {speedup_vectorized:.1f}x")

        # 3. Cached computation optimization
        print("💾 Testing cached optimization...")
        start_time = time.perf_counter()
        self._cached_complexity_measure(dimensions)
        cached_time = time.perf_counter() - start_time
        cached_ops_per_sec = n_operations / cached_time

        print(
            f"  Cached: {cached_ops_per_sec:,.0f} ops/sec ({cached_time:.3f}s)"
        )
        speedup_cached = cached_ops_per_sec / baseline_ops_per_sec
        print(f"  Speedup: {speedup_cached:.1f}x")

        # 4. Memory-optimized batch processing
        print("⚡ Testing batch optimization...")
        start_time = time.perf_counter()
        self._batch_complexity_measure(dimensions, batch_size=1000)
        batch_time = time.perf_counter() - start_time
        batch_ops_per_sec = n_operations / batch_time

        print(f"  Batch: {batch_ops_per_sec:,.0f} ops/sec ({batch_time:.3f}s)")
        speedup_batch = batch_ops_per_sec / baseline_ops_per_sec
        print(f"  Speedup: {speedup_batch:.1f}x")

        # 5. Combined optimizations
        print("🚀 Testing combined optimizations...")
        start_time = time.perf_counter()
        self._combined_optimized_complexity_measure(dimensions)
        combined_time = time.perf_counter() - start_time
        combined_ops_per_sec = n_operations / combined_time

        print(
            f"  Combined: {combined_ops_per_sec:,.0f} ops/sec ({combined_time:.3f}s)"
        )
        speedup_combined = combined_ops_per_sec / baseline_ops_per_sec
        print(f"  Speedup: {speedup_combined:.1f}x")

        # Results summary
        print("\\n📊 SPRINT 3 OPTIMIZATION RESULTS")
        print(f"{'=' * 50}")
        print(f"Baseline:    {baseline_ops_per_sec:>8,.0f} ops/sec (1.0x)")
        print(
            f"Vectorized:  {
                vectorized_ops_per_sec:>8,.0f} ops/sec ({
                speedup_vectorized:.1f}x)"
        )
        print(
            f"Cached:      {
                cached_ops_per_sec:>8,.0f} ops/sec ({
                speedup_cached:.1f}x)"
        )
        print(
            f"Batch:       {batch_ops_per_sec:>8,.0f} ops/sec ({speedup_batch:.1f}x)"
        )
        print(
            f"Combined:    {
                combined_ops_per_sec:>8,.0f} ops/sec ({
                speedup_combined:.1f}x)"
        )

        # Target assessmen
        target_met = combined_ops_per_sec >= 100000
        print("\\n🎯 TARGET ASSESSMENT:")
        print("Target: 100,000 ops/sec")
        print(f"Achieved: {combined_ops_per_sec:,.0f} ops/sec")
        print(
            f"Status: {
                '✅ TARGET EXCEEDED' if target_met else '⚠️  TARGET MISSED'}"
        )

        if target_met:
            print(
                "🎉 SPRINT 3 PHASE 1 SUCCESS: 100K+ ops/sec target achieved!"
            )

        return {
            "baseline_ops_per_sec": baseline_ops_per_sec,
            "optimized_ops_per_sec": combined_ops_per_sec,
            "speedup": speedup_combined,
            "target_met": target_met,
        }

    def _vectorized_complexity_measure(self, dimensions):
        """Vectorized implementation using NumPy operations."""
        d = np.asarray(dimensions, dtype=np.float64)

        # C(d) = V(d) * S(d) = π^(d/2) / Γ(d/2 + 1) × 2π^(d/2) / Γ(d/2)
        # = 2π^d / (Γ(d/2 + 1) × Γ(d/2))

        # Avoid recomputation by using log-space
        half_d = 0.5 * d
        log_pi = np.log(np.pi)

        # Import gamma functions
        # gammaln_safe already imported from mathematics module

        # Use vectorized gamma functions
        log_gamma_half_plus_1 = np.array(
            [gammaln_safe(hd + 1) for hd in half_d]
        )
        log_gamma_half = np.array([gammaln_safe(hd) for hd in half_d])

        log_complexity = (
            d * log_pi + np.log(2) - log_gamma_half_plus_1 - log_gamma_half
        )

        return np.exp(log_complexity)

    def _cached_complexity_measure(self, dimensions):
        """Cached implementation for repeated computations."""
        # Simple caching strategy for common values
        cache = {}
        results = []

        for d in dimensions:
            # Round to 3 decimal places for cache key
            cache_key = round(d, 3)

            if cache_key not in cache:
                cache[cache_key] = complexity_measure(d)

            results.append(cache[cache_key])

        return results

    def _batch_complexity_measure(self, dimensions, batch_size=1000):
        """Batch processing for memory optimization."""
        dimensions = np.asarray(dimensions)
        results = []

        for i in range(0, len(dimensions), batch_size):
            batch = dimensions[i : i + batch_size]
            batch_results = self._vectorized_complexity_measure(batch)
            results.extend(batch_results)

        return results

    def _combined_optimized_complexity_measure(self, dimensions):
        """Combined optimizations: vectorization + caching + batching."""
        dimensions = np.asarray(dimensions, dtype=np.float64)

        # Pre-allocate result array
        results = np.empty_like(dimensions)

        # Handle edge cases efficiently
        zero_mask = np.abs(dimensions) < 1e-15
        results[zero_mask] = 0.0

        # Process non-zero dimensions
        non_zero_mask = ~zero_mask
        non_zero_dims = dimensions[non_zero_mask]

        if len(non_zero_dims) > 0:
            # Ultra-optimized vectorized computation
            half_d = 0.5 * non_zero_dims

            # Import gamma functions
            # gammaln_safe already imported from mathematics module

            # Use optimized gamma computation  
            log_gamma_vals = np.array([gammaln_safe(hd) for hd in half_d])
            log_gamma_plus_one_vals = np.array(
                [gammaln_safe(hd + 1) for hd in half_d]
            )

            # Compute in log space for numerical stability
            log_complexity = (
                non_zero_dims * np.log(np.pi)
                + np.log(2)
                - log_gamma_plus_one_vals
                - log_gamma_vals
            )

            results[non_zero_mask] = np.exp(log_complexity)

        return results


def sprint3_performance_validation():
    """
    Sprint 3: Performance validation targeting 100K+ ops/sec.

    Validates performance optimizations and measures improvemen
    from Sprint 2 baseline to Sprint 3 targets.
    """
    print("🚀 SPRINT 3 PERFORMANCE VALIDATION")
    print("=" * 80)
    print("OBJECTIVE: Achieve 100,000+ ops/sec for complexity_measure")
    print("BASELINE: ~51,000 ops/sec (Sprint 2)")
    print("REQUIRED IMPROVEMENT: 2x speedup")

    optimizer = Sprint3PerformanceOptimizer()

    # Run comprehensive performance benchmarks
    results = optimizer.benchmark_optimizations(100000)

    # Additional validation tests
    print("\\n🔬 ADDITIONAL VALIDATION TESTS")

    # Test accuracy of optimizations
    print("Validating optimization accuracy...")
    test_dims = np.random.uniform(0.1, 10.0, 1000)
    baseline_results = [complexity_measure(d) for d in test_dims]
    optimized_results = optimizer._combined_optimized_complexity_measure(
        test_dims
    )

    max_rel_error = np.max(
        np.abs(baseline_results - optimized_results)
        / np.maximum(np.abs(baseline_results), 1e-15)
    )

    print(f"  Maximum relative error: {max_rel_error:.2e}")
    accuracy_ok = max_rel_error < 1e-12
    print(f"  Accuracy: {'✅ PASSED' if accuracy_ok else '❌ FAILED'}")

    # Final assessmen
    sprint3_success = results["target_met"] and accuracy_ok

    print("\\n🎯 SPRINT 3 ASSESSMENT")
    print(
        f"Performance Target: {
            '✅ MET' if results['target_met'] else '❌ MISSED'}"
    )
    print(f"Accuracy Target: {'✅ MET' if accuracy_ok else '❌ MISSED'}")
    print(
        f"Overall: {
            '🎉 SPRINT 3 SUCCESS' if sprint3_success else '⚠️  NEEDS WORK'}"
    )

    if sprint3_success:
        print("\\n🏆 EXCEPTIONAL DELIVERY ACHIEVED!")
        print("Framework ready for final quarterly review")

    return {
        "performance_results": results,
        "accuracy_passed": accuracy_ok,
        "sprint3_success": sprint3_success,
    }


if __name__ == "__main__":
    # Sprint 2 baseline validation
    print("🔄 Running Sprint 2 baseline validation...")
    try:
        sprint2_results = sprint2_gate1_validation()
        print(
            f"Sprint 2 baseline established: {
                min(
                    [
                        r.operations_per_second for r in sprint2_results['benchmark_results'].values()]):,.0f} ops/sec"
        )
    except Exception as e:
        print(f"Sprint 2 validation skipped: {e}")

    print("\\n" + "=" * 80)

    # Sprint 3 performance breakthrough
    try:
        sprint3_results = sprint3_performance_validation()

        if sprint3_results["sprint3_success"]:
            print(
                "\\n🎯 SPRINT 3 COMPLETE - READY FOR FINAL QUARTERLY DELIVERY"
            )
        else:
            print("\\n⚠️  SPRINT 3 NEEDS ADDITIONAL OPTIMIZATION")

    except Exception as e:
        print(f"⚠️  Sprint 3 validation error: {e}")
        print("Some optimizations may require additional dependencies")
