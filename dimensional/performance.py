#!/usr/bin/env python3
"""Performance optimization and benchmarking framework."""

import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from .gamma import factorial_extension
from .mathematics import (
    DimensionalError,
    NumericalInstabilityError,
    ball_volume,
    complexity_measure,
    gamma_safe,
    sphere_surface,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark results container."""
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
    """Property test results container."""
    property_name: str
    passed: bool
    error_magnitude: float
    test_cases: int
    failures: list[dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0


class PerformanceProfiler:
    """Performance profiling and optimization."""

    def __init__(self):
        self.baseline_results: dict[str, BenchmarkResult] = {}
        self.current_results: dict[str, BenchmarkResult] = {}
        self.regression_threshold = 0.1

    def benchmark_function(
        self,
        func: Callable,
        test_inputs: list[Any],
        name: str,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
    ) -> BenchmarkResult:
        """Benchmark function with statistical analysis."""
        logger.debug(f"Benchmarking {name}")

        # Warmup
        for _ in range(warmup_runs):
            for inp in test_inputs[:10]:
                try:
                    func(inp)
                except (DimensionalError, NumericalInstabilityError):
                    pass

        times = []
        errors = 0

        for run in range(benchmark_runs):
            start_time = time.perf_counter()
            for inp in test_inputs:
                try:
                    result = func(inp)
                    if not np.isfinite(result):
                        errors += 1
                except (DimensionalError, NumericalInstabilityError, OverflowError):
                    errors += 1
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        min_time = min(times)
        max_time = max(times)
        total_time = sum(times)
        total_operations = len(test_inputs) * benchmark_runs
        ops_per_second = total_operations / total_time
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
        """Run benchmark suite on core functions."""
        linear_inputs = np.linspace(0.1, 10.0, 1000)
        log_inputs = np.logspace(-1, 2, 1000)
        critical_inputs = [0.5, 1.0, 2.0, 3.0, 4.0, 5.256, 6.0, 7.256, 8.0, 10.0]

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
        """Detect performance regressions between baseline and current results."""
        regressions = {}
        for func_name in baseline.keys():
            if func_name in current:
                baseline_ops = baseline[func_name].operations_per_second
                current_ops = current[func_name].operations_per_second
                change_ratio = (current_ops - baseline_ops) / baseline_ops
                regressions[func_name] = change_ratio

                if change_ratio < -self.regression_threshold:
                    logger.warning(f"Regression in {func_name}: {change_ratio:.2%} slower")
                elif change_ratio > 0.1:
                    logger.info(f"Improvement in {func_name}: {change_ratio:.2%} faster")

        return regressions


class PropertyValidator:
    """Mathematical property validation."""

    def __init__(self):
        self.tolerance = 1e-12
        self.test_results: list[PropertyTestResult] = []

    def validate_gamma_properties(
        self, test_range: NDArray[np.float64]
    ) -> list[PropertyTestResult]:
        """Validate gamma function properties."""
        results = []
        logger.debug("Testing gamma recurrence relation")

        # Test Γ(z+1) = z·Γ(z)
        errors = []
        for z in test_range[test_range > 0]:
            gamma_z = gamma_safe(z)
            gamma_z_plus_1 = gamma_safe(z + 1)
            expected = z * gamma_z
            error = abs(gamma_z_plus_1 - expected) / max(abs(expected), 1e-10)

            if error > self.tolerance:
                errors.append({
                    "z": z,
                    "actual": gamma_z_plus_1,
                    "expected": expected,
                    "error": error,
                })

        results.append(PropertyTestResult(
            property_name="gamma_recurrence_relation",
            passed=len(errors) == 0,
            error_magnitude=max([e["error"] for e in errors], default=0.0),
            test_cases=len(test_range[test_range > 0]),
            failures=errors,
        ))

        # Test Γ(n) = (n-1)! for integers
        logger.debug("Testing gamma factorial identity")
        errors = []
        for n in range(1, 11):
            gamma_n = gamma_safe(n)
            factorial_n_minus_1 = factorial_extension(n - 1)
            error = abs(gamma_n - factorial_n_minus_1) / max(abs(factorial_n_minus_1), 1e-10)

            if error > self.tolerance:
                errors.append({
                    "n": n,
                    "gamma_n": gamma_n,
                    "factorial": factorial_n_minus_1,
                    "error": error,
                })

        results.append(PropertyTestResult(
            property_name="gamma_factorial_identity",
            passed=len(errors) == 0,
            error_magnitude=max([e["error"] for e in errors], default=0.0),
            test_cases=10,
            failures=errors,
        ))

        # Test Γ(1/2) = √π
        gamma_half = gamma_safe(0.5)
        sqrt_pi = math.sqrt(math.pi)
        error = abs(gamma_half - sqrt_pi) / sqrt_pi

        results.append(PropertyTestResult(
            property_name="gamma_half_sqrt_pi",
            passed=error < self.tolerance,
            error_magnitude=error,
            test_cases=1,
            failures=[{"error": error}] if error >= self.tolerance else [],
        ))

        return results

    def validate_dimensional_properties(
        self, test_range: np.ndarray
    ) -> list[PropertyTestResult]:
        """Validate dimensional measure properties."""
        results = []

        # Property 1: C(d) = V(d) * S(d)
        logger.debug("Testing complexity product property")
        errors = []
        for d in test_range[test_range > 0]:
            v_d = ball_volume(d)
            s_d = sphere_surface(d)
            c_d = complexity_measure(d)
            expected = v_d * s_d
            error = abs(c_d - expected) / max(abs(expected), 1e-10)

            if error > self.tolerance:
                errors.append({
                    "d": d,
                    "v_d": v_d,
                    "s_d": s_d,
                    "c_d": c_d,
                    "expected": expected,
                    "error": error,
                })

        results.append(PropertyTestResult(
            property_name="complexity_volume_surface_product",
            passed=len(errors) == 0,
            error_magnitude=max([e["error"] for e in errors], default=0.0),
            test_cases=len(test_range[test_range > 0]),
            failures=errors,
        ))

        # Property 2: S(d) = d * V(d)
        logger.debug("Testing surface-volume relationship")
        errors = []
        for d in test_range[test_range > 0]:
            v_d = ball_volume(d)
            s_d = sphere_surface(d)
            expected = d * v_d
            error = abs(s_d - expected) / max(abs(expected), 1e-10)

            if error > self.tolerance:
                errors.append({
                    "d": d,
                    "v_d": v_d,
                    "s_d": s_d,
                    "expected": expected,
                    "error": error,
                })

        results.append(PropertyTestResult(
            property_name="surface_volume_relationship",
            passed=len(errors) == 0,
            error_magnitude=max([e["error"] for e in errors], default=0.0),
            test_cases=len(test_range[test_range > 0]),
            failures=errors,
        ))

        return results

    def run_comprehensive_validation(self) -> dict[str, list[PropertyTestResult]]:
        """Run comprehensive mathematical property validation."""
        logger.info("Running comprehensive property validation")

        test_range = np.concatenate([
            np.linspace(0.1, 10.0, 100),
            np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.256, 6.0, 7.256, 8.0]),
        ])

        validation_results = {
            "gamma_properties": self.validate_gamma_properties(test_range),
            "dimensional_properties": self.validate_dimensional_properties(test_range),
        }

        total_tests = sum(len(results) for results in validation_results.values())
        total_passed = sum(
            sum(1 for result in results if result.passed)
            for results in validation_results.values()
        )

        logger.info(f"Validation: {total_passed}/{total_tests} tests passed")

        return validation_results
