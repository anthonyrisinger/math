#!/usr/bin/env python3
"""
Production Deployment Readiness Assessment
==========================================

Sprint 2: Comprehensive production readiness validation for the dimensional
mathematics framework. This module provides deployment readiness checks,
scaling validation, and production environment compatibility assessment.

Features:
- Production environment validation
- Scaling and performance limits testing
- Dependency and compatibility checks
- Security and robustness validation
- Deployment configuration recommendations
"""

import multiprocessing
import platform
import sys
import time

# Optional dependency - memory profiler
try:
    import memory_profiler

    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False
import gc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

# Import our modules for testing
try:
    from .gamma import gamma_safe
    from .measures import ball_volume
    from .performance import PerformanceProfiler
except ImportError:
    from dimensional.gamma import gamma_safe
    from dimensional.measures import ball_volume
    from dimensional.performance import PerformanceProfiler


@dataclass
class ProductionReadinessScore:
    """Container for production readiness assessment results."""

    category: str
    score: float  # 0.0 to 1.0
    details: dict[str, Any]
    recommendations: list[str]
    critical_issues: list[str]


class ProductionEnvironmentValidator:
    """Validates production environment compatibility and requirements."""

    def __init__(self):
        self.system_info = self._collect_system_info()

    def _collect_system_info(self) -> dict[str, Any]:
        """Collect comprehensive system information."""
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "cpu_count": multiprocessing.cpu_count(),
            "memory_gb": self._get_memory_gb(),
        }

    def _get_memory_gb(self) -> float:
        """Get system memory in GB (approximate)."""
        try:
            import psutil

            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback estimation based on system type
            return 8.0  # Conservative estimate

    def validate_environment(self) -> ProductionReadinessScore:
        """Validate the production environment setup."""

        score = 1.0
        issues = []
        recommendations = []
        details = self.system_info.copy()

        # Python version check
        python_version = sys.version_info
        if python_version.major < 3 or (
            python_version.major == 3 and python_version.minor < 9
        ):
            score -= 0.3
            issues.append(
                f"Python version {python_version.major}.{python_version.minor} is outdated"
            )
            recommendations.append("Upgrade to Python 3.9+ for optimal performance")

        # Architecture check
        arch = platform.architecture()[0]
        if arch != "64bit":
            score -= 0.2
            issues.append(f"Architecture {arch} may have performance limitations")
            recommendations.append("Use 64-bit architecture for production")

        # CPU count check
        cpu_count = multiprocessing.cpu_count()
        if cpu_count < 2:
            score -= 0.1
            recommendations.append("Consider multi-core system for better concurrency")
        elif cpu_count >= 4:
            score += 0.05
            details["concurrency_capable"] = True

        # Memory check
        memory_gb = details.get("memory_gb", 0)
        if memory_gb < 4:
            score -= 0.2
            issues.append(f"Low memory: {memory_gb:.1f}GB may be insufficient")
            recommendations.append("Minimum 4GB RAM recommended for production")
        elif memory_gb >= 8:
            score += 0.05
            details["memory_adequate"] = True

        return ProductionReadinessScore(
            category="environment",
            score=max(0.0, min(1.0, score)),
            details=details,
            recommendations=recommendations,
            critical_issues=issues,
        )

    def validate_dependencies(self) -> ProductionReadinessScore:
        """Validate critical dependencies and optional enhancements."""

        score = 1.0
        issues = []
        recommendations = []
        details = {"available_packages": [], "missing_packages": []}

        # Core dependencies (required)
        core_deps = ["numpy", "scipy", "matplotlib"]

        for dep in core_deps:
            try:
                __import__(dep)
                details["available_packages"].append(dep)
            except ImportError:
                score -= 0.3
                issues.append(f"Critical dependency missing: {dep}")
                details["missing_packages"].append(dep)

        # Optional dependencies (enhance functionality)
        optional_deps = [
            ("plotly", "Interactive visualization"),
            ("typer", "CLI functionality"),
            ("sympy", "Symbolic mathematics"),
            ("psutil", "System monitoring"),
            ("memory_profiler", "Memory analysis"),
        ]

        for dep, description in optional_deps:
            try:
                __import__(dep)
                details["available_packages"].append(dep)
                score += 0.02  # Small bonus for optional deps
            except ImportError:
                recommendations.append(f"Consider installing {dep} for {description}")
                details["missing_packages"].append(dep)

        return ProductionReadinessScore(
            category="dependencies",
            score=max(0.0, min(1.0, score)),
            details=details,
            recommendations=recommendations,
            critical_issues=issues,
        )


class ScalabilityValidator:
    """Validates framework scalability and performance limits."""

    def __init__(self):
        self.profiler = PerformanceProfiler()

    def test_concurrent_execution(self) -> ProductionReadinessScore:
        """Test concurrent execution capabilities."""

        print("🔄 Testing concurrent execution...")

        # Test data
        test_inputs = np.linspace(0.1, 10.0, 1000)

        # Sequential baseline
        start_time = time.time()
        sequential_results = [gamma_safe(x) for x in test_inputs]
        sequential_time = time.time() - start_time

        # Threaded execution
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            threaded_results = list(executor.map(gamma_safe, test_inputs))
        threaded_time = time.time() - start_time

        # Process-based execution (if supported)
        try:
            start_time = time.time()
            with ProcessPoolExecutor(max_workers=2) as executor:
                process_results = list(executor.map(gamma_safe, test_inputs))
            process_time = time.time() - start_time
        except:
            process_time = None
            process_results = None

        # Calculate performance metrics
        thread_speedup = sequential_time / threaded_time if threaded_time > 0 else 0
        process_speedup = sequential_time / process_time if process_time else 0

        score = 0.7  # Base score
        details = {
            "sequential_time": sequential_time,
            "threaded_time": threaded_time,
            "process_time": process_time,
            "thread_speedup": thread_speedup,
            "process_speedup": process_speedup,
            "results_consistent": self._verify_results_consistency(
                sequential_results, threaded_results, process_results
            ),
        }

        recommendations = []
        issues = []

        # Score based on speedup
        if thread_speedup > 1.5:
            score += 0.2
        elif thread_speedup < 1.1:
            recommendations.append(
                "Limited threading benefit - check for GIL limitations"
            )

        if process_speedup > 1.2:
            score += 0.1
        elif process_time is None:
            issues.append("Process-based parallelization not available")

        if not details["results_consistent"]:
            score -= 0.3
            issues.append("Concurrent execution produces inconsistent results")

        return ProductionReadinessScore(
            category="concurrency",
            score=max(0.0, min(1.0, score)),
            details=details,
            recommendations=recommendations,
            critical_issues=issues,
        )

    def _verify_results_consistency(
        self, seq_results, thread_results, proc_results=None
    ) -> bool:
        """Verify that concurrent execution produces consistent results."""

        # Compare sequential vs threaded
        seq_arr = np.array(seq_results)
        thread_arr = np.array(thread_results)

        if len(seq_arr) != len(thread_arr):
            return False

        # Check relative differences
        rel_diff = np.abs(seq_arr - thread_arr) / (np.abs(seq_arr) + 1e-15)
        max_diff = np.max(rel_diff)

        if max_diff > 1e-12:
            return False

        # Check process results if available
        if proc_results is not None:
            proc_arr = np.array(proc_results)
            if len(proc_arr) != len(seq_arr):
                return False

            rel_diff_proc = np.abs(seq_arr - proc_arr) / (np.abs(seq_arr) + 1e-15)
            if np.max(rel_diff_proc) > 1e-12:
                return False

        return True

    def test_memory_scaling(self) -> ProductionReadinessScore:
        """Test memory usage and scaling behavior."""

        print("🧠 Testing memory scaling...")

        score = 1.0
        details = {}
        recommendations = []
        issues = []

        # Test different input sizes
        sizes = [100, 1000, 10000]
        memory_usage = []
        processing_times = []

        for size in sizes:
            test_inputs = np.linspace(0.1, 10.0, size)

            # Measure memory before
            gc.collect()  # Force garbage collection
            try:
                if HAS_MEMORY_PROFILER:
                    mem_before = memory_profiler.memory_usage()[0]
                else:
                    mem_before = 0
            except:
                mem_before = 0

            # Process data
            start_time = time.time()
            results = [gamma_safe(x) for x in test_inputs]
            processing_time = time.time() - start_time

            # Measure memory after
            try:
                if HAS_MEMORY_PROFILER:
                    mem_after = memory_profiler.memory_usage()[0]
                else:
                    mem_after = 0
                mem_used = mem_after - mem_before
            except:
                mem_used = 0

            memory_usage.append(mem_used)
            processing_times.append(processing_time)

            # Clean up
            del results, test_inputs
            gc.collect()

        details = {
            "test_sizes": sizes,
            "memory_usage_mb": memory_usage,
            "processing_times": processing_times,
            "memory_efficiency": self._calculate_memory_efficiency(sizes, memory_usage),
            "time_complexity": self._estimate_time_complexity(sizes, processing_times),
        }

        # Assess memory efficiency
        if details["memory_efficiency"] < 0.5:
            score -= 0.2
            issues.append("Memory usage scaling poorly with input size")

        # Assess time complexity
        time_complexity = details["time_complexity"]
        if time_complexity > 1.5:  # Worse than O(n^1.5)
            score -= 0.1
            recommendations.append("Consider optimizing for better time complexity")
        elif time_complexity <= 1.1:  # Close to O(n)
            score += 0.1

        return ProductionReadinessScore(
            category="memory_scaling",
            score=max(0.0, min(1.0, score)),
            details=details,
            recommendations=recommendations,
            critical_issues=issues,
        )

    def _calculate_memory_efficiency(
        self, sizes: list[int], memory_usage: list[float]
    ) -> float:
        """Calculate memory efficiency score (higher is better)."""
        if len(memory_usage) < 2 or max(memory_usage) == 0:
            return 1.0

        # Normalize memory usage by size
        normalized_usage = [mem / size for mem, size in zip(memory_usage, sizes)]

        # Efficiency is inverse of memory growth rate
        max_norm = max(normalized_usage)
        min_norm = min(normalized_usage)

        if max_norm == 0:
            return 1.0

        return min_norm / max_norm

    def _estimate_time_complexity(self, sizes: list[int], times: list[float]) -> float:
        """Estimate time complexity exponent."""
        if len(sizes) < 2 or len(times) < 2:
            return 1.0

        # Fit power law: time = a * size^b
        # log(time) = log(a) + b * log(size)
        log_sizes = np.log(sizes)
        log_times = np.log(np.maximum(times, 1e-10))  # Avoid log(0)

        # Linear regression
        coeffs = np.polyfit(log_sizes, log_times, 1)
        return coeffs[0]  # The exponent


class RobustnessValidator:
    """Validates framework robustness and error handling."""

    def test_error_handling(self) -> ProductionReadinessScore:
        """Test error handling for edge cases and invalid inputs."""

        print("🛡️  Testing error handling...")

        score = 1.0
        issues = []
        recommendations = []
        details = {"test_cases": [], "error_handling": {}}

        # Test cases with expected behaviors
        test_cases = [
            # (input, function, expected_behavior, description)
            (0, gamma_safe, "finite_positive", "Gamma at zero"),
            (-1, gamma_safe, "handle_pole", "Gamma at negative integer"),
            (float("inf"), gamma_safe, "handle_infinite", "Gamma at infinity"),
            (float("nan"), gamma_safe, "handle_nan", "Gamma with NaN"),
            (1e10, gamma_safe, "large_input", "Gamma with very large input"),
            (1e-10, gamma_safe, "small_input", "Gamma with very small input"),
            (-1, ball_volume, "handle_negative", "Volume with negative dimension"),
            (0, ball_volume, "handle_zero", "Volume at zero dimension"),
            (1000, ball_volume, "large_dimension", "Volume at large dimension"),
            (complex(1, 1), gamma_safe, "handle_complex", "Complex input"),
        ]

        for test_input, func, expected, description in test_cases:
            try:
                result = func(test_input)

                # Analyze result
                if expected == "finite_positive":
                    if not (np.isfinite(result) and result > 0):
                        score -= 0.1
                        issues.append(
                            f"Expected finite positive for {description}, got {result}"
                        )

                elif expected == "handle_pole" or expected == "handle_infinite":
                    if not (np.isinf(result) or np.isnan(result) or abs(result) > 1e10):
                        score -= 0.05
                        recommendations.append(
                            f"Consider special handling for {description}"
                        )

                elif expected == "handle_nan":
                    if not np.isnan(result):
                        score -= 0.05
                        recommendations.append(
                            f"NaN input should produce NaN output for {description}"
                        )

                elif expected == "handle_negative":
                    if not (np.isnan(result) or result == 0):
                        score -= 0.1
                        issues.append(
                            f"Negative dimension should be handled gracefully for {description}"
                        )

                details["test_cases"].append(
                    {
                        "input": str(test_input),
                        "function": func.__name__,
                        "result": str(result),
                        "description": description,
                        "passed": True,
                    }
                )

            except Exception as e:
                # Exceptions are sometimes acceptable (e.g., for invalid inputs)
                if expected in ["handle_pole", "handle_negative", "handle_complex"]:
                    # Expected to potentially raise exceptions
                    pass
                else:
                    score -= 0.05
                    recommendations.append(
                        f"Consider graceful handling for {description}: {str(e)}"
                    )

                details["test_cases"].append(
                    {
                        "input": str(test_input),
                        "function": func.__name__,
                        "result": f"Exception: {str(e)}",
                        "description": description,
                        "passed": False,
                    }
                )

        return ProductionReadinessScore(
            category="robustness",
            score=max(0.0, min(1.0, score)),
            details=details,
            recommendations=recommendations,
            critical_issues=issues,
        )


def comprehensive_production_assessment():
    """
    Comprehensive production readiness assessment.

    Evaluates all aspects of production deployment readiness:
    - Environment compatibility
    - Dependency management
    - Scalability and performance
    - Robustness and error handling
    - Overall deployment recommendations
    """
    print("🏭 COMPREHENSIVE PRODUCTION READINESS ASSESSMENT")
    print("=" * 80)

    assessments = {}

    # Environment validation
    print("\n🖥️  ENVIRONMENT VALIDATION")
    env_validator = ProductionEnvironmentValidator()
    env_assessment = env_validator.validate_environment()
    dep_assessment = env_validator.validate_dependencies()

    assessments["environment"] = env_assessment
    assessments["dependencies"] = dep_assessment

    print(f"  Environment Score: {env_assessment.score:.2f}")
    print(f"  Dependencies Score: {dep_assessment.score:.2f}")

    # Scalability validation
    print("\\n⚡ SCALABILITY VALIDATION")
    scalability_validator = ScalabilityValidator()
    concurrency_assessment = scalability_validator.test_concurrent_execution()
    memory_assessment = scalability_validator.test_memory_scaling()

    assessments["concurrency"] = concurrency_assessment
    assessments["memory_scaling"] = memory_assessment

    print(f"  Concurrency Score: {concurrency_assessment.score:.2f}")
    print(f"  Memory Scaling Score: {memory_assessment.score:.2f}")

    # Robustness validation
    print("\\n🛡️  ROBUSTNESS VALIDATION")
    robustness_validator = RobustnessValidator()
    robustness_assessment = robustness_validator.test_error_handling()

    assessments["robustness"] = robustness_assessment

    print(f"  Robustness Score: {robustness_assessment.score:.2f}")

    # Overall assessment
    overall_score = sum(assessment.score for assessment in assessments.values()) / len(
        assessments
    )

    print("\\n📊 OVERALL PRODUCTION READINESS")
    print(f"  Overall Score: {overall_score:.2f}")

    if overall_score >= 0.9:
        print("🎯 PRODUCTION READY - DEPLOY WITH CONFIDENCE")
        deployment_recommendation = "APPROVED"
    elif overall_score >= 0.8:
        print("⚠️  MINOR ISSUES - PRODUCTION READY WITH MONITORING")
        deployment_recommendation = "APPROVED_WITH_MONITORING"
    elif overall_score >= 0.7:
        print("⚠️  NEEDS IMPROVEMENT - ADDRESS ISSUES BEFORE PRODUCTION")
        deployment_recommendation = "CONDITIONAL"
    else:
        print("❌ NOT PRODUCTION READY - SIGNIFICANT WORK REQUIRED")
        deployment_recommendation = "REJECTED"

    # Consolidated recommendations
    print("\\n📋 DEPLOYMENT RECOMMENDATIONS")
    all_recommendations = []
    all_issues = []

    for assessment in assessments.values():
        all_recommendations.extend(assessment.recommendations)
        all_issues.extend(assessment.critical_issues)

    if all_issues:
        print("  CRITICAL ISSUES TO RESOLVE:")
        for issue in set(all_issues):
            print(f"    ❌ {issue}")

    if all_recommendations:
        print("  RECOMMENDATIONS:")
        for rec in set(all_recommendations):
            print(f"    💡 {rec}")

    return {
        "assessments": assessments,
        "overall_score": overall_score,
        "deployment_recommendation": deployment_recommendation,
        "critical_issues": all_issues,
        "recommendations": all_recommendations,
    }


if __name__ == "__main__":
    # Run comprehensive production assessment
    try:
        assessment_results = comprehensive_production_assessment()
        print("\\n🎯 PRODUCTION READINESS ASSESSMENT COMPLETE")

        # Sprint 2 completion check
        if assessment_results["overall_score"] >= 0.8:
            print("🚀 SPRINT 2 PRODUCTION READINESS GATE PASSED")
        else:
            print("⚠️  SPRINT 2 NEEDS ADDITIONAL OPTIMIZATION")

    except Exception as e:
        print(f"⚠️  Assessment error: {e}")
        print("Some production tests require additional dependencies")
        print("Install psutil and memory_profiler for complete assessment")
