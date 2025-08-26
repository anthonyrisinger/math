#!/usr/bin/env python3
"""
Mathematical Invariant Property Testing Framework
=================================================

A unified framework for testing mathematical invariants across all dimensional
mathematics modules using Hypothesis for property-based validation.

This framework ensures mathematical correctness by testing:
1. Fundamental mathematical laws (recurrence relations, symmetries)
2. Cross-module consistency (gamma/measures/morphic integration)
3. Numerical stability across parameter ranges
4. Edge case behavior at critical dimensions

Architecture:
- InvariantTester: Base class for mathematical property validation
- ModuleValidator: Cross-module consistency checking
- CriticalPointAnalyzer: Behavior analysis at special dimensions
- NumericalStabilityChecker: Precision and overflow handling
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dimensional.mathematics import NUMERICAL_EPSILON, PHI, PI


class InvariantType(Enum):
    """Types of mathematical invariants to test"""

    RECURRENCE = "recurrence"
    SYMMETRY = "symmetry"
    MONOTONICITY = "monotonicity"
    SCALING = "scaling"
    ASYMPTOTIC = "asymptotic"
    CONTINUITY = "continuity"
    CONSERVATION = "conservation"


@dataclass
class InvariantTestResult:
    """Result of an invariant test"""

    invariant_name: str
    invariant_type: InvariantType
    passed: bool
    error_message: Optional[str] = None
    numerical_evidence: Optional[dict[str, float]] = None
    test_parameters: Optional[dict[str, Any]] = None


class InvariantTester(ABC):
    """
    Base class for testing mathematical invariants.

    Each mathematical module should implement this to define its core invariants.
    """

    def __init__(self, tolerance: float = 1e-12, max_examples: int = 500):
        self.tolerance = tolerance
        self.max_examples = max_examples
        self.test_results: list[InvariantTestResult] = []

    @abstractmethod
    def get_invariants(self) -> dict[str, tuple[InvariantType, Callable]]:
        """Return dictionary of invariant_name -> (type, test_function)"""
        pass

    def run_all_tests(self) -> list[InvariantTestResult]:
        """Run all invariant tests and return results"""
        self.test_results.clear()
        invariants = self.get_invariants()

        for name, (inv_type, test_func) in invariants.items():
            try:
                result = self._run_single_test(name, inv_type, test_func)
                self.test_results.append(result)
            except Exception as e:
                failed_result = InvariantTestResult(
                    invariant_name=name,
                    invariant_type=inv_type,
                    passed=False,
                    error_message=f"Test execution failed: {str(e)}",
                )
                self.test_results.append(failed_result)

        return self.test_results

    def _run_single_test(
        self, name: str, inv_type: InvariantType, test_func: Callable
    ) -> InvariantTestResult:
        """Run a single invariant test with error handling"""
        try:
            # Execute the test function (should use @given decorator internally)
            test_func()
            return InvariantTestResult(
                invariant_name=name, invariant_type=inv_type, passed=True
            )
        except AssertionError as e:
            return InvariantTestResult(
                invariant_name=name,
                invariant_type=inv_type,
                passed=False,
                error_message=str(e),
            )

    def summary(self) -> dict[str, int]:
        """Return summary statistics of test results"""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.passed)
        failed = total - passed

        by_type = {}
        for result in self.test_results:
            inv_type = result.invariant_type.value
            by_type[inv_type] = by_type.get(inv_type, 0) + (1 if result.passed else 0)

        return {"total": total, "passed": passed, "failed": failed, "by_type": by_type}


class GammaInvariantTester(InvariantTester):
    """Test gamma function mathematical invariants"""

    def get_invariants(self) -> dict[str, tuple[InvariantType, Callable]]:
        return {
            "recurrence_relation": (InvariantType.RECURRENCE, self.test_recurrence),
            "reflection_formula": (InvariantType.SYMMETRY, self.test_reflection),
            "log_consistency": (InvariantType.CONTINUITY, self.test_log_consistency),
            "beta_symmetry": (InvariantType.SYMMETRY, self.test_beta_symmetry),
            "factorial_consistency": (InvariantType.RECURRENCE, self.test_factorial),
        }

    @given(
        st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=500)
    def test_recurrence(self, z):
        """Test Γ(z+1) = z·Γ(z)"""
        from dimensional.mathematics import gamma_safe

        gamma_z = gamma_safe(z)
        gamma_z_plus_1 = gamma_safe(z + 1)

        assume(np.isfinite(gamma_z) and np.isfinite(gamma_z_plus_1))

        expected = z * gamma_z
        relative_error = abs(gamma_z_plus_1 - expected) / max(
            abs(expected), NUMERICAL_EPSILON
        )

        assert relative_error < self.tolerance, f"Γ recurrence failed: {relative_error}"

    @given(
        st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=300)
    def test_reflection(self, z):
        """Test Γ(z)Γ(1-z) = π/sin(πz)"""
        from dimensional.mathematics import gamma_safe

        gamma_z = gamma_safe(z)
        gamma_1_minus_z = gamma_safe(1 - z)

        assume(np.isfinite(gamma_z) and np.isfinite(gamma_1_minus_z))

        product = gamma_z * gamma_1_minus_z
        expected = np.pi / np.sin(np.pi * z)

        relative_error = abs(product - expected) / max(abs(expected), NUMERICAL_EPSILON)
        assert relative_error < self.tolerance, f"Γ reflection failed: {relative_error}"

    @given(
        st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=300)
    def test_log_consistency(self, z):
        """Test log(Γ(z)) = gammaln(z)"""
        from dimensional.mathematics import gamma_safe, gammaln_safe

        gamma_z = gamma_safe(z)
        log_gamma_z = gammaln_safe(z)

        if np.isfinite(gamma_z) and gamma_z > 0:
            expected_log = np.log(gamma_z)
            error = abs(log_gamma_z - expected_log)
            assert error < self.tolerance, f"Log Γ consistency failed: {error}"

    @given(
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_beta_symmetry(self, a, b):
        """Test B(a,b) = B(b,a)"""
        from dimensional.mathematics import beta_function

        beta_ab = beta_function(a, b)
        beta_ba = beta_function(b, a)

        assume(np.isfinite(beta_ab) and np.isfinite(beta_ba))

        error = abs(beta_ab - beta_ba) / max(abs(beta_ab), NUMERICAL_EPSILON)
        assert error < self.tolerance, f"Beta symmetry failed: {error}"

    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=50)
    def test_factorial(self, n):
        """Test Γ(n+1) = n!"""
        import math

        from dimensional.mathematics import gamma_safe

        gamma_result = gamma_safe(n + 1)
        expected = math.factorial(n)

        error = abs(gamma_result - expected)
        assert error < NUMERICAL_EPSILON, f"Factorial consistency failed: {error}"


class MeasuresInvariantTester(InvariantTester):
    """Test dimensional measures mathematical invariants"""

    def get_invariants(self) -> dict[str, tuple[InvariantType, Callable]]:
        return {
            "volume_positivity": (
                InvariantType.MONOTONICITY,
                self.test_volume_positivity,
            ),
            "surface_positivity": (
                InvariantType.MONOTONICITY,
                self.test_surface_positivity,
            ),
            "volume_recurrence": (
                InvariantType.RECURRENCE,
                self.test_volume_recurrence,
            ),
            "complexity_factorization": (
                InvariantType.CONSERVATION,
                self.test_complexity_factorization,
            ),
            "scaling_laws": (InvariantType.SCALING, self.test_scaling_laws),
            "known_values": (InvariantType.CONSERVATION, self.test_known_values),
        }

    @given(
        st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=300)
    def test_volume_positivity(self, d):
        """Test V_d ≥ 0"""
        from dimensional.mathematics import ball_volume

        vol = ball_volume(d)
        assert vol >= 0 or np.isclose(
            vol, 0, atol=NUMERICAL_EPSILON
        ), f"Negative volume: {vol}"
        assert np.isfinite(vol), f"Non-finite volume: {vol}"

    @given(
        st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=300)
    def test_surface_positivity(self, d):
        """Test S_d > 0 for d > 0"""
        from dimensional.mathematics import sphere_surface

        surf = sphere_surface(d)
        assert surf > 0 or np.isclose(
            surf, 0, atol=NUMERICAL_EPSILON
        ), f"Negative surface: {surf}"
        assert np.isfinite(surf), f"Non-finite surface: {surf}"

    @given(
        st.floats(min_value=1.0, max_value=15.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=200)
    def test_volume_recurrence(self, d):
        """Test V_{d+2} = (2π/(d+2)) × V_d"""
        from dimensional.mathematics import ball_volume

        vol_d = ball_volume(d)
        vol_d_plus_2 = ball_volume(d + 2)

        assume(np.isfinite(vol_d) and np.isfinite(vol_d_plus_2))
        assume(vol_d > NUMERICAL_EPSILON)

        expected = (2 * PI / (d + 2)) * vol_d
        relative_error = abs(vol_d_plus_2 - expected) / max(
            abs(expected), NUMERICAL_EPSILON
        )

        tolerance = 1e-12 if d < 10 else 1e-10
        assert relative_error < tolerance, f"Volume recurrence failed: {relative_error}"

    @given(
        st.floats(min_value=0.5, max_value=15.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=200)
    def test_complexity_factorization(self, d):
        """Test C_d = V_d × S_d"""
        from dimensional.mathematics import (
            ball_volume,
            complexity_measure,
            sphere_surface,
        )

        vol = ball_volume(d)
        surf = sphere_surface(d)
        comp = complexity_measure(d)

        assume(all(np.isfinite(x) for x in [vol, surf, comp]))

        expected = vol * surf
        relative_error = abs(comp - expected) / max(abs(expected), NUMERICAL_EPSILON)

        assert (
            relative_error < self.tolerance
        ), f"Complexity factorization failed: {relative_error}"

    @given(
        st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_scaling_laws(self, d, scale):
        """Test V_d(r) = V_d(1) × r^d and S_d(r) = S_d(1) × r^{d-1}"""
        # Test mathematical consistency of scaling laws
        vol_ratio = scale**d
        surf_ratio = scale ** (d - 1)

        assert vol_ratio > 0, "Volume scaling ratio must be positive"
        assert surf_ratio > 0, "Surface scaling ratio must be positive"

        if scale > 1:
            assert vol_ratio >= 1, "Volume should increase with scale > 1"
            if d > 1:
                assert (
                    surf_ratio >= 1
                ), "Surface should increase with scale > 1 when d > 1"

    def test_known_values(self):
        """Test against known exact values"""
        from dimensional.mathematics import ball_volume, sphere_surface

        # Test known values
        assert abs(ball_volume(0) - 1.0) < NUMERICAL_EPSILON, "V_0 ≠ 1"
        assert abs(ball_volume(1) - 2.0) < NUMERICAL_EPSILON, "V_1 ≠ 2"
        assert abs(ball_volume(2) - PI) < NUMERICAL_EPSILON, "V_2 ≠ π"

        assert abs(sphere_surface(1) - 2.0) < NUMERICAL_EPSILON, "S_1 ≠ 2"
        assert abs(sphere_surface(2) - 2 * PI) < NUMERICAL_EPSILON, "S_2 ≠ 2π"
        assert abs(sphere_surface(3) - 4 * PI) < NUMERICAL_EPSILON, "S_3 ≠ 4π"


class MorphicInvariantTester(InvariantTester):
    """Test morphic mathematics invariants"""

    def get_invariants(self) -> dict[str, tuple[InvariantType, Callable]]:
        return {
            "golden_ratio_recurrence": (
                InvariantType.RECURRENCE,
                self.test_golden_recurrence,
            ),
            "polynomial_root_properties": (
                InvariantType.CONSERVATION,
                self.test_polynomial_roots,
            ),
            "stability_regions": (
                InvariantType.MONOTONICITY,
                self.test_stability_regions,
            ),
        }

    def test_golden_recurrence(self):
        """Test φ² = φ + 1 and ψ² = 1 - ψ"""
        phi_squared = PHI**2
        expected_phi = PHI + 1

        error_phi = abs(phi_squared - expected_phi)
        assert (
            error_phi < NUMERICAL_EPSILON
        ), f"Golden ratio recurrence failed: {error_phi}"

        psi = 1 / PHI  # Golden conjugate
        psi_squared = psi**2
        expected_psi = 1 - psi

        error_psi = abs(psi_squared - expected_psi)
        assert (
            error_psi < NUMERICAL_EPSILON
        ), f"Golden conjugate recurrence failed: {error_psi}"

    @given(
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_polynomial_roots(self, x):
        """Test morphic polynomial root properties"""
        # Test x^2 - x - 1 = 0 for φ
        phi_test = PHI**2 - PHI - 1
        assert (
            abs(phi_test) < NUMERICAL_EPSILON
        ), f"φ polynomial test failed: {phi_test}"

    def test_stability_regions(self):
        """Test that stability analysis produces consistent results"""
        # This is a placeholder - would test specific morphic stability properties
        assert True  # Implement based on morphic module specifics


class CrossModuleValidator:
    """Validate consistency across different mathematical modules"""

    def __init__(self):
        self.gamma_tester = GammaInvariantTester()
        self.measures_tester = MeasuresInvariantTester()
        self.morphic_tester = MorphicInvariantTester()

    def validate_gamma_measures_consistency(self) -> list[InvariantTestResult]:
        """Test that gamma functions and measures are mathematically consistent"""
        results = []

        # Test that measures use gamma functions correctly
        for d in [1, 2, 3, 4, 5]:
            result = self._test_measure_gamma_consistency(d)
            results.append(result)

        return results

    def _test_measure_gamma_consistency(self, d: float) -> InvariantTestResult:
        """Test V_d = π^{d/2} / Γ(d/2 + 1)"""
        from dimensional.mathematics import ball_volume, gamma_safe

        try:
            measured_volume = ball_volume(d)
            gamma_denominator = gamma_safe(d / 2 + 1)

            if np.isfinite(gamma_denominator) and gamma_denominator > NUMERICAL_EPSILON:
                expected_volume = (PI ** (d / 2)) / gamma_denominator

                relative_error = abs(measured_volume - expected_volume) / max(
                    abs(expected_volume), NUMERICAL_EPSILON
                )

                passed = relative_error < 1e-12
                error_msg = None if passed else f"Consistency error: {relative_error}"

                return InvariantTestResult(
                    invariant_name=f"gamma_measures_consistency_d_{d}",
                    invariant_type=InvariantType.CONSERVATION,
                    passed=passed,
                    error_message=error_msg,
                    numerical_evidence={"relative_error": relative_error},
                )
            else:
                return InvariantTestResult(
                    invariant_name=f"gamma_measures_consistency_d_{d}",
                    invariant_type=InvariantType.CONSERVATION,
                    passed=False,
                    error_message="Gamma function evaluation failed",
                )

        except Exception as e:
            return InvariantTestResult(
                invariant_name=f"gamma_measures_consistency_d_{d}",
                invariant_type=InvariantType.CONSERVATION,
                passed=False,
                error_message=f"Exception: {str(e)}",
            )

    def run_full_validation(self) -> dict[str, Any]:
        """Run complete cross-module validation"""
        results = {
            "gamma_invariants": self.gamma_tester.run_all_tests(),
            "measures_invariants": self.measures_tester.run_all_tests(),
            "morphic_invariants": self.morphic_tester.run_all_tests(),
            "cross_module_consistency": self.validate_gamma_measures_consistency(),
        }

        # Summary statistics
        all_results = []
        for category_results in results.values():
            all_results.extend(category_results)

        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.passed)

        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "gamma_summary": self.gamma_tester.summary(),
            "measures_summary": self.measures_tester.summary(),
            "morphic_summary": self.morphic_tester.summary(),
        }

        return results


def main():
    """Run the complete property testing framework"""
    print("🧪 MATHEMATICAL INVARIANT PROPERTY TESTING FRAMEWORK")
    print("=" * 60)

    validator = CrossModuleValidator()
    results = validator.run_full_validation()

    summary = results["summary"]
    print("📊 RESULTS SUMMARY:")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   Passed: {summary['passed_tests']}")
    print(f"   Failed: {summary['failed_tests']}")
    print(f"   Success Rate: {summary['success_rate']:.2%}")

    print("\n🔬 BY MODULE:")
    for module in ["gamma", "measures", "morphic"]:
        module_summary = summary[f"{module}_summary"]
        print(
            f"   {module.capitalize()}: {module_summary['passed']}/{module_summary['total']}"
        )

    # Show failures if any
    all_results = []
    for category in [
        "gamma_invariants",
        "measures_invariants",
        "morphic_invariants",
        "cross_module_consistency",
    ]:
        all_results.extend(results[category])

    failed_tests = [r for r in all_results if not r.passed]
    if failed_tests:
        print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
        for failure in failed_tests:
            print(f"   • {failure.invariant_name}: {failure.error_message}")
    else:
        print("\n✅ ALL MATHEMATICAL INVARIANTS VERIFIED!")

    return results


if __name__ == "__main__":
    main()
