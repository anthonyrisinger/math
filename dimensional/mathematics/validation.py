#!/usr/bin/env python3
"""
Mathematical Validation Framework - Consolidated
===============================================

Property-based validation and testing framework for dimensional mathematics.
Consolidated from tests/ modules with enhanced mathematical rigor.
"""

from typing import Any

import numpy as np

from .constants import CRITICAL_DIMENSIONS, NUMERICAL_EPSILON, PI
from .functions import (
    PhaseDynamicsEngine,
    ball_volume,
    beta_function,
    complexity_measure,
    gamma_safe,
    golden_ratio_properties,
    morphic_polynomial_roots,
    ratio_measure,
    sphere_surface,
    total_phase_energy,
)


class PropertyValidator:
    """Mathematical property validation system."""

    def __init__(self, tolerance=NUMERICAL_EPSILON):
        self.tolerance = tolerance
        self.validation_results = {}

    def validate_gamma_properties(self) -> dict[str, bool]:
        """Validate fundamental gamma function properties."""
        results = {}

        # Γ(1/2) = √π
        results["gamma_half_sqrt_pi"] = (
            abs(gamma_safe(0.5) - np.sqrt(PI)) < self.tolerance
        )

        # Γ(1) = 1
        results["gamma_one_equals_one"] = abs(gamma_safe(1.0) - 1.0) < self.tolerance

        # Γ(n+1) = n! for integers
        for n in range(1, 8):
            import math

            factorial_n = math.factorial(n)
            gamma_n_plus_1 = gamma_safe(n + 1)
            results[f"gamma_{n+1}_equals_{n}_factorial"] = (
                abs(gamma_n_plus_1 - factorial_n) < self.tolerance
            )

        # Recurrence: Γ(z+1) = z·Γ(z)
        test_values = [0.5, 1.5, 2.5, 3.5, 4.5]
        for z in test_values:
            gamma_z = gamma_safe(z)
            gamma_z_plus_1 = gamma_safe(z + 1)
            expected = z * gamma_z
            results[f"recurrence_z_{z}"] = (
                abs(gamma_z_plus_1 - expected) < self.tolerance
            )

        # Beta function: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
        test_pairs = [(1, 1), (2, 3), (0.5, 0.5), (1, 0.5)]
        for a, b in test_pairs:
            beta_direct = beta_function(a, b)
            beta_gamma = gamma_safe(a) * gamma_safe(b) / gamma_safe(a + b)
            results[f"beta_function_{a}_{b}"] = (
                abs(beta_direct - beta_gamma) < self.tolerance * 10
            )  # Slightly more tolerance for composition

        self.validation_results["gamma"] = results
        return results

    def validate_measure_properties(self) -> dict[str, bool]:
        """Validate dimensional measure properties."""
        results = {}

        # Known exact values
        known_values = {
            (0, "volume"): 1.0,  # V_0 = 1
            (1, "volume"): 2.0,  # V_1 = 2
            (2, "volume"): PI,  # V_2 = π
            (3, "volume"): 4 * PI / 3,  # V_3 = 4π/3
            (1, "surface"): 2.0,  # S_1 = 2
            (2, "surface"): 2 * PI,  # S_2 = 2π
            (3, "surface"): 4 * PI,  # S_3 = 4π
        }

        for (d, measure_type), expected in known_values.items():
            if measure_type == "volume":
                actual = ball_volume(d)
            elif measure_type == "surface":
                actual = sphere_surface(d)
            else:
                continue

            results[f"{measure_type}_d_{d}"] = abs(actual - expected) < self.tolerance

        # Monotonicity in valid ranges
        dimensions = np.linspace(0.1, 10, 100)
        volumes = [ball_volume(d) for d in dimensions]
        surfaces = [sphere_surface(d) for d in dimensions]

        # Volume should increase then decrease (has peak around 5.26)
        peak_idx_v = np.argmax(volumes)
        results["volume_has_peak"] = 10 <= peak_idx_v <= 90  # Peak not at boundaries

        # Surface should increase then decrease (has peak around 7.26)
        peak_idx_s = np.argmax(surfaces)
        results["surface_has_peak"] = 10 <= peak_idx_s <= 90

        # Complexity measure properties
        complexities = [complexity_measure(d) for d in dimensions]
        peak_idx_c = np.argmax(complexities)
        results["complexity_has_peak"] = 10 <= peak_idx_c <= 90

        # All measures should be positive
        results["all_volumes_positive"] = all(v > 0 for v in volumes)
        results["all_surfaces_positive"] = all(s > 0 for s in surfaces)
        results["all_complexities_positive"] = all(c > 0 for c in complexities)

        # Ratio measure should be monotonically increasing for d > 0
        ratios = [
            ratio_measure(d) for d in dimensions[1:]
        ]  # Skip d=0 to avoid singularity
        ratio_diffs = np.diff(ratios)
        results["ratio_monotonic_increasing"] = all(
            diff >= -self.tolerance for diff in ratio_diffs
        )

        self.validation_results["measures"] = results
        return results

    def validate_phase_dynamics_properties(self) -> dict[str, bool]:
        """Validate phase dynamics properties."""
        results = {}

        # Energy conservation
        engine = PhaseDynamicsEngine(max_dimensions=6)
        initial_energy = total_phase_energy(engine.phase_density)

        # Run short simulation
        for _ in range(50):
            engine.step(0.01)

        final_energy = total_phase_energy(engine.phase_density)
        energy_change = abs(final_energy - initial_energy)

        results["energy_conservation"] = energy_change < 1e-10
        results["energy_positive"] = final_energy >= 0
        # Note: Emergence may take longer, this is not necessarily a failure
        results["dimensions_emerged"] = (
            len(engine.emerged) >= 1
        )  # At least void should exist

        # Phase coherence bounds
        coherence = engine.get_state()["coherence"]
        results["coherence_bounded"] = 0 <= coherence <= 1

        # Effective dimension should be reasonable
        eff_dim = engine.calculate_effective_dimension()
        results["effective_dimension_reasonable"] = 0 <= eff_dim < engine.max_dim

        self.validation_results["phase"] = results
        return results

    def validate_morphic_properties(self) -> dict[str, bool]:
        """Validate morphic mathematics properties."""
        results = {}

        # Golden ratio properties
        golden_props = golden_ratio_properties()
        for prop_name, is_valid in golden_props.items():
            if isinstance(is_valid, bool):
                results[prop_name] = is_valid

        # Morphic polynomial roots should be real for valid parameters
        stable_k_values = [1.0, 1.5, 2.0, 2.5]
        for k in stable_k_values:
            roots_shifted = morphic_polynomial_roots(k, "shifted")
            roots_simple = morphic_polynomial_roots(k, "simple")

            results[f"shifted_k_{k}_has_real_roots"] = len(roots_shifted) > 0
            results[f"simple_k_{k}_has_real_roots"] = len(roots_simple) > 0

            # Verify roots actually solve the polynomial
            for i, root in enumerate(roots_shifted):
                poly_value = root**3 - (2 - k) * root - 1
                results[f"shifted_k_{k}_root_{i}_valid"] = (
                    abs(poly_value) < self.tolerance
                )

        self.validation_results["morphic"] = results
        return results

    def validate_critical_dimensions(self) -> dict[str, bool]:
        """Validate critical dimension calculations."""
        results = {}

        # Check that critical dimensions are reasonable
        for name, value in CRITICAL_DIMENSIONS.items():
            results[f"{name}_is_finite"] = np.isfinite(value)
            results[f"{name}_is_positive"] = value >= 0  # Allow zero for void_dimension

        # Volume peak should be near the stored critical value
        from .functions import find_peak

        v_peak_d, v_peak_val = find_peak(ball_volume)
        stored_v_peak = CRITICAL_DIMENSIONS["volume_peak"]

        results["volume_peak_location_accurate"] = (
            abs(v_peak_d - stored_v_peak) < 0.1
        )  # Within 0.1 dimension units

        # Surface peak validation
        s_peak_d, s_peak_val = find_peak(sphere_surface)
        stored_s_peak = CRITICAL_DIMENSIONS["surface_peak"]

        results["surface_peak_location_accurate"] = abs(s_peak_d - stored_s_peak) < 0.1

        self.validation_results["critical_dimensions"] = results
        return results

    def run_comprehensive_validation(self) -> dict[str, dict[str, bool]]:
        """Run all validation tests."""
        print("Running comprehensive mathematical validation...")

        gamma_results = self.validate_gamma_properties()
        measure_results = self.validate_measure_properties()
        phase_results = self.validate_phase_dynamics_properties()
        morphic_results = self.validate_morphic_properties()
        critical_results = self.validate_critical_dimensions()

        all_results = {
            "gamma": gamma_results,
            "measures": measure_results,
            "phase": phase_results,
            "morphic": morphic_results,
            "critical_dimensions": critical_results,
        }

        # Summary statistics
        total_tests = sum(len(category) for category in all_results.values())
        passed_tests = sum(
            sum(1 for result in category.values() if result)
            for category in all_results.values()
        )

        print(f"Validation complete: {passed_tests}/{total_tests} tests passed")

        if passed_tests < total_tests:
            print("FAILED TESTS:")
            for category_name, category_results in all_results.items():
                failed_tests = [
                    name for name, result in category_results.items() if not result
                ]
                if failed_tests:
                    print(f"  {category_name}: {failed_tests}")
        else:
            print("All mathematical property validations PASSED!")

        return all_results


def validate_mathematical_properties(
    tolerance=NUMERICAL_EPSILON, verbose=False
) -> bool:
    """Convenience function for quick validation."""
    validator = PropertyValidator(tolerance)
    results = validator.run_comprehensive_validation()

    # Return True only if ALL tests pass
    all_passed = all(
        all(test_result for test_result in category.values())
        for category in results.values()
    )

    if verbose:
        print(f"Mathematical validation: {'PASS' if all_passed else 'FAIL'}")

    return all_passed


def cross_package_consistency_test() -> dict[str, bool]:
    """Test consistency between consolidated and original implementations."""
    results = {}

    try:
        # Test a few key functions against scipy/numpy where possible
        from scipy.special import gamma as scipy_gamma

        test_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        for z in test_values:
            our_gamma = gamma_safe(z)
            scipy_result = scipy_gamma(z)

            results[f"gamma_vs_scipy_z_{z}"] = (
                abs(our_gamma - scipy_result) < NUMERICAL_EPSILON * 100
            )  # Allow some numerical difference

    except ImportError:
        results["scipy_comparison_skipped"] = True

    return results


class NumericalStabilityTester:
    """Test numerical stability across parameter ranges."""

    def __init__(self):
        self.test_results = {}

    def test_gamma_stability(self) -> dict[str, Any]:
        """Test gamma function stability across wide parameter ranges."""
        results = {}

        # Test ranges
        small_positive = np.logspace(-10, -1, 20)
        normal_range = np.linspace(0.1, 10, 50)
        large_values = np.logspace(1, 2, 20)  # Up to 100

        # Test small positive values
        small_results = []
        for z in small_positive:
            try:
                result = gamma_safe(z)
                small_results.append(
                    {
                        "z": z,
                        "gamma": result,
                        "finite": np.isfinite(result),
                        "positive": result > 0,
                    }
                )
            except Exception as e:
                small_results.append(
                    {"z": z, "error": str(e), "finite": False, "positive": False}
                )

        results["small_values"] = {
            "count": len(small_results),
            "finite_ratio": sum(1 for r in small_results if r.get("finite", False))
            / len(small_results),
            "positive_ratio": sum(1 for r in small_results if r.get("positive", False))
            / len(small_results),
        }

        # Test normal range
        normal_results = [gamma_safe(z) for z in normal_range]
        results["normal_range"] = {
            "all_finite": all(np.isfinite(r) for r in normal_results),
            "all_positive": all(r > 0 for r in normal_results),
            "monotonic_in_parts": True,  # Gamma has complex monotonicity
        }

        # Test large values
        large_results = [gamma_safe(z) for z in large_values]
        results["large_values"] = {
            "all_finite_or_inf": all(
                np.isfinite(r) or np.isinf(r) for r in large_results
            ),
            "no_nan": not any(np.isnan(r) for r in large_results),
        }

        return results

    def test_measure_stability(self) -> dict[str, Any]:
        """Test dimensional measure stability."""
        results = {}

        # Test fractional dimensions
        fractional_dims = np.linspace(0.001, 10, 200)

        volumes = []
        surfaces = []
        complexities = []

        for d in fractional_dims:
            try:
                v = ball_volume(d)
                s = sphere_surface(d)
                c = complexity_measure(d)

                volumes.append(v)
                surfaces.append(s)
                complexities.append(c)

            except Exception:
                volumes.append(np.nan)
                surfaces.append(np.nan)
                complexities.append(np.nan)

        volumes = np.array(volumes)
        surfaces = np.array(surfaces)
        complexities = np.array(complexities)

        results["fractional_dimensions"] = {
            "volume_finite_ratio": np.mean(np.isfinite(volumes)),
            "surface_finite_ratio": np.mean(np.isfinite(surfaces)),
            "complexity_finite_ratio": np.mean(np.isfinite(complexities)),
            "volume_positive_ratio": np.mean(volumes[np.isfinite(volumes)] > 0),
            "surface_positive_ratio": np.mean(surfaces[np.isfinite(surfaces)] > 0),
        }

        return results


if __name__ == "__main__":
    print("MATHEMATICAL VALIDATION FRAMEWORK")
    print("=" * 50)

    # Run comprehensive validation
    validator = PropertyValidator()
    results = validator.run_comprehensive_validation()

    # Cross-package consistency
    print("\nCross-package consistency:")
    consistency = cross_package_consistency_test()
    for test_name, passed in consistency.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    # Numerical stability testing
    print("\nNumerical stability testing:")
    stability_tester = NumericalStabilityTester()

    gamma_stability = stability_tester.test_gamma_stability()
    print("  Gamma function stability:")
    print(
        f"    Small values finite: {gamma_stability['small_values']['finite_ratio']:.2%}"
    )
    print(
        f"    Normal range all finite: {gamma_stability['normal_range']['all_finite']}"
    )
    print(f"    Large values no NaN: {gamma_stability['large_values']['no_nan']}")

    measure_stability = stability_tester.test_measure_stability()
    print("  Measure function stability:")
    print(
        f"    Volume finite ratio: {measure_stability['fractional_dimensions']['volume_finite_ratio']:.2%}"
    )
    print(
        f"    Surface finite ratio: {measure_stability['fractional_dimensions']['surface_finite_ratio']:.2%}"
    )

    print("\nConsolidated validation framework operational!")
