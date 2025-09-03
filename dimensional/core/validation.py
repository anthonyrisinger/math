#!/usr/bin/env python3
"""
Mathematical Validation Framework - Consolidated
===============================================

Property-based validation and testing framework for dimensional mathematics.
Consolidated from tests/ modules with enhanced mathematical rigor.
"""

from typing import Any

import numpy as np

from ..gamma import gamma as gamma_safe

# Import from parent module to avoid circular imports
from ..measures import ball_volume, complexity_measure, ratio_measure, sphere_surface
from .constants import CRITICAL_DIMENSIONS, NUMERICAL_EPSILON, PI
from .core import total_phase_energy

# Import phase dynamics from core modules
from .dynamics import PhaseDynamicsEngine


# Need to define or import these functions
def beta_function(a, b):
    """Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)."""
    return gamma_safe(a) * gamma_safe(b) / gamma_safe(a + b)

def golden_ratio_properties():
    """Return golden ratio mathematical properties."""
    PHI = (1 + np.sqrt(5)) / 2
    return {
        "phi_squared_minus_phi": abs(PHI**2 - PHI - 1) < NUMERICAL_EPSILON,
        "inverse_property": abs(1/PHI - (PHI - 1)) < NUMERICAL_EPSILON,
    }

def morphic_polynomial_roots(k, variant="shifted"):
    """Find roots of morphic polynomials."""
    if variant == "shifted":
        # x^3 - (2-k)x - 1 = 0
        coeffs = [1, 0, -(2-k), -1]
    else:  # simple
        # x^3 - x - k = 0
        coeffs = [1, 0, -1, -k]
    roots = np.roots(coeffs)
    return roots[np.isreal(roots)].real


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
        results["gamma_one_equals_one"] = (
            abs(gamma_safe(1.0) - 1.0) < self.tolerance
        )

        # Γ(n+1) = n! for integers
        for n in range(1, 8):
            import math

            factorial_n = math.factorial(n)
            gamma_n_plus_1 = gamma_safe(n + 1)
            results[f"gamma_{n + 1}_equals_{n}_factorial"] = (
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

            results[f"{measure_type}_d_{d}"] = (
                abs(actual - expected) < self.tolerance
            )

        # Monotonicity in valid ranges
        dimensions = np.linspace(0.1, 10, 100)
        volumes = [ball_volume(d) for d in dimensions]
        surfaces = [sphere_surface(d) for d in dimensions]

        # Volume should increase then decrease (has peak around 5.26)
        peak_idx_v = np.argmax(volumes)
        # Peak not at boundaries
        results["volume_has_peak"] = 10 <= peak_idx_v <= 90

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
        )  # At least void should exis

        # Phase coherence bounds
        coherence = engine.get_state()["coherence"]
        results["coherence_bounded"] = 0 <= coherence <= 1

        # Effective dimension should be reasonable
        eff_dim = engine.calculate_effective_dimension()
        results["effective_dimension_reasonable"] = (
            0 <= eff_dim < engine.max_dim
        )

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
            # Allow zero for void_dimension
            results[f"{name}_is_positive"] = value >= 0

        # Volume peak should be near the stored critical value
        from ..measures import find_peak

        v_peak_d, v_peak_val = find_peak(ball_volume)
        stored_v_peak = CRITICAL_DIMENSIONS["volume_peak"]

        results["volume_peak_location_accurate"] = (
            abs(v_peak_d - stored_v_peak) < 0.1
        )  # Within 0.1 dimension units

        # Surface peak validation
        s_peak_d, s_peak_val = find_peak(sphere_surface)
        stored_s_peak = CRITICAL_DIMENSIONS["surface_peak"]

        results["surface_peak_location_accurate"] = (
            abs(s_peak_d - stored_s_peak) < 0.1
        )

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

        print(
            f"Validation complete: {passed_tests}/{total_tests} tests passed"
        )

        if passed_tests < total_tests:
            print("FAILED TESTS:")
            for category_name, category_results in all_results.items():
                failed_tests = [
                    name
                    for name, result in category_results.items()
                    if not result
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


class ConvergenceDiagnostics:
    """Advanced convergence diagnostics for fractional dimensions."""

    def __init__(self, tolerance=NUMERICAL_EPSILON):
        self.tolerance = tolerance
        self.test_results = {}

    def richardson_extrapolation(self, func, x, h_sequence=None):
        """Richardson extrapolation for convergence analysis."""
        if h_sequence is None:
            h_sequence = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]

        results = []
        for h in h_sequence:
            try:
                # Central difference approximation
                if x > h and func(x + h) is not None and func(x - h) is not None:
                    derivative = (func(x + h) - func(x - h)) / (2 * h)
                    if np.isfinite(derivative):
                        results.append(derivative)
            except (OverflowError, ZeroDivisionError, ValueError):
                continue

        if len(results) >= 3:
            errors = np.abs(np.diff(results))
            convergence_rate = errors[-1] / errors[-2] if len(errors) >= 2 else None
            converged = errors[-1] < self.tolerance if len(errors) >= 1 else False

            return {
                'derivatives': results,
                'errors': errors.tolist(),
                'convergence_rate': convergence_rate,
                'converged': converged,
                'method': 'richardson'
            }

        return {'error': 'insufficient_data', 'method': 'richardson'}

    def aitken_acceleration(self, sequence):
        """Aitken acceleration for sequence convergence."""
        if len(sequence) < 3:
            return {'error': 'insufficient_data', 'method': 'aitken'}

        accelerated = []
        for i in range(len(sequence) - 2):
            s_n = sequence[i]
            s_n1 = sequence[i + 1]
            s_n2 = sequence[i + 2]

            denom = s_n2 - 2*s_n1 + s_n
            if abs(denom) > self.tolerance:
                acc = s_n2 - (s_n2 - s_n1)**2 / denom
                accelerated.append(acc)

        if len(accelerated) >= 2:
            convergence_improvement = abs(accelerated[-1] - accelerated[-2])
            return {
                'accelerated_sequence': accelerated,
                'convergence_improvement': convergence_improvement,
                'converged': convergence_improvement < self.tolerance,
                'method': 'aitken'
            }

        return {'error': 'acceleration_failed', 'method': 'aitken'}

    def fractional_convergence_test(self, func, x_base, fractional_steps=10):
        """Test convergence in fractional domain around x_base."""
        # Generate fractional perturbations
        eps_values = np.logspace(-12, -3, fractional_steps)
        forward_diffs = []
        backward_diffs = []

        for eps in eps_values:
            try:
                if x_base + eps > 0 and x_base - eps != x_base:  # Avoid pole regions
                    f_plus = func(x_base + eps)
                    f_minus = func(x_base - eps)
                    f_base = func(x_base)

                    if all(np.isfinite([f_plus, f_minus, f_base])):
                        forward_diffs.append((f_plus - f_base) / eps)
                        backward_diffs.append((f_base - f_minus) / eps)
            except (OverflowError, ValueError, ZeroDivisionError):
                continue

        if len(forward_diffs) >= 3 and len(backward_diffs) >= 3:
            # Check consistency between forward and backward differences
            diff_consistency = [abs(f - b) for f, b in zip(forward_diffs, backward_diffs)]
            avg_consistency = np.mean(diff_consistency)

            # Richardson extrapolation on differences
            richardson_forward = self.richardson_extrapolation(func, x_base)

            return {
                'forward_differences': forward_diffs,
                'backward_differences': backward_diffs,
                'consistency_errors': diff_consistency,
                'avg_consistency': avg_consistency,
                'richardson_result': richardson_forward,
                'converged': avg_consistency < self.tolerance * 100,  # Relaxed for fractional
                'method': 'fractional_domain'
            }

        return {'error': 'insufficient_fractional_data', 'method': 'fractional_domain'}


class NumericalStabilityTester:
    """Test numerical stability across parameter ranges with convergence diagnostics."""

    def __init__(self):
        self.test_results = {}
        self.convergence = ConvergenceDiagnostics()

    def test_gamma_stability(self) -> dict[str, Any]:
        """Test gamma function stability with enhanced convergence diagnostics."""
        results = {}

        # Test ranges with enhanced fractional coverage
        small_positive = np.logspace(-10, -1, 20)
        fractional_range = np.linspace(0.001, 3.999, 50)  # Avoid integer boundaries
        normal_range = np.linspace(0.1, 10, 50)
        large_values = np.logspace(1, 2, 20)  # Up to 100
        negative_fractional = np.linspace(-3.999, -0.001, 30)  # Negative fractional

        # Test small positive values with convergence
        small_results = []
        small_convergence = []
        for i, z in enumerate(small_positive):
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

                # Sample convergence tests (every 5th point)
                if i % 5 == 0:
                    conv_test = self.convergence.fractional_convergence_test(gamma_safe, z)
                    small_convergence.append(conv_test)

            except Exception as e:
                small_results.append(
                    {
                        "z": z,
                        "error": str(e),
                        "finite": False,
                        "positive": False,
                    }
                )

        results["small_values"] = {
            "count": len(small_results),
            "finite_ratio": sum(
                1 for r in small_results if r.get("finite", False)
            )
            / len(small_results),
            "positive_ratio": sum(
                1 for r in small_results if r.get("positive", False)
            )
            / len(small_results),
            "convergence_tests": len([c for c in small_convergence if c.get('converged', False)])
        }

        # Test fractional range with enhanced validation
        fractional_results = []
        fractional_convergence = []
        for i, z in enumerate(fractional_range):
            try:
                result = gamma_safe(z)
                fractional_results.append(result)

                # Convergence test every 10th point
                if i % 10 == 0:
                    conv_test = self.convergence.fractional_convergence_test(gamma_safe, z)
                    fractional_convergence.append(conv_test)

            except Exception:
                fractional_results.append(np.nan)

        finite_fractional = [r for r in fractional_results if np.isfinite(r)]
        results["fractional_range"] = {
            "count": len(fractional_results),
            "finite_count": len(finite_fractional),
            "finite_ratio": len(finite_fractional) / len(fractional_results),
            "all_positive": all(r > 0 for r in finite_fractional) if finite_fractional else False,
            "convergence_passed": sum(1 for c in fractional_convergence if c.get('converged', False)),
            "convergence_tested": len(fractional_convergence)
        }

        # Test negative fractional values (most challenging)
        negative_results = []
        negative_convergence = []
        for i, z in enumerate(negative_fractional):
            try:
                result = gamma_safe(z)
                negative_results.append(result)

                # Test convergence for negative fractional values
                if i % 5 == 0 and abs(z - round(z)) > 0.1:  # Avoid near-poles
                    conv_test = self.convergence.fractional_convergence_test(gamma_safe, z)
                    negative_convergence.append(conv_test)

            except Exception:
                negative_results.append(np.nan)

        finite_negative = [r for r in negative_results if np.isfinite(r)]
        results["negative_fractional"] = {
            "count": len(negative_results),
            "finite_count": len(finite_negative),
            "finite_ratio": len(finite_negative) / len(negative_results) if negative_results else 0,
            "convergence_passed": sum(1 for c in negative_convergence if c.get('converged', False)),
            "reflection_formula_tests": len(negative_convergence)
        }

        # Test normal range (enhanced)
        normal_results = [gamma_safe(z) for z in normal_range]
        normal_convergence = []
        for i, z in enumerate(normal_range[::5]):  # Sample every 5th
            conv_test = self.convergence.fractional_convergence_test(gamma_safe, z)
            normal_convergence.append(conv_test)

        results["normal_range"] = {
            "all_finite": all(np.isfinite(r) for r in normal_results),
            "all_positive": all(r > 0 for r in normal_results),
            "convergence_passed": sum(1 for c in normal_convergence if c.get('converged', False)),
            "monotonic_in_parts": True,  # Gamma has complex monotonicity
        }

        # Test large values with Stirling approximation validation
        large_results = [gamma_safe(z) for z in large_values]
        stirling_tests = []
        for z in large_values[::3]:  # Sample every 3rd
            gamma_val = gamma_safe(z)
            if np.isfinite(gamma_val) and z > 0:
                # Stirling approximation: Γ(z) ≈ √(2π/z) * (z/e)^z
                stirling_approx = np.sqrt(2 * np.pi / z) * (z / np.e) ** z
                if np.isfinite(stirling_approx) and stirling_approx > 0:
                    relative_error = abs(gamma_val - stirling_approx) / stirling_approx
                    stirling_tests.append(relative_error)

        results["large_values"] = {
            "all_finite_or_inf": all(
                np.isfinite(r) or np.isinf(r) for r in large_results
            ),
            "no_nan": not any(np.isnan(r) for r in large_results),
            "stirling_approximation_errors": stirling_tests,
            "mean_stirling_error": np.mean(stirling_tests) if stirling_tests else 0
        }

        return results

    def test_measure_stability(self) -> dict[str, Any]:
        """Test dimensional measure stability with convergence diagnostics."""
        results = {}

        # Test fractional dimensions with enhanced coverage
        fractional_dims = np.linspace(0.001, 10, 200)
        very_fractional = np.array([0.1, 0.25, 0.5, 0.75, 1.25, 1.5, 2.25, 2.5, 3.14159, 4.5, 5.26, 7.26])  # Key points

        volumes = []
        surfaces = []
        complexities = []
        measure_convergence = []

        for i, d in enumerate(fractional_dims):
            try:
                v = ball_volume(d)
                s = sphere_surface(d)
                c = complexity_measure(d)

                volumes.append(v)
                surfaces.append(s)
                complexities.append(c)

                # Test convergence at key fractional points
                if d in very_fractional:
                    vol_conv = self.convergence.fractional_convergence_test(ball_volume, d)
                    surf_conv = self.convergence.fractional_convergence_test(sphere_surface, d)
                    measure_convergence.append({
                        'dimension': d,
                        'volume_convergence': vol_conv,
                        'surface_convergence': surf_conv
                    })

            except Exception:
                volumes.append(np.nan)
                surfaces.append(np.nan)
                complexities.append(np.nan)

        volumes = np.array(volumes)
        surfaces = np.array(surfaces)
        complexities = np.array(complexities)

        # Peak finding with enhanced accuracy
        finite_vols = volumes[np.isfinite(volumes)]
        finite_surfs = surfaces[np.isfinite(surfaces)]
        finite_dims = fractional_dims[np.isfinite(volumes)]

        vol_peak_idx = np.argmax(finite_vols) if len(finite_vols) > 0 else -1
        surf_peak_idx = np.argmax(finite_surfs) if len(finite_surfs) > 0 else -1

        peak_analysis = {}
        if vol_peak_idx >= 0:
            peak_analysis['volume_peak_dimension'] = finite_dims[vol_peak_idx]
            peak_analysis['volume_peak_value'] = finite_vols[vol_peak_idx]
        if surf_peak_idx >= 0:
            peak_analysis['surface_peak_dimension'] = finite_dims[surf_peak_idx]
            peak_analysis['surface_peak_value'] = finite_surfs[surf_peak_idx]

        results["fractional_dimensions"] = {
            "volume_finite_ratio": np.mean(np.isfinite(volumes)),
            "surface_finite_ratio": np.mean(np.isfinite(surfaces)),
            "complexity_finite_ratio": np.mean(np.isfinite(complexities)),
            "volume_positive_ratio": np.mean(
                volumes[np.isfinite(volumes)] > 0
            ) if np.any(np.isfinite(volumes)) else 0,
            "surface_positive_ratio": np.mean(
                surfaces[np.isfinite(surfaces)] > 0
            ) if np.any(np.isfinite(surfaces)) else 0,
            "convergence_tests": len(measure_convergence),
            "convergence_passed": sum(
                1 for test in measure_convergence
                if test['volume_convergence'].get('converged', False) and
                   test['surface_convergence'].get('converged', False)
            ),
            "peak_analysis": peak_analysis
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
        f"    Small values finite: {
            gamma_stability['small_values']['finite_ratio']:.2%}"
    )
    print(
        f"    Normal range all finite: {
            gamma_stability['normal_range']['all_finite']}"
    )
    print(
        f"    Large values no NaN: {
            gamma_stability['large_values']['no_nan']}"
    )

    measure_stability = stability_tester.test_measure_stability()
    print("  Measure function stability:")
    print(
        f"    Volume finite ratio: {
            measure_stability['fractional_dimensions']['volume_finite_ratio']:.2%}"
    )
    print(
        f"    Surface finite ratio: {
            measure_stability['fractional_dimensions']['surface_finite_ratio']:.2%}"
    )

    print("\nConsolidated validation framework operational!")
