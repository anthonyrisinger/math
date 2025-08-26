#!/usr/bin/env python3
"""
Mathematical Invariant Validator
================================

Validates mathematical invariants across core/ and dimensional/ packages
without requiring external dependencies like Hypothesis. Uses systematic
testing with deterministic parameter ranges to ensure mathematical correctness.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.constants import NUMERICAL_EPSILON, PI, PHI
from core.gamma import gamma_safe, gammaln_safe, beta_function
from core.measures import ball_volume, sphere_surface, complexity_measure
import dimensional as dm


class InvariantValidator:
    """Systematic validation of mathematical invariants"""

    def __init__(self, tolerance=1e-12):
        self.tolerance = tolerance
        self.passed = 0
        self.failed = 0
        self.failures = []

    def assert_close(self, actual, expected, name, context=""):
        """Assert two values are numerically close"""
        if np.isfinite(actual) and np.isfinite(expected):
            error = abs(actual - expected)
            relative_error = error / max(abs(expected), NUMERICAL_EPSILON)

            if relative_error < self.tolerance:
                self.passed += 1
                return True
            else:
                self.failed += 1
                self.failures.append(f"{name}: {actual} ≠ {expected} (error: {relative_error:.2e}) {context}")
                return False
        else:
            if np.isfinite(actual) == np.isfinite(expected):
                self.passed += 1
                return True
            else:
                self.failed += 1
                self.failures.append(f"{name}: Finite mismatch - actual: {actual}, expected: {expected} {context}")
                return False

    def validate_gamma_invariants(self):
        """Validate gamma function mathematical invariants"""
        print("🧪 Validating Gamma Function Invariants...")

        # Test recurrence relation: Γ(z+1) = z·Γ(z)
        test_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.5, 10.0]
        for z in test_values:
            gamma_z = gamma_safe(z)
            gamma_z_plus_1 = gamma_safe(z + 1)
            if np.isfinite(gamma_z) and np.isfinite(gamma_z_plus_1):
                expected = z * gamma_z
                self.assert_close(gamma_z_plus_1, expected,
                                f"Γ({z}+1) = {z}·Γ({z})", f"at z={z}")

        # Test reflection formula: Γ(z)Γ(1-z) = π/sin(πz) for 0 < z < 1
        test_values = [0.1, 0.25, 0.3, 0.5, 0.7, 0.9]
        for z in test_values:
            gamma_z = gamma_safe(z)
            gamma_1_minus_z = gamma_safe(1 - z)
            if np.isfinite(gamma_z) and np.isfinite(gamma_1_minus_z):
                product = gamma_z * gamma_1_minus_z
                expected = PI / np.sin(PI * z)
                self.assert_close(product, expected,
                                f"Γ({z})Γ(1-{z}) = π/sin(π{z})", f"at z={z}")

        # Test log consistency: log(Γ(z)) = gammaln(z)
        test_values = [0.5, 1.0, 2.0, 3.5, 10.0, 50.0]
        for z in test_values:
            gamma_z = gamma_safe(z)
            log_gamma_z = gammaln_safe(z)
            if np.isfinite(gamma_z) and gamma_z > 0:
                expected_log = np.log(gamma_z)
                self.assert_close(log_gamma_z, expected_log,
                                f"log(Γ({z})) consistency", f"at z={z}")

        # Test beta function symmetry: B(a,b) = B(b,a)
        test_pairs = [(1.0, 2.0), (0.5, 1.5), (2.5, 3.0), (0.3, 0.7)]
        for a, b in test_pairs:
            beta_ab = beta_function(a, b)
            beta_ba = beta_function(b, a)
            if np.isfinite(beta_ab) and np.isfinite(beta_ba):
                self.assert_close(beta_ab, beta_ba,
                                f"B({a},{b}) = B({b},{a})", f"symmetry test")

        print(f"   Gamma invariants: {self.passed - self.failed} tests completed")

    def validate_measures_invariants(self):
        """Validate dimensional measures invariants"""
        print("📐 Validating Dimensional Measures Invariants...")

        # Test known values
        known_tests = [
            (0, 1.0, "V_0 = 1"),
            (1, 2.0, "V_1 = 2"),
            (2, PI, "V_2 = π"),
        ]

        for d, expected, name in known_tests:
            vol = ball_volume(d)
            self.assert_close(vol, expected, name, f"known value test")

        # Test surface-volume relationship: S_d = d × V_d for unit sphere
        test_dimensions = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        for d in test_dimensions:
            vol = ball_volume(d)
            surf = sphere_surface(d)
            if np.isfinite(vol) and np.isfinite(surf) and vol > NUMERICAL_EPSILON:
                expected_surface = d * vol
                self.assert_close(surf, expected_surface,
                                f"S_{d} = {d} × V_{d}", f"at d={d}")

        # Test complexity factorization: C_d = V_d × S_d
        for d in test_dimensions:
            vol = ball_volume(d)
            surf = sphere_surface(d)
            comp = complexity_measure(d)
            if all(np.isfinite(x) for x in [vol, surf, comp]):
                expected = vol * surf
                self.assert_close(comp, expected,
                                f"C_{d} = V_{d} × S_{d}", f"at d={d}")

        # Test volume recurrence: V_{d+2} = (2π/(d+2)) × V_d
        test_base_dims = [1.0, 2.0, 3.0, 4.0, 5.0]
        for d in test_base_dims:
            vol_d = ball_volume(d)
            vol_d_plus_2 = ball_volume(d + 2)
            if np.isfinite(vol_d) and np.isfinite(vol_d_plus_2) and vol_d > NUMERICAL_EPSILON:
                expected = (2 * PI / (d + 2)) * vol_d
                self.assert_close(vol_d_plus_2, expected,
                                f"V_{{{d+2}}} = (2π/{d+2}) × V_{d}", f"recurrence test")

        print(f"   Measures invariants: tests completed")

    def validate_cross_package_consistency(self):
        """Validate consistency between core/ and dimensional/ packages"""
        print("🔄 Validating Cross-Package Consistency...")

        # Test that dimensional/ and core/ give identical results
        test_dimensions = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

        for d in test_dimensions:
            # Volume consistency
            core_vol = ball_volume(d)  # from core/
            dim_vol = dm.v(d)  # from dimensional/
            self.assert_close(dim_vol, core_vol,
                            f"dimensional.v({d}) vs core.ball_volume({d})",
                            "package consistency")

            dim_vol_upper = dm.V(d)  # uppercase alias
            self.assert_close(dim_vol_upper, core_vol,
                            f"dimensional.V({d}) vs core.ball_volume({d})",
                            "alias consistency")

            # Surface consistency
            core_surf = sphere_surface(d)
            dim_surf = dm.s(d)
            self.assert_close(dim_surf, core_surf,
                            f"dimensional.s({d}) vs core.sphere_surface({d})",
                            "package consistency")

            # Complexity consistency
            core_comp = complexity_measure(d)
            dim_comp = dm.c(d)
            self.assert_close(dim_comp, core_comp,
                            f"dimensional.c({d}) vs core.complexity_measure({d})",
                            "package consistency")

        # Test that gamma functions are consistent
        test_gamma_values = [0.5, 1.0, 2.0, 3.5, 10.0]
        for z in test_gamma_values:
            core_gamma = gamma_safe(z)  # from core/
            dim_gamma = dm.gamma_safe(z)  # from dimensional/
            self.assert_close(dim_gamma, core_gamma,
                            f"dimensional.gamma_safe({z}) vs core.gamma_safe({z})",
                            "gamma consistency")

        print(f"   Cross-package consistency: tests completed")

    def validate_morphic_invariants(self):
        """Validate morphic mathematics invariants"""
        print("🌀 Validating Morphic Mathematics Invariants...")

        # Test golden ratio properties
        phi_squared = PHI * PHI
        phi_plus_one = PHI + 1
        self.assert_close(phi_squared, phi_plus_one, "φ² = φ + 1", "golden ratio property")

        psi = 1 / PHI
        psi_squared = psi * psi
        one_minus_psi = 1 - psi
        self.assert_close(psi_squared, one_minus_psi, "ψ² = 1 - ψ", "golden conjugate property")

        # Test φ · ψ = 1
        product = PHI * psi
        self.assert_close(product, 1.0, "φ · ψ = 1", "reciprocal property")

        # Test φ + ψ = √5
        sum_phi_psi = PHI + psi
        expected = np.sqrt(5)
        self.assert_close(sum_phi_psi, expected, "φ + ψ = √5", "sum property")

        print(f"   Morphic invariants: tests completed")

    def run_all_validations(self):
        """Run complete mathematical invariant validation"""
        print("🔬 MATHEMATICAL INVARIANT VALIDATION")
        print("=" * 60)

        initial_passed = self.passed
        initial_failed = self.failed

        self.validate_gamma_invariants()
        self.validate_measures_invariants()
        self.validate_cross_package_consistency()
        self.validate_morphic_invariants()

        total_tests = self.passed + self.failed
        new_tests = total_tests - (initial_passed + initial_failed)

        print(f"\n📊 VALIDATION RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {self.passed}")
        print(f"   Failed: {self.failed}")
        print(f"   Success Rate: {self.passed/total_tests:.1%}" if total_tests > 0 else "   No tests run")

        if self.failures:
            print(f"\n❌ FAILED TESTS ({len(self.failures)}):")
            for failure in self.failures[:10]:  # Show first 10 failures
                print(f"   • {failure}")
            if len(self.failures) > 10:
                print(f"   ... and {len(self.failures) - 10} more")
        else:
            print(f"\n✅ ALL MATHEMATICAL INVARIANTS VERIFIED!")

        return self.failed == 0


def main():
    """Run mathematical invariant validation"""
    validator = InvariantValidator()
    success = validator.run_all_validations()

    if success:
        print(f"\n🎉 SPRINT 1 GATE 2: MATHEMATICAL INVARIANT VALIDATION COMPLETE")
        print(f"   Core/dimensional package consistency: ✅")
        print(f"   Mathematical properties preserved: ✅")
        print(f"   Property testing framework operational: ✅")
        return 0
    else:
        print(f"\n⚠️  MATHEMATICAL INVARIANT FAILURES DETECTED")
        print(f"   Investigate failures before proceeding to Sprint 2")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)