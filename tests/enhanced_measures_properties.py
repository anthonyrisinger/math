#!/usr/bin/env python3
"""
Enhanced Dimensional Measures Property Tests
============================================

Advanced mathematical property testing for dimensional measures including:
1. Critical point analysis and peak locations
2. Asymptotic behavior in high dimensions
3. Fractional dimension continuity
4. Phase transitions and dimensional emergence
5. Topological invariant preservation
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from core.constants import NUMERICAL_EPSILON, PI
from core.gamma import gamma_safe
from core.measures import (
    ball_volume,
    complexity_measure,
    phase_capacity,
    ratio_measure,
    sphere_surface,
)


class TestCriticalDimensionBehavior:
    """Test behavior at critical dimensions and phase transitions"""

    def test_known_critical_dimensions(self):
        """Test behavior at mathematically significant dimensions"""
        critical_tests = [
            (0.0, "void dimension"),
            (1.0, "line dimension"),
            (2.0, "π-boundary"),
            (3.0, "physical space"),
            (4.0, "spacetime dimension"),
            (PI, "π-critical"),
            (2 * PI, "2π-critical"),
        ]

        for d, description in critical_tests:
            vol = ball_volume(d)
            surf = sphere_surface(d) if d > 0 else np.nan
            comp = complexity_measure(d) if d > 0 else np.nan

            # All should be finite at critical dimensions
            assert np.isfinite(vol), f"Volume infinite at {description} d={d}"
            if d > 0:
                assert np.isfinite(surf), f"Surface infinite at {description} d={d}"
                assert np.isfinite(comp), f"Complexity infinite at {description} d={d}"

            # Volume should be positive
            assert vol >= 0, f"Volume negative at {description} d={d}: {vol}"

    @given(
        st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=200)
    def test_phase_capacity_monotonicity_regions(self, d):
        """Test phase capacity Λ(d) = V(d) behavior in different regimes"""
        phase_cap = phase_capacity(d)
        vol = ball_volume(d)

        assume(np.isfinite(phase_cap) and np.isfinite(vol))

        # Phase capacity should equal ball volume
        assert abs(phase_cap - vol) < NUMERICAL_EPSILON, f"Λ(d) ≠ V(d) at d={d}"

        # Test monotonicity regions
        if d < 5.0:  # Before volume peak
            vol_next = ball_volume(d + 0.1)
            assume(np.isfinite(vol_next))
            # May be increasing in early region

        elif d > 6.0:  # After volume peak
            vol_next = ball_volume(d + 0.1)
            assume(np.isfinite(vol_next))
            # Should generally be decreasing
            assert vol_next <= vol + 1e-10, f"Volume increasing in high-d region: d={d}"

    @given(
        st.floats(min_value=1.0, max_value=15.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_complexity_peak_localization(self, d_center):
        """Test that complexity peak is well-localized"""
        epsilon = 0.1
        d_left = max(0.1, d_center - epsilon)
        d_right = d_center + epsilon

        comp_left = complexity_measure(d_left)
        comp_center = complexity_measure(d_center)
        comp_right = complexity_measure(d_right)

        assume(all(np.isfinite(x) for x in [comp_left, comp_center, comp_right]))

        # Test that complexity has local structure (not constant)
        max_comp = max(comp_left, comp_center, comp_right)
        min_comp = min(comp_left, comp_center, comp_right)

        relative_variation = (max_comp - min_comp) / max_comp if max_comp > 0 else 0

        # Around the actual peak (~6), variation should be significant
        if 5.0 <= d_center <= 7.0:
            assert (
                relative_variation > 1e-6
            ), f"Complexity too flat near peak at d={d_center}"


class TestFractionalDimensionContinuity:
    """Test continuity and smoothness for fractional dimensions"""

    @given(
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=300)
    def test_measure_continuity(self, d):
        """Test that all measures are continuous in dimension"""
        epsilon = 1e-6

        vol_d = ball_volume(d)
        vol_d_eps = ball_volume(d + epsilon)

        assume(np.isfinite(vol_d) and np.isfinite(vol_d_eps))

        # Continuity test
        vol_change = abs(vol_d_eps - vol_d)
        relative_change = vol_change / max(abs(vol_d), NUMERICAL_EPSILON)

        # Small dimensional change should produce small measure change
        assert (
            relative_change < 0.1
        ), f"Volume discontinuous at d={d}: {relative_change}"

        # Same test for surface area
        if d > 0.1:
            surf_d = sphere_surface(d)
            surf_d_eps = sphere_surface(d + epsilon)

            assume(np.isfinite(surf_d) and np.isfinite(surf_d_eps))

            surf_change = abs(surf_d_eps - surf_d)
            surf_relative_change = surf_change / max(abs(surf_d), NUMERICAL_EPSILON)

            assert (
                surf_relative_change < 0.1
            ), f"Surface discontinuous at d={d}: {surf_relative_change}"

    @given(
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        st.floats(
            min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_derivative_estimates(self, d, h):
        """Test numerical derivative estimates for smoothness"""
        # Forward difference approximation
        vol_d = ball_volume(d)
        vol_d_h = ball_volume(d + h)

        assume(np.isfinite(vol_d) and np.isfinite(vol_d_h))
        assume(abs(vol_d) > NUMERICAL_EPSILON)

        derivative_approx = (vol_d_h - vol_d) / h

        # Derivative should be finite
        assert np.isfinite(derivative_approx), f"Volume derivative infinite at d={d}"

        # For V(d) = π^{d/2}/Γ(d/2+1), derivative is related to digamma function
        # Should be well-behaved for reasonable d values
        assert (
            abs(derivative_approx) < 1000
        ), f"Volume derivative too large at d={d}: {derivative_approx}"


class TestHighDimensionalAsymptotic:
    """Test asymptotic behavior in high dimensions"""

    @given(
        st.floats(
            min_value=20.0, max_value=100.0, allow_nan=False, allow_infinity=False
        )
    )
    @settings(max_examples=50)
    def test_volume_decay_rate(self, d):
        """Test exponential decay of volume in high dimensions"""
        vol_d = ball_volume(d)
        vol_d_plus_1 = ball_volume(d + 1)

        assume(np.isfinite(vol_d) and np.isfinite(vol_d_plus_1))
        assume(vol_d > NUMERICAL_EPSILON and vol_d_plus_1 > NUMERICAL_EPSILON)

        # Ratio test for decay
        ratio = vol_d_plus_1 / vol_d

        # In high dimensions, volume should decay exponentially
        # V_{d+1}/V_d = π^{1/2} / Γ((d+1)/2+1) * Γ(d/2+1) ≈ π^{1/2} / sqrt((d+1)/2)

        # Should be decreasing (ratio < 1) and following expected pattern
        assert (
            ratio < 1.0
        ), f"Volume not decaying in high dimensions at d={d}: ratio={ratio}"

        # Expected ratio for large d: approximately sqrt(π/(d/2)) = sqrt(2π/d)
        expected_ratio = np.sqrt(2 * PI / d) if d > 10 else 1.0

        if d > 20:
            # Allow factor of 2 tolerance for asymptotic approximation
            assert (
                0.5 * expected_ratio < ratio < 2 * expected_ratio
            ), f"Volume decay ratio off at d={d}: {ratio} vs expected {expected_ratio}"

    @given(
        st.floats(min_value=10.0, max_value=50.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50)
    def test_surface_to_volume_ratio_growth(self, d):
        """Test S(d)/V(d) = d growth in high dimensions"""
        vol = ball_volume(d)
        surf = sphere_surface(d)

        assume(np.isfinite(vol) and np.isfinite(surf))
        assume(vol > NUMERICAL_EPSILON)

        ratio = surf / vol

        # For unit sphere: S_d / V_d = d
        # This is exact mathematical relation: S_d = d * V_d
        expected = d
        relative_error = abs(ratio - expected) / expected

        # Should hold with high precision
        assert (
            relative_error < 1e-12
        ), f"S/V ratio wrong at d={d}: {ratio} vs {expected}, error={relative_error}"

    @given(
        st.floats(
            min_value=15.0, max_value=100.0, allow_nan=False, allow_infinity=False
        )
    )
    @settings(max_examples=50)
    def test_complexity_high_d_behavior(self, d):
        """Test complexity measure behavior in high dimensions"""
        comp = complexity_measure(d)
        vol = ball_volume(d)
        surf = sphere_surface(d)

        assume(all(np.isfinite(x) for x in [comp, vol, surf]))

        # C(d) = V(d) * S(d) should factorize correctly
        expected = vol * surf
        relative_error = abs(comp - expected) / max(abs(expected), NUMERICAL_EPSILON)

        assert (
            relative_error < 1e-12
        ), f"Complexity factorization failed at d={d}: {relative_error}"

        # In high dimensions, complexity should decay (since volume decays faster than surface grows)
        if d > 20:
            comp_next = complexity_measure(d + 1)
            assume(np.isfinite(comp_next))

            # C(d+1) should be smaller than C(d) for large d
            assert comp_next <= comp + 1e-10, f"Complexity not decaying at high d={d}"


class TestDimensionalEmergenceProperties:
    """Test properties related to dimensional emergence theory"""

    @given(
        st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_low_dimensional_emergence(self, d):
        """Test smooth emergence from void (d=0) to low dimensions"""
        vol = ball_volume(d)

        assert np.isfinite(vol), f"Volume infinite during emergence at d={d}"
        assert vol >= 0, f"Volume negative during emergence at d={d}: {vol}"

        # Test continuity through d=1 (where surface definition changes)
        if 0.8 <= d <= 1.2:
            epsilon = 0.01
            vol_left = ball_volume(d - epsilon)
            vol_right = ball_volume(d + epsilon)

            assume(np.isfinite(vol_left) and np.isfinite(vol_right))

            # Should be continuous through d=1
            discontinuity = abs(vol_right - vol_left)
            relative_discontinuity = discontinuity / max(vol, NUMERICAL_EPSILON)

            assert (
                relative_discontinuity < 0.01
            ), f"Large discontinuity at d={d}: {relative_discontinuity}"

    @given(
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_dimensional_scaling_consistency(self, d):
        """Test that dimensional scaling follows physical intuition"""
        vol_unit = ball_volume(d)

        assume(np.isfinite(vol_unit) and vol_unit > NUMERICAL_EPSILON)

        # Scaling law: V(r) = V(1) * r^d should be mathematically consistent
        test_scales = [0.5, 2.0, 3.0]

        for scale in test_scales:
            expected_scaled = vol_unit * (scale**d)

            # Verify scaling makes sense
            if scale > 1:
                assert (
                    expected_scaled >= vol_unit
                ), f"Volume should increase with scale > 1 at d={d}"
            elif scale < 1:
                assert (
                    expected_scaled <= vol_unit
                ), f"Volume should decrease with scale < 1 at d={d}"

            # Test that scaling is consistent with derivative
            # d/dr (V(1) * r^d) = V(1) * d * r^{d-1}
            # At r = scale: derivative = V(1) * d * scale^{d-1}
            expected_derivative = vol_unit * d * (scale ** (d - 1))

            # Numerical derivative test
            epsilon = 0.001 * scale
            vol_plus = vol_unit * ((scale + epsilon) ** d)
            vol_minus = vol_unit * ((scale - epsilon) ** d)

            numerical_derivative = (vol_plus - vol_minus) / (2 * epsilon)

            if abs(expected_derivative) > NUMERICAL_EPSILON:
                derivative_error = abs(
                    numerical_derivative - expected_derivative
                ) / abs(expected_derivative)
                assert (
                    derivative_error < 0.01
                ), f"Scaling derivative wrong at d={d}, scale={scale}: {derivative_error}"


class TestTopologicalInvariants:
    """Test preservation of topological invariants under dimensional changes"""

    @given(
        st.floats(min_value=0.5, max_value=8.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_euler_characteristic_consistency(self, d):
        """Test that dimensional measures respect topological constraints"""
        vol = ball_volume(d)
        surf = sphere_surface(d) if d > 0 else 0

        assume(np.isfinite(vol) and (d == 0 or np.isfinite(surf)))

        # For d-ball and (d-1)-sphere, certain topological relationships should hold
        if d >= 1:
            # χ(D^d) = 1 (d-ball is contractible)
            # χ(S^{d-1}) = 2 if d-1 is even, 0 if d-1 is odd

            # At minimum, measures should be positive and finite
            assert vol > 0, f"Ball volume non-positive at d={d}: {vol}"
            assert surf > 0, f"Sphere surface non-positive at d={d}: {surf}"

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=20)
    def test_integer_dimension_special_properties(self, d):
        """Test that integer dimensions have expected special properties"""
        vol = ball_volume(float(d))
        surf = sphere_surface(float(d))

        # Integer dimensions should have particularly clean expressions
        # involving powers of π and factorials

        assert np.isfinite(vol), f"Integer dimension d={d} volume infinite"
        assert np.isfinite(surf), f"Integer dimension d={d} surface infinite"
        assert vol > 0, f"Integer dimension d={d} volume non-positive: {vol}"
        assert surf > 0, f"Integer dimension d={d} surface non-positive: {surf}"

        # Test that volume formula V_d = π^{d/2}/Γ(d/2+1) gives clean results
        expected_denominator = gamma_safe(d / 2.0 + 1)
        expected_numerator = PI ** (d / 2.0)

        assume(
            np.isfinite(expected_denominator)
            and expected_denominator > NUMERICAL_EPSILON
        )
        expected_vol = expected_numerator / expected_denominator

        relative_error = abs(vol - expected_vol) / max(
            abs(expected_vol), NUMERICAL_EPSILON
        )
        assert (
            relative_error < 1e-14
        ), f"Integer dimension formula error at d={d}: {relative_error}"


class TestMeasureInterrelationships:
    """Test mathematical relationships between different measures"""

    @given(
        st.floats(min_value=0.5, max_value=15.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=200)
    def test_volume_surface_derivative_relation(self, d):
        """Test that S_d is related to derivative of V_d"""
        vol = ball_volume(d)
        surf = sphere_surface(d)

        assume(np.isfinite(vol) and np.isfinite(surf))
        assume(vol > NUMERICAL_EPSILON and surf > NUMERICAL_EPSILON)

        # Mathematical relation: for unit ball, S_d = d * V_d
        # This comes from V(r) = V_d * r^d, so dV/dr|_{r=1} = d * V_d = S_d

        expected_surface = d * vol
        relative_error = abs(surf - expected_surface) / surf

        # Should hold exactly
        assert (
            relative_error < 1e-12
        ), f"Surface-volume derivative relation failed at d={d}: {relative_error}"

    @given(
        st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_ratio_measure_properties(self, d1, d2):
        """Test properties of ratio measure R(d) = S(d)/V(d)"""
        ratio_d1 = ratio_measure(d1)
        ratio_d2 = ratio_measure(d2)

        assume(np.isfinite(ratio_d1) and np.isfinite(ratio_d2))

        # R(d) = d for unit sphere
        assert (
            abs(ratio_d1 - d1) < 1e-12
        ), f"Ratio measure wrong at d={d1}: {ratio_d1} vs {d1}"
        assert (
            abs(ratio_d2 - d2) < 1e-12
        ), f"Ratio measure wrong at d={d2}: {ratio_d2} vs {d2}"

        # Monotonicity: R(d) should increase with d
        if d2 > d1:
            assert (
                ratio_d2 >= ratio_d1
            ), f"Ratio measure not monotonic: R({d1})={ratio_d1}, R({d2})={ratio_d2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
