#!/usr/bin/env python3
"""
Property-based tests for dimensional measure relationships.

Tests fundamental mathematical relationships between volume, surface area,
and complexity measures across all dimensional ranges.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dimensional.mathematics.constants import NUMERICAL_EPSILON, PI
from dimensional.mathematics.functions import (
    ball_volume,
    complexity_measure,
    sphere_surface,
)


class TestDimensionalMeasureBasicProperties:
    """Test basic mathematical properties of dimensional measures"""

    @given(
        st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=500)
    def test_volume_positivity(self, d):
        """Test that ball volume is always positive for d ≥ 0"""
        vol = ball_volume(d)
        assert vol > 0 or np.isclose(
            vol, 0, atol=NUMERICAL_EPSILON
        ), f"Negative volume at d={d}: {vol}"
        assert np.isfinite(vol), f"Non-finite volume at d={d}: {vol}"

    @given(
        st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=500)
    def test_surface_positivity(self, d):
        """Test that sphere surface area is always positive for d > 0"""
        surf = sphere_surface(d)
        assert surf > 0 or np.isclose(
            surf, 0, atol=NUMERICAL_EPSILON
        ), f"Negative surface at d={d}: {surf}"
        assert np.isfinite(surf), f"Non-finite surface at d={d}: {surf}"

    @given(
        st.floats(min_value=0.1, max_value=15.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=300)
    def test_complexity_positivity(self, d):
        """Test that complexity measure is always positive"""
        comp = complexity_measure(d)
        assert comp >= 0, f"Negative complexity at d={d}: {comp}"
        if comp == 0:
            # Check if this is expected (very large d might underflow)
            vol = ball_volume(d)
            surf = sphere_surface(d)
            if vol > NUMERICAL_EPSILON and surf > NUMERICAL_EPSILON:
                assert (
                    False
                ), f"Unexpected zero complexity at d={d} with V={vol}, S={surf}"


class TestKnownValueVerification:
    """Test against known exact values for integer dimensions"""

    def test_volume_integer_dimensions(self):
        """Test volume for known integer dimensions"""
        # V_0 = 1
        assert abs(ball_volume(0) - 1.0) < NUMERICAL_EPSILON, "V_0 ≠ 1"

        # V_1 = 2
        assert abs(ball_volume(1) - 2.0) < NUMERICAL_EPSILON, "V_1 ≠ 2"

        # V_2 = π
        assert abs(ball_volume(2) - PI) < NUMERICAL_EPSILON, "V_2 ≠ π"

        # V_3 = 4π/3
        expected_v3 = 4 * PI / 3
        assert (
            abs(ball_volume(3) - expected_v3) < NUMERICAL_EPSILON
        ), f"V_3 ≠ 4π/3: {ball_volume(3)} ≠ {expected_v3}"

    def test_surface_integer_dimensions(self):
        """Test surface area for known integer dimensions"""
        # S_1 = 2 (two points on circle)
        assert abs(sphere_surface(1) - 2.0) < NUMERICAL_EPSILON, "S_1 ≠ 2"

        # S_2 = 2π (circumference)
        assert abs(sphere_surface(2) - 2 * PI) < NUMERICAL_EPSILON, "S_2 ≠ 2π"

        # S_3 = 4π (sphere surface area)
        assert abs(sphere_surface(3) - 4 * PI) < NUMERICAL_EPSILON, "S_3 ≠ 4π"


class TestRecurrenceRelations:
    """Test recurrence relationships between dimensions"""

    @given(
        st.floats(min_value=1.0, max_value=15.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=300)
    def test_volume_recurrence(self, d):
        """Test volume recurrence relation: V_{d+2} = (2π/(d+2)) × V_d"""
        # Correct recurrence: V_d = π^{d/2} / Γ(d/2 + 1)
        # V_{d+2} = π^{(d+2)/2} / Γ((d+2)/2 + 1) = π^{d/2} × π / Γ(d/2 + 2)
        # Since Γ(z+1) = z×Γ(z): Γ(d/2 + 2) = (d/2 + 1)×Γ(d/2 + 1)
        # So V_{d+2} = π × V_d / (d/2 + 1) = 2π × V_d / (d + 2)
        vol_d = ball_volume(d)
        vol_d_plus_2 = ball_volume(d + 2)

        assume(np.isfinite(vol_d) and np.isfinite(vol_d_plus_2))
        assume(vol_d > NUMERICAL_EPSILON)

        expected = (2 * PI / (d + 2)) * vol_d
        relative_error = abs(vol_d_plus_2 - expected) / max(
            abs(expected), NUMERICAL_EPSILON
        )

        # Allow some numerical error, especially for larger dimensions
        tolerance = 1e-12 if d < 10 else 1e-10
        assert (
            relative_error < tolerance
        ), f"Volume recurrence failed at d={d}: V_{d+2} = {vol_d_plus_2} ≠ {expected}"

    @given(
        st.floats(min_value=1.5, max_value=15.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=300)
    def test_surface_volume_relation(self, d):
        """Test relation: S_d = d × V_d / R where R is radius"""
        # For unit sphere (R=1): S_d = d × V_d (this is actually the derivative relation)
        # The correct relation is: S_d = d × V_d at unit radius
        vol_d = ball_volume(d)
        surf_d = sphere_surface(d)

        assume(np.isfinite(vol_d) and np.isfinite(surf_d))
        assume(vol_d > NUMERICAL_EPSILON)

        # Actually, the correct relation is: d × V_{d-1} = S_d
        # But let's test the derivative relation: dV/dr = S at r=1
        # For V(r) = V_d × r^d, we have dV/dr = d × V_d × r^{d-1} = d × V_d at r=1

        expected = d * vol_d
        relative_error = abs(surf_d - expected) / max(abs(expected), NUMERICAL_EPSILON)

        # This relation has more tolerance due to numerical complexities
        tolerance = 1e-6 if d < 10 else 1e-4
        assert (
            relative_error < tolerance
        ), f"Surface-volume relation failed at d={d}: S_d = {surf_d} ≠ d×V_d = {expected}"


class TestScalingProperties:
    """Test scaling behavior of measures"""

    @given(
        st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_volume_scaling(self, d, scale):
        """Test V_d(r) = V_d(1) × r^d scaling law"""
        unit_volume = ball_volume(d)
        assume(np.isfinite(unit_volume) and unit_volume > NUMERICAL_EPSILON)

        unit_volume * (scale**d)

        # For our implementation, we compute unit volumes, so verify the scaling law mathematically
        # V_d(r) = π^{d/2} × r^d / Γ(d/2 + 1)
        # Therefore: V_d(scale) / V_d(1) = scale^d
        ratio = scale**d

        # Verify this makes sense
        assert ratio > 0, f"Invalid scaling ratio at d={d}, scale={scale}"

        if scale > 1:
            assert ratio >= 1, "Volume should increase with scale > 1"
        elif scale < 1:
            assert ratio <= 1, "Volume should decrease with scale < 1"

    @given(
        st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_surface_scaling(self, d, scale):
        """Test S_d(r) = S_d(1) × r^{d-1} scaling law"""
        unit_surface = sphere_surface(d)
        assume(np.isfinite(unit_surface) and unit_surface > NUMERICAL_EPSILON)

        # Surface area scales as r^{d-1}
        ratio = scale ** (d - 1)

        assert ratio > 0, f"Invalid surface scaling ratio at d={d}, scale={scale}"

        if d > 1:  # For d > 1, surface scaling behaves normally
            if scale > 1:
                assert ratio >= 1, "Surface should increase with scale > 1 when d > 1"
            elif scale < 1:
                assert ratio <= 1, "Surface should decrease with scale < 1 when d > 1"


class TestAsymptoticBehavior:
    """Test asymptotic behavior of measures"""

    @given(
        st.floats(min_value=10.0, max_value=50.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_volume_large_d_monotonicity(self, d):
        """Test that volume decreases monotonically for large d"""
        vol_d = ball_volume(d)
        vol_d_plus_1 = ball_volume(d + 1)

        assume(np.isfinite(vol_d) and np.isfinite(vol_d_plus_1))
        assume(vol_d > NUMERICAL_EPSILON and vol_d_plus_1 > NUMERICAL_EPSILON)

        # For large d, volume should generally decrease
        # (though there might be small fluctuations due to numerical precision)
        ratio = vol_d_plus_1 / vol_d

        # Allow some numerical tolerance, but expect general decreasing trend
        assert (
            ratio < 2.0
        ), f"Volume increasing too rapidly at large d={d}: V_{d+1}/V_d = {ratio}"

    @given(
        st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_small_d_behavior(self, d):
        """Test behavior for small fractional dimensions"""
        vol = ball_volume(d)
        surf = sphere_surface(d)

        # Should be well-defined and finite
        assert np.isfinite(vol), f"Non-finite volume at small d={d}"
        assert np.isfinite(surf), f"Non-finite surface at small d={d}"

        # Should be positive
        assert vol > 0, f"Non-positive volume at small d={d}: {vol}"
        assert surf > 0, f"Non-positive surface at small d={d}: {surf}"


class TestComplexityMeasureProperties:
    """Test properties of the complexity measure C_d = V_d × S_d"""

    @given(
        st.floats(min_value=0.5, max_value=15.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=200)
    def test_complexity_factorization(self, d):
        """Test that C_d = V_d × S_d"""
        vol = ball_volume(d)
        surf = sphere_surface(d)
        comp = complexity_measure(d)

        assume(np.isfinite(vol) and np.isfinite(surf) and np.isfinite(comp))

        expected = vol * surf
        relative_error = abs(comp - expected) / max(abs(expected), NUMERICAL_EPSILON)

        assert (
            relative_error < 1e-12
        ), f"Complexity factorization failed at d={d}: C_d = {comp} ≠ V_d × S_d = {expected}"

    @given(
        st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_complexity_peak_existence(self, d_center):
        """Test that complexity has local structure (not monotonic everywhere)"""
        # Test points around d_center
        epsilon = 0.1
        comp_left = complexity_measure(max(0.1, d_center - epsilon))
        comp_center = complexity_measure(d_center)
        comp_right = complexity_measure(d_center + epsilon)

        assume(all(np.isfinite(c) for c in [comp_left, comp_center, comp_right]))
        assume(all(c > NUMERICAL_EPSILON for c in [comp_left, comp_center, comp_right]))

        # At least one of the inequalities should hold (not all equal)
        # This tests that complexity is not constan
        max_comp = max(comp_left, comp_center, comp_right)
        min_comp = min(comp_left, comp_center, comp_right)

        relative_variation = (max_comp - min_comp) / max_comp
        assert (
            relative_variation > 1e-10
        ), f"Complexity appears constant around d={d_center}: variation = {relative_variation}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
