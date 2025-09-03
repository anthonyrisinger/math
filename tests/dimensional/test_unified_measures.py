#!/usr/bin/env python3
"""Comprehensive tests for unified measures module."""

import warnings

import numpy as np
import pytest

from dimensional.measures import (
    C,
    R,
    S,
    V,
    ball_volume,
    ball_volume_fast,  # Compat
    batch_measures,
    c,
    complexity_measure,
    complexity_measure_fast,
    convergence_analysis,
    find_all_peaks,
    find_peak,
    r,
    ratio_measure,
    s,
    sphere_surface,
    sphere_surface_fast,
    v,  # All aliases
)


class TestUnifiedMeasuresCore:
    """Test core measure functions."""

    def test_ball_volume_scalar(self):
        """Test ball volume with scalar input."""
        # Known values
        assert abs(ball_volume(0) - 1.0) < 1e-10
        assert abs(ball_volume(1) - 2.0) < 1e-10
        assert abs(ball_volume(2) - np.pi) < 1e-10
        assert abs(ball_volume(3) - 4*np.pi/3) < 1e-10

    def test_ball_volume_vector(self):
        """Test ball volume with vector input."""
        d = np.array([0, 1, 2, 3])
        result = ball_volume(d)
        assert result.shape == d.shape
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1] - 2.0) < 1e-10
        assert abs(result[2] - np.pi) < 1e-10

    def test_sphere_surface_scalar(self):
        """Test sphere surface with scalar input."""
        # Known values
        assert abs(sphere_surface(0) - 2.0) < 1e-10
        assert abs(sphere_surface(1) - 2.0) < 1e-10
        assert abs(sphere_surface(2) - 2*np.pi) < 1e-10
        assert abs(sphere_surface(3) - 4*np.pi) < 1e-10

    def test_complexity_measure(self):
        """Test complexity measure C = V * S."""
        d = 4.0
        v_val = ball_volume(d)
        s_val = sphere_surface(d)
        c_val = complexity_measure(d)
        assert abs(c_val - v_val * s_val) < 1e-10

    def test_ratio_measure(self):
        """Test ratio measure R = S / V."""
        d = 4.0
        v_val = ball_volume(d)
        s_val = sphere_surface(d)
        r_val = ratio_measure(d)
        assert abs(r_val - s_val / v_val) < 1e-10


class TestValidationParameter:
    """Test the validate parameter functionality."""

    def test_validation_on_warnings(self):
        """Test that validation produces warnings for edge cases."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Large dimension should warn
            ball_volume(150, validate=True)
            assert len(w) > 0
            assert "may underflow" in str(w[0].message)

    def test_validation_off_no_warnings(self):
        """Test that skipping validation avoids warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # No warnings when validation is off
            ball_volume(150, validate=False)
            assert len(w) == 0

    def test_negative_dimension_warning(self):
        """Test warning for negative dimensions."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ball_volume(-2.5, validate=True)
            assert len(w) > 0
            assert "Negative dimension" in str(w[0].message)

    def test_performance_without_validation(self):
        """Test that skipping validation is faster."""
        import time

        d = np.random.uniform(1, 10, 1000)

        # With validation
        start = time.perf_counter()
        for _ in range(100):
            ball_volume(d, validate=True)
        time_with = time.perf_counter() - start

        # Without validation
        start = time.perf_counter()
        for _ in range(100):
            ball_volume(d, validate=False)
        time_without = time.perf_counter() - start

        # Without validation should be at least slightly faster
        assert time_without <= time_with * 1.1  # Allow 10% margin


class TestBatchMeasures:
    """Test batch computation functionality."""

    def test_batch_measures_output(self):
        """Test batch_measures returns all measures."""
        d = np.array([2, 3, 4, 5])
        result = batch_measures(d)

        assert 'volume' in result
        assert 'surface' in result
        assert 'complexity' in result
        assert 'ratio' in result

        # Check shapes
        for key, values in result.items():
            assert values.shape == d.shape

    def test_batch_measures_consistency(self):
        """Test batch_measures matches individual functions."""
        d = np.array([2, 3, 4, 5])
        batch = batch_measures(d, validate=False)

        np.testing.assert_allclose(batch['volume'], ball_volume(d, validate=False))
        np.testing.assert_allclose(batch['surface'], sphere_surface(d, validate=False))
        np.testing.assert_allclose(batch['complexity'], complexity_measure(d, validate=False))
        np.testing.assert_allclose(batch['ratio'], ratio_measure(d, validate=False))

    def test_batch_efficiency(self):
        """Test that batch computation is more efficient."""
        import time

        d = np.linspace(1, 10, 100)

        # Individual calls
        start = time.perf_counter()
        ball_volume(d, validate=False)
        sphere_surface(d, validate=False)
        complexity_measure(d, validate=False)
        ratio_measure(d, validate=False)
        time_individual = time.perf_counter() - start

        # Batch call
        start = time.perf_counter()
        batch_measures(d, validate=False)
        time_batch = time.perf_counter() - start

        # Batch should be faster (reuses gamma calculations)
        assert time_batch <= time_individual * 1.2  # Allow some margin


class TestAnalysisFunctions:
    """Test analysis and peak finding functions."""

    def test_find_peak(self):
        """Test find_peak function."""
        peak_dim, peak_val = find_peak(ball_volume)

        assert peak_dim > 0
        assert peak_dim < 20  # Should be in reasonable range
        assert peak_val > 0

        # Peak should be a local maximum
        eps = 0.1
        assert ball_volume(peak_dim, validate=False) >= ball_volume(peak_dim - eps, validate=False)
        assert ball_volume(peak_dim, validate=False) >= ball_volume(peak_dim + eps, validate=False)

    def test_find_all_peaks(self):
        """Test find_all_peaks function."""
        peaks = find_all_peaks()

        assert 'volume' in peaks
        assert 'surface' in peaks
        assert 'complexity' in peaks

        for name, peak in peaks.items():
            assert len(peak) == 2
            assert peak[0] > 0  # Dimension
            assert peak[1] > 0  # Value

    @pytest.mark.skip(reason='Deprecated')
    def test_convergence_analysis(self):
        """Test convergence analysis."""
        result = convergence_analysis(d_start=1.0, d_end=100.0)

        assert 'volume' in result
        assert 'surface' in result
        assert 'complexity' in result
        assert 'ratio' in result

        for name, analysis in result.items():
            assert 'converge_dimension' in analysis
            assert 'final_value' in analysis
            assert 'max_value' in analysis
            assert 'max_dimension' in analysis

            # Final values should be very small
            assert analysis['final_value'] < 1e-10 or analysis['final_value'] > 1e10


class TestAliases:
    """Test all aliases work correctly."""

    def test_lowercase_aliases(self):
        """Test lowercase aliases v, s, c, r."""
        assert v(4.0) == ball_volume(4.0)
        assert s(4.0) == sphere_surface(4.0)
        assert c(4.0) == complexity_measure(4.0)
        assert r(4.0) == ratio_measure(4.0)

    def test_uppercase_aliases(self):
        """Test uppercase aliases V, S, C, R."""
        assert V(4.0) == ball_volume(4.0)
        assert S(4.0) == sphere_surface(4.0)
        assert C(4.0) == complexity_measure(4.0)
        assert R(4.0) == ratio_measure(4.0)

    def test_fast_aliases(self):
        """Test fast aliases for backward compatibility."""
        d = np.array([2, 3, 4, 5])

        # Fast versions should skip validation
        result = ball_volume_fast(d)
        expected = ball_volume(d, validate=False)
        np.testing.assert_allclose(result, expected)

        result = sphere_surface_fast(d)
        expected = sphere_surface(d, validate=False)
        np.testing.assert_allclose(result, expected)

        result = complexity_measure_fast(d)
        expected = complexity_measure(d, validate=False)
        np.testing.assert_allclose(result, expected)


class TestEdgeCases:
    """Test edge cases and special values."""

    def test_zero_dimension(self):
        """Test measures at d=0."""
        assert ball_volume(0) == 1.0
        assert sphere_surface(0) == 2.0
        assert complexity_measure(0) == 2.0
        assert ratio_measure(0) == 2.0

    def test_one_dimension(self):
        """Test measures at d=1."""
        assert ball_volume(1) == 2.0
        assert sphere_surface(1) == 2.0
        assert complexity_measure(1) == 4.0
        assert ratio_measure(1) == 1.0

    def test_large_dimensions(self):
        """Test behavior at large dimensions."""
        d = 500.0
        # Should not crash, but will underflow
        v_val = ball_volume(d, validate=False)
        s_val = sphere_surface(d, validate=False)

        assert v_val >= 0  # Non-negative
        assert s_val >= 0  # Non-negative

        # Should be very close to zero
        assert v_val < 1e-100
        assert s_val < 1e-100

    def test_negative_dimensions(self):
        """Test measures at negative dimensions."""
        # Should compute mathematical extension
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v_val = ball_volume(-2.5)
            s_val = sphere_surface(-2.5)

            # Should return values (mathematical extension)
            assert np.isfinite(v_val) or np.isinf(v_val)
            assert np.isfinite(s_val) or np.isinf(s_val)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
