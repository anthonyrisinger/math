#!/usr/bin/env python3
"""
Tests for mathematical functions module.
"""

import pytest

from dimensional.gamma import gamma, gammaln


class TestMathematicalFunctions:
    """Test mathematical functions."""

    def test_gamma_safe_basic(self):
        """Test basic gamma function behavior."""
        assert gamma(1.0) == pytest.approx(1.0)
        assert gamma(2.0) == pytest.approx(1.0)

    def test_gammaln_safe_basic(self):
        """Test basic log-gamma function behavior."""
        assert gammaln(1.0) == pytest.approx(0.0)
