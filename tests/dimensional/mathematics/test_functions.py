#!/usr/bin/env python3
"""
Tests for mathematical functions module.
"""

import pytest

from dimensional.mathematics.functions import gamma_safe, gammaln_safe


class TestMathematicalFunctions:
    """Test mathematical functions."""

    def test_gamma_safe_basic(self):
        """Test basic gamma function behavior."""
        assert gamma_safe(1.0) == pytest.approx(1.0)
        assert gamma_safe(2.0) == pytest.approx(1.0)

    def test_gammaln_safe_basic(self):
        """Test basic log-gamma function behavior."""
        assert gammaln_safe(1.0) == pytest.approx(0.0)
