#!/usr/bin/env python3
"""
Tests for mathematical constants module.
"""

from dimensional.mathematics.constants import NUMERICAL_EPSILON


class TestConstants:
    """Test mathematical constants."""

    def test_numerical_epsilon_defined(self):
        """Test that numerical epsilon is properly defined."""
        assert NUMERICAL_EPSILON > 0
        assert isinstance(NUMERICAL_EPSILON, float)
