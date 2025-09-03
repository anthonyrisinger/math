#!/usr/bin/env python3
"""
Tests for validation module.
"""

from dimensional.core.validation import PropertyValidator


class TestValidation:
    """Test validation functions."""

    def test_property_validator_creation(self):
        """Test PropertyValidator can be instantiated."""
        validator = PropertyValidator()
        assert validator is not None
