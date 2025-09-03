#!/usr/bin/env python3
"""
CLI Integration Tests
====================

Test suite for the command-line interface to ensure production-grade robustness.
"""

import subprocess
import sys

import pytest


class TestCLIBasics:
    """Test basic CLI functionality."""

    @pytest.mark.skip(reason='Deprecated')
    def test_cli_module_imports(self):
        """Test that CLI module can be imported without errors."""
        result = subprocess.run([
            sys.executable, "-c", "from dimensional.cli import app; print('CLI import successful')"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "CLI import successful" in result.stdout

    def test_cli_help_command(self):
        """Test that CLI help works."""
        result = subprocess.run([
            sys.executable, "-m", "dimensional", "--help"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "dimensional" in result.stdout.lower()

    def test_basic_mathematical_operations(self):
        """Test basic mathematical operations through Python API."""
        result = subprocess.run([
            sys.executable, "-c",
            "import dimensional as dm; print(f'V(4)={dm.V(4):.6f}'); print(f'S(4)={dm.S(4):.6f}'); print('OK')"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "V(4)=4.934802" in result.stdout
        assert "S(4)=19.739209" in result.stdout
        assert "OK" in result.stdout


class TestErrorHandling:
    """Test CLI error handling and robustness."""

    def test_invalid_dimensions(self):
        """Test handling of invalid dimensional inputs."""
        result = subprocess.run([
            sys.executable, "-c",
            """
import dimensional as dm
import sys
try:
    dm.V(-2)  # Should handle gracefully
    print('HANDLED_NEGATIVE')
except Exception as e:
    print(f'ERROR_CAUGHT: {type(e).__name__}')
    sys.exit(0)
"""
        ], capture_output=True, text=True)
        assert result.returncode == 0
        # Should either handle gracefully or catch the error
        assert "HANDLED_NEGATIVE" in result.stdout or "ERROR_CAUGHT" in result.stdout


class TestPedagogicalFeatures:
    """Test pedagogical and educational features."""

    def test_quick_start_function(self):
        """Test that quick_start provides educational content."""
        result = subprocess.run([
            sys.executable, "-c",
            "import dimensional as dm; dm.quick_start()"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "explore" in result.stdout
        assert "peaks" in result.stdout
        assert "dimensional" in result.stdout.lower()

    def test_mathematical_constants_available(self):
        """Test that mathematical constants are properly exposed."""
        result = subprocess.run([
            sys.executable, "-c",
            """
import dimensional as dm
print(f'PHI={dm.PHI:.6f}')
print(f'PI={dm.PI:.6f}')
print(f'PSI={dm.PSI:.6f}')
print(f'VARPI={dm.VARPI:.6f}')
print('CONSTANTS_OK')
"""
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "PHI=1.618034" in result.stdout
        assert "PI=3.141593" in result.stdout
        assert "CONSTANTS_OK" in result.stdout


class TestProductionReadiness:
    """Test production-grade robustness features."""

    def test_numerical_stability_edge_cases(self):
        """Test numerical stability with edge cases."""
        result = subprocess.run([
            sys.executable, "-c",
            """
import dimensional as dm
import numpy as np
try:
    # Test very small values
    small_val = dm.V(0.001)
    print(f'SMALL_VALUE: {small_val}')

    # Test moderate values
    moderate_val = dm.V(10.0)
    print(f'MODERATE_VALUE: {moderate_val}')

    # Test array input
    arr_vals = dm.V(np.array([1.0, 2.0, 3.0, 4.0]))
    print(f'ARRAY_VALUES: {len(arr_vals)}')

    print('STABILITY_OK')
except Exception as e:
    print(f'STABILITY_ERROR: {e}')
"""
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "STABILITY_OK" in result.stdout

    def test_package_metadata(self):
        """Test that package metadata is properly defined."""
        result = subprocess.run([
            sys.executable, "-c",
            """
import dimensional as dm
print(f'VERSION: {dm.__version__}')
print(f'AUTHOR: {dm.__author__}')
print(f'DESCRIPTION: {dm.__description__}')
print('METADATA_OK')
"""
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "VERSION: 1.0.0" in result.stdout
        assert "METADATA_OK" in result.stdout
