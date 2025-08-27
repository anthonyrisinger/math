"""
Pytest Configuration for Mathematical Modeling Library
======================================================

Provides shared fixtures, configuration, and testing utilities for the entire test suite.
Ensures consistent testing environment and mathematical validation across all modules.
"""

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# Add the project root to Python path for testing
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure numpy for strict mathematical testing
np.seterr(all="raise", under="ignore")  # Convert warnings to errors, ignore underflow
warnings.filterwarnings(
    "error", category=RuntimeWarning
)  # Mathematical warnings as errors

# Mathematical Testing Constants
MATHEMATICAL_TOLERANCE = 1e-12  # Tolerance for exact mathematical relationships
NUMERICAL_TOLERANCE = 1e-10  # Tolerance for numerical computations
PROPERTY_TEST_TOLERANCE = 1e-8  # Tolerance for property-based tests

# ===== Core Fixtures =====


@pytest.fixture
def math_tolerance():
    """Standard tolerance for mathematical property tests."""
    return MATHEMATICAL_TOLERANCE


@pytest.fixture
def numerical_tolerance():
    """Standard tolerance for numerical computation tests."""
    return NUMERICAL_TOLERANCE


@pytest.fixture
def test_dimensions():
    """Standard set of test dimensions covering all mathematical regimes."""
    return [
        0,  # Point (special case)
        0.5,  # Half dimension (quantum mechanics)
        1,  # Line
        1.5,  # Fractional (between line and plane)
        2,  # Plane
        3,  # Our familiar 3D space
        4,  # Spacetime
        5,  # Near complexity peak
        6,  # Complexity peak region
        7,  # Past complexity peak
        8,  # High dimension
        10,  # Very high dimension
        15,  # Extreme dimension for stability testing
    ]


@pytest.fixture
def critical_dimensions():
    """Dictionary of mathematically significant dimensions."""
    return {
        "void": 0,
        "half": 0.5,
        "line": 1,
        "plane": 2,
        "space": 3,
        "spacetime": 4,
        "pi_boundary": math.pi,  # First stability boundary
        "complexity_peak": 6.0,  # Approximate complexity peak
        "two_pi_boundary": 2 * math.pi,  # Compression boundary
        "leech_limit": 24,  # Leech lattice dimension
    }


@pytest.fixture
def known_gamma_values():
    """Dictionary of known exact gamma function values for validation."""
    sqrt_pi = math.sqrt(math.pi)
    return {
        1: 1.0,  # Γ(1) = 0!
        2: 1.0,  # Γ(2) = 1!
        3: 2.0,  # Γ(3) = 2!
        4: 6.0,  # Γ(4) = 3!
        5: 24.0,  # Γ(5) = 4!
        0.5: sqrt_pi,  # Γ(1/2) = √π
        1.5: sqrt_pi / 2,  # Γ(3/2) = √π/2
        2.5: 3 * sqrt_pi / 4,  # Γ(5/2) = 3√π/4
        3.5: 15 * sqrt_pi / 8,  # Γ(7/2) = 15√π/8
        4.5: 105 * sqrt_pi / 16,  # Γ(9/2) = 105√π/16
    }


@pytest.fixture
def known_dimensional_measures():
    """Dictionary of known exact dimensional measure values."""
    pi = math.pi
    return {
        # Ball volumes V_d
        "ball_volume": {
            0: 1,  # Poin
            1: 2,  # Line segment [-1,1]
            2: pi,  # Disk
            3: 4 * pi / 3,  # Ball
            4: pi**2 / 2,  # 4D hypersphere
        },
        # Sphere surfaces S_d
        "sphere_surface": {
            1: 2,  # Two points
            2: 2 * pi,  # Circle
            3: 4 * pi,  # Sphere
            4: 2 * pi**2,  # 3-sphere in 4D
        },
    }


@pytest.fixture
def golden_phase_engine():
    """Phase dynamics engine with well-tested 'golden' parameters."""
    from dimensional.phase import PhaseDynamicsEngine
    return PhaseDynamicsEngine(max_dimensions=6, use_adaptive=True)


@pytest.fixture
def stress_test_engine():
    """Phase engine configured for stress testing extreme conditions."""
    from dimensional.phase import PhaseDynamicsEngine

    engine = PhaseDynamicsEngine(max_dimensions=12, use_adaptive=True)
    # Add some initial energy for interesting dynamics
    engine.inject_energy(dimension=1, amount=0.5)
    engine.inject_energy(dimension=2, amount=0.2)
    return engine


# ===== Helper Functions =====


def assert_mathematical_equality(
    actual, expected, tolerance=MATHEMATICAL_TOLERANCE, context=""
):
    """
    Assert mathematical equality with proper tolerance and informative error messages.

    Parameters
    ----------
    actual : floa
        Computed value
    expected : floa
        Expected exact value
    tolerance : floa
        Acceptable numerical error
    context : str
        Description for error message
    """
    if not np.isfinite(actual):
        pytest.fail(f"Non-finite result: {actual} {context}")

    if not np.isfinite(expected):
        pytest.fail(f"Non-finite expected value: {expected} {context}")

    error = abs(actual - expected)
    relative_error = error / abs(expected) if expected != 0 else error

    if error > tolerance:
        pytest.fail(
            f"Mathematical equality failed {context}:\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            f"  Error:    {error} (tolerance: {tolerance})\n"
            f"  Relative: {relative_error:.2e}"
        )


def assert_conserved_quantity(
    initial_value, final_value, tolerance=1e-12, quantity_name="quantity"
):
    """Assert that a physical quantity is conserved (energy, probability, etc.)."""
    drift = abs(final_value - initial_value)
    relative_drift = drift / abs(initial_value) if initial_value != 0 else drift

    if drift > tolerance:
        pytest.fail(
            f"{quantity_name} not conserved:\n"
            f"  Initial:  {initial_value}\n"
            f"  Final:    {final_value}\n"
            f"  Drift:    {drift} (tolerance: {tolerance})\n"
            f"  Relative: {relative_drift:.2e}"
        )


def assert_integer_invariant(value, tolerance=1e-10, context=""):
    """Assert that a topological invariant remains integer-valued."""
    nearest_int = round(value)
    deviation = abs(value - nearest_int)

    if deviation > tolerance:
        pytest.fail(
            f"Non-integer topological invariant {context}:\n"
            f"  Value:    {value}\n"
            f"  Nearest:  {nearest_int}\n"
            f"  Deviation: {deviation} (tolerance: {tolerance})"
        )


# ===== Test Markers and Collection =====


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (> 5 seconds)")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line(
        "markers", "mathematical: marks core mathematical property tests"
    )
    config.addinivalue_line("markers", "numerical: marks numerical stability tests")
    config.addinivalue_line("markers", "property: marks property-based tests")
    config.addinivalue_line("markers", "benchmark: marks performance benchmark tests")
    config.addinivalue_line(
        "markers", "visualization: marks visualization tests (may require display)"
    )


def pytest_collection_modifyitems(config, items):
    """Customize test collection and execution order."""
    # Run mathematical property tests first (they're fastest and most important)
    mathematical_tests = []
    numerical_tests = []
    integration_tests = []
    other_tests = []

    for item in items:
        if "mathematical" in item.keywords:
            mathematical_tests.append(item)
        elif "numerical" in item.keywords:
            numerical_tests.append(item)
        elif "integration" in item.keywords:
            integration_tests.append(item)
        else:
            other_tests.append(item)

    # Reorder: mathematical → numerical → other → integration (slowest last)
    items[:] = mathematical_tests + numerical_tests + other_tests + integration_tests

    # Auto-mark slow tests
    slow_keywords = ["integration", "benchmark", "visualization"]
    for item in items:
        if any(keyword in item.keywords for keyword in slow_keywords):
            item.add_marker(pytest.mark.slow)


# ===== Fixtures for Specific Test Categories =====


@pytest.fixture
def regression_data_path():
    """Path to golden reference data for regression tests."""
    return Path(__file__).parentt / "tests" / "regression" / "golden_reference"


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs (plots, data files, etc.)."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(scope="session")
def benchmark_results():
    """Session-scoped fixture to collect benchmark results across tests."""
    return {}


# ===== Exception Handling for Tests =====


@pytest.fixture(autouse=True)
def handle_test_warnings():
    """Automatically handle warnings during tests."""
    with warnings.catch_warnings():
        # Convert mathematical warnings to errors
        warnings.filterwarnings("error", category=RuntimeWarning)
        # Direct numpy warning handling - eliminate defensive try/excep
        if hasattr(np, 'ComplexWarning'):
            warnings.filterwarnings("error", category=np.ComplexWarning)

        # Allow some specific warnings that are expected
        warnings.filterwarnings(
            "ignore", message=".*overflow.*", category=RuntimeWarning
        )
        warnings.filterwarnings(
            "ignore", message=".*divide by zero.*", category=RuntimeWarning
        )
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module="matplotlib.*"
        )

        yield


# ===== Performance Testing Utilities =====


@pytest.fixture
def performance_threshold():
    """Performance thresholds for benchmark tests."""
    return {
        "gamma_function_1k_evals": 0.1,  # seconds
        "phase_evolution_100_steps": 1.0,  # seconds
        "visualization_render": 2.0,  # seconds
        "convergence_analysis": 5.0,  # seconds
    }


# ===== Make fixtures available globally =====
pytest_plugins = []
