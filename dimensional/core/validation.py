"""Validation module for dimensional mathematics."""

import numpy as np


class PropertyValidator:
    """Basic property validator for mathematical functions."""

    def __init__(self):
        """Initialize the validator."""
        pass


class ConvergenceDiagnostics:
    """Diagnostics for convergence analysis."""

    def __init__(self, tolerance=1e-10, threshold=None):
        """Initialize convergence diagnostics."""
        # Support both parameter names for compatibility
        self.threshold = threshold if threshold is not None else tolerance
        self.tolerance = self.threshold
        self.results = {}

    def analyze(self, values, dimensions=None):
        """Analyze convergence of values."""
        values = np.asarray(values)

        if dimensions is None:
            dimensions = np.arange(len(values))

        # Find where values drop below threshold
        below_threshold = values < self.threshold
        if np.any(below_threshold):
            converge_idx = np.argmax(below_threshold)
            converge_dim = float(dimensions[converge_idx])
        else:
            converge_dim = None

        self.results = {
            'converged': converge_dim is not None,
            'convergence_dimension': converge_dim,
            'final_value': float(values[-1]),
            'min_value': float(np.min(values)),
            'max_value': float(np.max(values)),
            'threshold': self.threshold,
        }

        return self.results

    def get_report(self):
        """Get convergence report."""
        return self.results.copy()

    def richardson_extrapolation(self, func, x, h=1e-4, order=2):
        """Apply Richardson extrapolation for improved accuracy."""
        # Richardson extrapolation formula
        f1 = func(x, h)
        f2 = func(x, h/2)

        if order == 2:
            # Second-order extrapolation
            result = (4*f2 - f1) / 3
        else:
            # General order
            result = (2**order * f2 - f1) / (2**order - 1)

        return result

    def aitken_acceleration(self, sequence):
        """Apply Aitken's delta-squared acceleration."""
        sequence = np.asarray(sequence)

        if len(sequence) < 3:
            return sequence

        # Aitken's formula
        accelerated = []
        for i in range(len(sequence) - 2):
            s_n = sequence[i]
            s_n1 = sequence[i + 1]
            s_n2 = sequence[i + 2]

            denominator = s_n2 - 2*s_n1 + s_n
            if abs(denominator) > 1e-15:
                a_n = s_n - (s_n1 - s_n)**2 / denominator
            else:
                a_n = s_n2

            accelerated.append(a_n)

        return np.array(accelerated)

    def test_fractional_convergence(self, func, dimensions):
        """Test convergence over fractional dimensions."""
        dimensions = np.asarray(dimensions)
        values = np.array([func(d) for d in dimensions])

        # Analyze convergence
        convergence_info = self.analyze(values, dimensions)

        # Check if converging
        if len(values) > 1:
            differences = np.abs(np.diff(values))
            converging = np.all(differences[1:] <= differences[:-1] * 1.1)  # Allow small increase
        else:
            converging = False

        convergence_info['converging'] = converging

        return convergence_info


class DomainValidator:
    """Validate mathematical domains."""

    def __init__(self):
        """Initialize domain validator."""
        self.errors = []

    def validate_positive(self, x, name="value"):
        """Validate that values are positive."""
        x = np.asarray(x)
        if np.any(x <= 0):
            self.errors.append(f"{name} must be positive, got {x[x <= 0]}")
            return False
        return True

    def validate_nonnegative(self, x, name="value"):
        """Validate that values are non-negative."""
        x = np.asarray(x)
        if np.any(x < 0):
            self.errors.append(f"{name} must be non-negative, got {x[x < 0]}")
            return False
        return True

    def validate_finite(self, x, name="value"):
        """Validate that values are finite."""
        x = np.asarray(x)
        if not np.all(np.isfinite(x)):
            self.errors.append(f"{name} must be finite")
            return False
        return True

    def get_errors(self):
        """Get validation errors."""
        return self.errors.copy()

    def clear_errors(self):
        """Clear errors."""
        self.errors = []


def validate_mathematical_properties():
    """Validate basic mathematical properties of the library."""
    import numpy as np

    from ..core import c, gamma, r, s, v

    # Test basic properties
    tests_passed = []

    # Test 1: Gamma(1) = 1
    tests_passed.append(abs(gamma(1.0) - 1.0) < 1e-10)

    # Test 2: V(2) = π
    tests_passed.append(abs(v(2.0) - np.pi) < 1e-10)

    # Test 3: S(2) = 2π
    tests_passed.append(abs(s(2.0) - 2*np.pi) < 1e-10)

    # Test 4: C = V * S
    d = 3.0
    tests_passed.append(abs(c(d) - v(d) * s(d)) < 1e-10)

    # Test 5: R = S/V
    tests_passed.append(abs(r(d) - s(d)/v(d)) < 1e-10)

    return all(tests_passed)


class NumericalStabilityTester:
    """Test numerical stability of mathematical functions."""

    def __init__(self, epsilon=1e-10):
        """Initialize stability tester."""
        self.epsilon = epsilon
        self.results = {}

    def test_continuity(self, func, x, h=1e-8):
        """Test continuity at point x."""
        y = func(x)
        y_plus = func(x + h)
        y_minus = func(x - h)

        # Check if function values are close
        continuous = (
            np.abs(y_plus - y) < self.epsilon * max(1, abs(y)) and
            np.abs(y_minus - y) < self.epsilon * max(1, abs(y))
        )

        self.results['continuity'] = {
            'continuous': continuous,
            'point': x,
            'value': y,
            'value_plus': y_plus,
            'value_minus': y_minus,
        }

        return continuous

    def test_monotonicity(self, func, x_range):
        """Test monotonicity over a range."""
        x_range = np.asarray(x_range)
        values = np.array([func(x) for x in x_range])

        # Check if strictly increasing
        increasing = np.all(np.diff(values) > 0)
        # Check if strictly decreasing
        decreasing = np.all(np.diff(values) < 0)
        # Check if non-decreasing
        non_decreasing = np.all(np.diff(values) >= -self.epsilon)
        # Check if non-increasing
        non_increasing = np.all(np.diff(values) <= self.epsilon)

        self.results['monotonicity'] = {
            'increasing': increasing,
            'decreasing': decreasing,
            'non_decreasing': non_decreasing,
            'non_increasing': non_increasing,
            'values': values.tolist(),
        }

        return increasing or decreasing

    def test_boundedness(self, func, x_range):
        """Test if function is bounded over a range."""
        x_range = np.asarray(x_range)
        values = np.array([func(x) for x in x_range])

        bounded = np.all(np.isfinite(values))

        self.results['boundedness'] = {
            'bounded': bounded,
            'min': float(np.min(values)) if bounded else None,
            'max': float(np.max(values)) if bounded else None,
            'range': [float(x_range[0]), float(x_range[-1])],
        }

        return bounded

    def test_stability(self, func, x, perturbation=1e-10):
        """Test numerical stability with small perturbations."""
        y = func(x)
        y_perturbed = func(x + perturbation)

        # Relative change in output
        if abs(y) > self.epsilon:
            relative_change = abs(y_perturbed - y) / abs(y)
        else:
            relative_change = abs(y_perturbed - y)

        # Condition number approximation
        condition = relative_change / (perturbation / max(1, abs(x)))

        stable = condition < 1000  # Arbitrary threshold for stability

        self.results['stability'] = {
            'stable': stable,
            'condition_number': condition,
            'relative_change': relative_change,
            'point': x,
        }

        return stable

    def get_report(self):
        """Get stability test report."""
        return self.results.copy()
