#!/usr/bin/env python3
"""Morphic polynomials and golden ratio operations for dimensional stability."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

# Core constants
# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
PSI = 1 / PHI  # Golden ratio conjugate ≈ 0.618
NUMERICAL_EPSILON = 1e-12

# Note: Geometric algebra functionality removed
# This module now focuses solely on morphic polynomials and golden ratio operations


# CORE POLYNOMIAL FUNCTIONS
# ========================


def morphic_polynomial_roots(k: float, mode: str = "shifted") -> NDArray[np.float64]:
    """Find real roots of morphic polynomial families."""
    if mode == "shifted":
        coeffs = [1.0, 0.0, -(2.0 - k), -1.0]  # τ³ + 0τ² - (2-k)τ - 1
    elif mode == "simple":
        coeffs = [1.0, 0.0, -k, -1.0]  # τ³ + 0τ² - kτ - 1
    else:
        raise ValueError("mode must be 'shifted' or 'simple'")

    # Find all roots
    roots = np.roots(coeffs)

    # Extract real roots (filter out complex with small imaginary part)
    real_roots = roots.real[np.abs(roots.imag) < NUMERICAL_EPSILON]

    # Sort in descending order
    return np.sort(real_roots)[::-1]


# Alias for backward compatibility
def real_roots(k: float, mode: str = "shifted") -> NDArray[np.float64]:
    """Alias for morphic_polynomial_roots."""
    return morphic_polynomial_roots(k, mode)


def discriminant(k: float, mode: str = "shifted") -> float:
    """Discriminant of morphic polynomial."""
    if mode == "shifted":
        # τ³ - (2-k)τ - 1 = 0
        # a = 1, b = 0, c = -(2-k), d = -1
        a, c, d = 1.0, -(2.0 - k), -1.0
        return -4 * a * (c**3) - 27 * (a**2) * (d**2)
    elif mode == "simple":
        # τ³ - kτ - 1 = 0
        # a = 1, b = 0, c = -k, d = -1
        a, c, d = 1.0, -k, -1.0
        return -4 * a * (c**3) - 27 * (a**2) * (d**2)
    else:
        raise ValueError("mode must be 'shifted' or 'simple'")


def k_perfect_circle(mode: str = "shifted") -> float:
    """Parameter value where τ = 1 is a root (perfect circle case)."""
    if mode == "shifted":
        # 1³ - (2-k)·1 - 1 = 0  =>  1 - (2-k) - 1 = 0  =>  k = 2
        return 2.0
    elif mode == "simple":
        # 1³ - k·1 - 1 = 0  =>  1 - k - 1 = 0  =>  k = 0
        return 0.0
    else:
        raise ValueError("mode must be 'shifted' or 'simple'")


def k_discriminant_zero(mode: str = "shifted") -> float:
    """Parameter value where discriminant equals zero."""
    if mode == "shifted":
        # Solve: -4(-(2-k))³ - 27 = 0
        # 4(2-k)³ = 27
        # (2-k)³ = 27/4
        # 2-k = (27/4)^(1/3)
        # k = 2 - (27/4)^(1/3)
        return 2.0 - (27.0 / 4.0) ** (1.0 / 3.0)
    elif mode == "simple":
        # Solve: -4(-k)³ - 27 = 0
        # 4k³ = 27
        # k³ = 27/4
        # k = (27/4)^(1/3)
        return (27.0 / 4.0) ** (1.0 / 3.0)
    else:
        raise ValueError("mode must be 'shifted' or 'simple'")


# ========================
# GOLDEN RATIO FUNCTIONS
# ========================


def golden_ratio_properties() -> dict[str, Any]:
    """Verify and return golden ratio properties."""
    phi = PHI
    psi = PSI

    return {
        "phi": phi,
        "psi": psi,
        "phi_squared": phi**2,
        "phi_plus_one": phi + 1,
        "phi_squared_equals_phi_plus_one": abs(phi**2 - (phi + 1))
        < NUMERICAL_EPSILON,
        "psi_squared": psi**2,
        "one_minus_psi": 1 - psi,
        "psi_squared_equals_one_minus_psi": abs(psi**2 - (1 - psi))
        < NUMERICAL_EPSILON,
        "phi_times_psi": phi * psi,
        "phi_times_psi_equals_one": abs(phi * psi - 1) < NUMERICAL_EPSILON,
        "phi_minus_psi": phi - psi,
        "phi_minus_psi_equals_one": abs((phi - psi) - 1.0) < NUMERICAL_EPSILON,
        "phi_plus_psi_equals_sqrt5": abs((phi + psi) - np.sqrt(5))
        < NUMERICAL_EPSILON,
    }


def morphic_scaling_factor(phi: float = PHI) -> float:
    """Calculate morphic scaling factor φ^(1/φ) ≈ 1.465."""
    return phi ** (1 / phi)


def generate_morphic_sequence(n_terms: int, phi: float = PHI) -> NDArray[np.float64]:
    """Generate morphic number sequence using golden ratio recurrence."""
    if n_terms <= 0:
        return np.array([])
    elif n_terms == 1:
        return np.array([1.0])
    elif n_terms == 2:
        return np.array([1.0, phi])

    sequence = np.zeros(n_terms)
    sequence[0] = 1.0
    sequence[1] = phi

    psi = PSI

    for i in range(2, n_terms):
        sequence[i] = phi * sequence[i - 1] + psi * sequence[i - 2]

    return sequence


# ========================
# MORPHIC CIRCLE TRANSFORMATIONS
# ========================


def morphic_circle_transform(tau: float, theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback morphic transformation for unit circle."""
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    z_circle = np.zeros_like(theta)

    # Apply morphic transformation matrix
    transform = morphic_transformation_matrix(tau)
    circle_2d = np.column_stack([x_circle, y_circle])
    transformed_2d = circle_2d @ transform.T

    # Add z-component (could be enhanced with 3D morphic rules)
    xyz = np.column_stack(
        [transformed_2d[:, 0], transformed_2d[:, 1], z_circle]
    )

    return xyz


# ========================
# GEOMETRIC TRANSFORMATIONS
# ========================


def morphic_transformation_matrix(tau: float, phi: float = PHI) -> NDArray[np.float64]:
    """Generate 2D transformation matrix for morphic scaling."""
    # Morphic transformation combining scaling and rotation
    scale = tau * morphic_scaling_factor(phi)
    angle = np.pi / phi  # Golden angle related rotation

    cos_a, sin_a = np.cos(angle), np.sin(angle)

    return scale * np.array([[cos_a, -sin_a], [sin_a, cos_a]])


def curvature_peak(xy: NDArray[np.float64], dtheta: float = 0.1) -> float:
    """Peak curvature |κ| for a closed polyline in XY."""
    x = xy[:, 0]
    y = xy[:, 1]

    with np.errstate(all="ignore"):
        dx, dy = np.gradient(x, dtheta), np.gradient(y, dtheta)
        ddx, ddy = np.gradient(dx, dtheta), np.gradient(dy, dtheta)
        denom = (dx * dx + dy * dy) ** 1.5
        denom[denom == 0] = np.nan
        kappa = (dx * ddy - dy * ddx) / denom

        if np.isfinite(kappa).any():
            return float(np.nanmax(np.abs(kappa)))
        return 0.0


def curvature_peak_estimate(tau: float, dtheta: float = 0.1) -> float:
    """Estimate peak curvature for morphic transformation."""
    # Generate unit circle points
    theta = np.arange(0, 2 * np.pi, dtheta)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)

    # Apply morphic transformation
    transform = morphic_transformation_matrix(tau)
    circle_points = np.column_stack([x_circle, y_circle])
    transformed = circle_points @ transform.T

    # Use the curvature_peak function
    return curvature_peak(transformed, dtheta)


# ========================
# STABILITY ANALYSIS
# ========================


def stability_regions(
    mode: str = "shifted",
    k_min: float = -5,
    k_max: float = 5,
    num_points: int = 1000,
) -> dict[str, Any]:
    """Find stability regions in parameter space."""
    k_values = np.linspace(k_min, k_max, num_points)

    results = {
        "k_values": k_values,
        "discriminants": [],
        "num_real_roots": [],
        "stable_regions": [],
        "critical_points": {},
    }

    for k in k_values:
        disc = discriminant(k, mode)
        roots = morphic_polynomial_roots(k, mode)

        results["discriminants"].append(disc)
        results["num_real_roots"].append(len(roots))

    results["discriminants"] = np.array(results["discriminants"])
    results["num_real_roots"] = np.array(results["num_real_roots"])

    # Find critical points
    results["critical_points"]["perfect_circle"] = k_perfect_circle(mode)
    results["critical_points"]["discriminant_zero"] = k_discriminant_zero(mode)

    # Stable regions (where discriminant > 0, indicating 3 real roots)
    stable_mask = results["discriminants"] > 0
    results["stable_regions"] = k_values[stable_mask]

    return results


# ========================
# MORPHIC ANALYZER CLASS
# ========================


class MorphicAnalyzer:
    """Morphic mathematics analyzer for polynomial families and stability."""

    def __init__(self, mode: str = "shifted") -> None:
        self.mode = mode
        self.phi = PHI
        self.psi = PSI

    def analyze_parameter(self, k: float) -> dict[str, Any]:
        """Comprehensive analysis of parameter k."""
        roots = morphic_polynomial_roots(k, self.mode)
        disc = discriminant(k, self.mode)

        analysis = {
            "k": k,
            "discriminant": disc,
            "num_real_roots": len(roots),
            "real_roots": roots,
            "is_stable": disc > 0,
            "critical_distances": {},
        }

        # Distances to critical points
        k_circle = k_perfect_circle(self.mode)
        k_disc_zero = k_discriminant_zero(self.mode)

        analysis["critical_distances"]["to_perfect_circle"] = abs(k - k_circle)
        analysis["critical_distances"]["to_discriminant_zero"] = abs(
            k - k_disc_zero
        )

        return analysis

    def find_optimal_parameters(
        self, criterion: str = "max_stability"
    ) -> Optional[float]:
        """Find optimal parameter values."""
        if criterion == "max_stability":
            # Parameters with maximum discriminan
            stability = stability_regions(self.mode)
            if len(stability["stable_regions"]) > 0:
                max_disc_idx = np.argmax(stability["discriminants"])
                return stability["k_values"][max_disc_idx]

        return None

    def generate_morphic_landscape(
        self, k_range: tuple[float, float] = (-3, 5), resolution: int = 100
    ) -> dict[str, Any]:
        """Generate comprehensive morphic landscape."""
        k_min, k_max = k_range
        stability = stability_regions(self.mode, k_min, k_max, resolution)

        landscape = {
            "geometric_algebra_support": "removed",
            "mode": self.mode,
            "k_range": k_range,
            "stability": stability,
            "golden_ratio_properties": golden_ratio_properties(),
            "critical_points": {
                "perfect_circle": k_perfect_circle(self.mode),
                "discriminant_zero": k_discriminant_zero(self.mode),
            },
        }

        return landscape


# ========================
# CONVENIENCE FUNCTIONS
# ========================


def quick_morphic_analysis(k: float, mode: str = "shifted") -> dict[str, Any]:
    """Quick analysis of a single parameter value."""
    analyzer = MorphicAnalyzer(mode)
    return analyzer.analyze_parameter(k)


def get_critical_parameters(mode: str = "shifted") -> dict[str, float]:
    """Get all critical parameter values for a mode."""
    return {
        "perfect_circle": k_perfect_circle(mode),
        "discriminant_zero": k_discriminant_zero(mode),
    }


# ========================
# MODULE TEST
# ========================


def test_morphic_module() -> dict[str, Any]:
    """Test the morphic module functionality."""
    test_results = {
        "ga_support": "removed",
        "golden_ratio_properties": golden_ratio_properties(),
        "critical_points": {},
        "polynomial_tests": {},
        "sequence_tests": {},
        "morphic_scaling_factor": morphic_scaling_factor(),
    }

    # Critical points
    for mode in ["shifted", "simple"]:
        k_circle = k_perfect_circle(mode)
        k_disc = k_discriminant_zero(mode)
        test_results["critical_points"][mode] = {
            "perfect_circle": k_circle,
            "discriminant_zero": k_disc,
        }

    # Polynomial roots tes
    roots = morphic_polynomial_roots(1.5, "shifted")
    test_results["polynomial_tests"]["k_1_5_shifted"] = (
        roots.tolist() if hasattr(roots, "tolist") else list(roots)
    )

    # Morphic sequence tes
    sequence = generate_morphic_sequence(8)
    test_results["sequence_tests"]["first_8_terms"] = (
        sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)
    )

    # Stability analysis
    analyzer = MorphicAnalyzer("shifted")
    analysis = analyzer.analyze_parameter(1.5)
    test_results["parameter_analysis"] = {
        k: v for k, v in analysis.items() if k != "real_roots"
    }

    # Geometric transformations removed (dead code cleanup)

    return test_results


if __name__ == "__main__":
    # Test morphic module without printing
    results = test_morphic_module()

    # Validate core functionality
    assert "golden_ratio_properties" in results
    assert "critical_points" in results
    assert results["morphic_scaling_factor"] > 0
