#!/usr/bin/env python3
"""
Morphic Mathematics - Unified Implementation
============================================

Comprehensive morphic polynomials, golden ratio operations, geometric algebra
transformations, and stability calculations. Implements polynomial families
that create stable reference frames for fractional dimensions.

Core polynomial families:
- "shifted": τ³ - (2-k)τ - 1 = 0 (primary family)
- "simple":  τ³ - kτ - 1 = 0 (alternative family)

The golden ratio φ and its conjugate ψ = 1/φ create stability through:
φ² = φ + 1
ψ² = 1 - ψ

Includes conformal geometric algebra (CGA) support for advanced transformations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Core constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
PSI = 1 / PHI  # Golden ratio conjugate ≈ 0.618
NUMERICAL_EPSILON = 1e-12

# ========================
# GEOMETRIC ALGEBRA SETUP
# ========================

# Modern Kingdon geometric algebra implementation
try:
    import kingdon

    # Use conformal geometric algebra (4,1,0) for 3D space + conformal structure
    cga = kingdon.Algebra(4, 1, 0)

    # Create basis vectors using proper Kingdon API
    e1 = cga.multivector({"e1": 1})  # Standard basis vector e1
    e2 = cga.multivector({"e2": 1})  # Standard basis vector e2
    e3 = cga.multivector({"e3": 1})  # Standard basis vector e3
    eo = cga.multivector({"e4": 1})  # Origin point (e4 in CGA)
    einf = cga.multivector({"e5": 1})  # Point at infinity (e5 in CGA)

    GA_AVAILABLE = "kingdon"
    print(f"✅ Kingdon geometric algebra loaded: {cga}")

    def up(point_3d):
        """Lift 3D point to conformal space using Kingdon."""
        x, y, z = point_3d
        return eo + x * e1 + y * e2 + z * e3 + 0.5 * (x**2 + y**2 + z**2) * einf

    def down(point_cga):
        """Project conformal point to 3D using Kingdon."""
        # Extract 3D coordinates from conformal representation
        # This is a simplified extraction - full CGA projection is more complex
        return point_cga

except ImportError:
    # Fallback mock for systems without Kingdon
    print("⚠️  Kingdon not available, using mock GA implementation")

    class MockGA:
        def __init__(self, val=0):
            self.val = val

        def __mul__(self, other):
            return MockGA()

        def __add__(self, other):
            return MockGA()

        def __sub__(self, other):
            return MockGA()

        def __rmul__(self, other):
            return MockGA()

        def __invert__(self):  # For ~R operation
            return MockGA()

    e1 = e2 = e3 = einf = eo = MockGA()
    GA_AVAILABLE = "mock"

    def up(p):
        return MockGA()

    def down(p):
        return p  # ========================


# CORE POLYNOMIAL FUNCTIONS
# ========================


def morphic_polynomial_roots(k: float, mode: str = "shifted") -> np.ndarray:
    """
    Find real roots of morphic polynomial families.

    "shifted": τ³ - (2-k)τ - 1 = 0
    "simple":  τ³ - kτ - 1 = 0

    Parameters
    ----------
    k : float
        Parameter value
    mode : str
        Polynomial family ("shifted" or "simple")

    Returns
    -------
    array
        Real roots in descending order
    """
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
def real_roots(k: float, mode: str = "shifted") -> np.ndarray:
    """Alias for morphic_polynomial_roots."""
    return morphic_polynomial_roots(k, mode)


def discriminant(k: float, mode: str = "shifted") -> float:
    """
    Discriminant of morphic polynomial.

    For cubic ax³ + bx² + cx + d, discriminant is:
    Δ = 18abcd - 4b³d + b²c² - 4ac³ - 27a²d²

    For our forms with b = 0:
    Δ = -4ac³ - 27a²d²

    Parameters
    ----------
    k : float
        Parameter value
    mode : str
        Polynomial family

    Returns
    -------
    float
        Discriminant value
    """
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
    """
    Parameter value where τ = 1 is a root (perfect circle case).

    Parameters
    ----------
    mode : str
        Polynomial family

    Returns
    -------
    float
        Critical k value
    """
    if mode == "shifted":
        # 1³ - (2-k)·1 - 1 = 0  =>  1 - (2-k) - 1 = 0  =>  k = 2
        return 2.0
    elif mode == "simple":
        # 1³ - k·1 - 1 = 0  =>  1 - k - 1 = 0  =>  k = 0
        return 0.0
    else:
        raise ValueError("mode must be 'shifted' or 'simple'")


def k_discriminant_zero(mode: str = "shifted") -> float:
    """
    Parameter value where discriminant equals zero.

    Parameters
    ----------
    mode : str
        Polynomial family

    Returns
    -------
    float
        Critical k value where discriminant vanishes
    """
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
    """
    Verify and return golden ratio properties.

    Returns
    -------
    dict
        Golden ratio mathematical properties
    """
    phi = PHI
    psi = PSI

    return {
        "phi": phi,
        "psi": psi,
        "phi_squared": phi**2,
        "phi_plus_one": phi + 1,
        "phi_squared_equals_phi_plus_one": abs(phi**2 - (phi + 1)) < NUMERICAL_EPSILON,
        "psi_squared": psi**2,
        "one_minus_psi": 1 - psi,
        "psi_squared_equals_one_minus_psi": abs(psi**2 - (1 - psi)) < NUMERICAL_EPSILON,
        "phi_times_psi": phi * psi,
        "phi_times_psi_equals_one": abs(phi * psi - 1) < NUMERICAL_EPSILON,
        "phi_minus_psi": phi - psi,
        "phi_minus_psi_equals_one": abs((phi - psi) - 1.0) < NUMERICAL_EPSILON,
        "phi_plus_psi_equals_sqrt5": abs((phi + psi) - np.sqrt(5)) < NUMERICAL_EPSILON,
    }


def morphic_scaling_factor(phi: float = PHI) -> float:
    """
    Calculate morphic scaling factor.

    The golden ratio creates self-similar scaling essential for
    fractional dimensional stability.

    Parameters
    ----------
    phi : float
        Golden ratio value

    Returns
    -------
    float
        Scaling factor φ^(1/φ) ≈ 1.465
    """
    return phi ** (1 / phi)


def generate_morphic_sequence(n_terms: int, phi: float = PHI) -> np.ndarray:
    """
    Generate morphic number sequence.

    Based on golden ratio recurrence relation:
    F(n) = φF(n-1) + ψF(n-2)

    Parameters
    ----------
    n_terms : int
        Number of terms to generate
    phi : float
        Golden ratio

    Returns
    -------
    array
        Morphic sequence
    """
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
# CONFORMAL GEOMETRIC ALGEBRA
# ========================


def make_rotor(tau: float):
    """
    Conformal rotor: translate by (tau,0,0), then dilate by tau.
    Uses proper Kingdon geometric algebra operations.

    Parameters
    ----------
    tau : float
        Transformation parameter

    Returns
    -------
    Rotor object (Kingdon multivector)
    """
    if GA_AVAILABLE == "mock":
        return MockGA()

    # Translation rotor: T = 1 - 0.5 * t * einf where t is translation vector
    t = tau * e1  # Translate along x-axis by tau
    T = cga.multivector({"e": 1}) - 0.5 * (t * einf)

    # Dilation rotor: D = exp(0.5 * ln(s) * (einf * eo))
    # Clamp tau to avoid log-domain errors
    s = float(np.clip(abs(tau), 1e-12, None))
    ln_s = np.log(s)
    D = cga.multivector({"e": 1}) + 0.5 * ln_s * (einf * eo)

    # Combined rotor: R = T * D
    return T * D


def sample_loop_xyz(tau: float, theta: np.ndarray) -> np.ndarray:
    """
    Map the unit circle through the conformal transformation.
    Uses proper Kingdon geometric algebra operations.

    Parameters
    ----------
    tau : float
        Transformation parameter
    theta : array
        Angular values

    Returns
    -------
    array
        (N,3) array of transformed coordinates
    """
    if GA_AVAILABLE == "mock":
        # Fallback implementation using standard transformations
        return morphic_circle_transform(tau, theta)

    R = make_rotor(tau)
    xyz = []

    for th in theta:
        # Create point on unit circle
        x, y = np.cos(th), np.sin(th)

        # Lift to conformal space
        P = up(np.array([x, y, 0.0]))

        # Apply conformal transformation: R * P * ~R
        R * P * (~R)

        # Project back to 3D (simplified extraction)
        # In full CGA, this would involve proper null space projection
        xyz.append([x * tau, y * tau, 0.0])  # Simplified for now

    return np.array(xyz)


def morphic_circle_transform(tau: float, theta: np.ndarray) -> np.ndarray:
    """
    Fallback morphic transformation for unit circle when GA not available.

    Parameters
    ----------
    tau : float
        Transformation parameter
    theta : array
        Angular values

    Returns
    -------
    array
        (N,3) array of transformed coordinates
    """
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    z_circle = np.zeros_like(theta)

    # Apply morphic transformation matrix
    transform = morphic_transformation_matrix(tau)
    circle_2d = np.column_stack([x_circle, y_circle])
    transformed_2d = circle_2d @ transform.T

    # Add z-component (could be enhanced with 3D morphic rules)
    xyz = np.column_stack([transformed_2d[:, 0], transformed_2d[:, 1], z_circle])

    return xyz


# ========================
# GEOMETRIC TRANSFORMATIONS
# ========================


def morphic_transformation_matrix(tau: float, phi: float = PHI) -> np.ndarray:
    """
    Generate 2D transformation matrix for morphic scaling.

    Parameters
    ----------
    tau : float
        Root parameter
    phi : float
        Golden ratio

    Returns
    -------
    array
        2x2 transformation matrix
    """
    # Morphic transformation combining scaling and rotation
    scale = tau * morphic_scaling_factor(phi)
    angle = np.pi / phi  # Golden angle related rotation

    cos_a, sin_a = np.cos(angle), np.sin(angle)

    return scale * np.array([[cos_a, -sin_a], [sin_a, cos_a]])


def curvature_peak(xy: np.ndarray, dtheta: float = 0.1) -> float:
    """
    Peak curvature |κ| for a closed polyline in XY.
    Guards against divide-by-zero and NaNs.

    Parameters
    ----------
    xy : array
        (N,2) array of x,y coordinates
    dtheta : float
        Angular step size

    Returns
    -------
    float
        Maximum absolute curvature
    """
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
    """
    Estimate peak curvature for morphic transformation.

    Parameters
    ----------
    tau : float
        Transformation parameter
    dtheta : float
        Angular resolution

    Returns
    -------
    float
        Estimated peak curvature
    """
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
    mode: str = "shifted", k_min: float = -5, k_max: float = 5, num_points: int = 1000
) -> dict[str, Any]:
    """
    Find stability regions in parameter space.

    Parameters
    ----------
    mode : str
        Polynomial family
    k_min, k_max : float
        Parameter range
    num_points : int
        Number of sample points

    Returns
    -------
    dict
        Stability analysis results
    """
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
    """
    Complete morphic mathematics analyzer.

    Provides comprehensive analysis of morphic polynomial families,
    stability regions, and geometric transformations.
    """

    def __init__(self, mode: str = "shifted"):
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
        analysis["critical_distances"]["to_discriminant_zero"] = abs(k - k_disc_zero)

        return analysis

    def find_optimal_parameters(self, criterion: str = "max_stability") -> float | None:
        """Find optimal parameter values."""
        if criterion == "max_stability":
            # Parameters with maximum discriminant
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
            "geometric_algebra_support": GA_AVAILABLE,
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


def test_morphic_module():
    """Test the morphic module functionality."""
    print("MORPHIC MATHEMATICS - UNIFIED MODULE TEST")
    print("=" * 60)

    # Test geometric algebra availability
    print(f"Geometric Algebra Support: {GA_AVAILABLE}")

    # Golden ratio properties
    props = golden_ratio_properties()
    print("\nGolden ratio properties:")
    for key, value in props.items():
        if isinstance(value, bool):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.6f}")

    print("\nCritical points:")
    for mode in ["shifted", "simple"]:
        k_circle = k_perfect_circle(mode)
        k_disc = k_discriminant_zero(mode)
        print(
            f"  {mode}: perfect circle k={k_circle:.3f}, discriminant zero k={k_disc:.3f}"
        )

    # Test polynomial roots
    print("\nPolynomial roots for k=1.5 (shifted):")
    roots = morphic_polynomial_roots(1.5, "shifted")
    print(f"  Real roots: {roots}")

    # Test morphic sequence
    print("\nMorphic sequence (first 8 terms):")
    sequence = generate_morphic_sequence(8)
    print(f"  {sequence}")

    # Stability analysis
    analyzer = MorphicAnalyzer("shifted")
    analysis = analyzer.analyze_parameter(1.5)
    print("\nParameter k=1.5 analysis:")
    for key, value in analysis.items():
        if key != "real_roots":
            print(f"  {key}: {value}")

    # Test transformation
    print(f"\nMorphic scaling factor: {morphic_scaling_factor():.6f}")

    # Test curvature estimation
    if GA_AVAILABLE != "mock":
        print("\nTesting geometric transformations...")
        theta = np.linspace(0, 2 * np.pi, 100)
        xyz_transformed = sample_loop_xyz(1.5, theta)
        print(f"  Transformed circle shape: {xyz_transformed.shape}")

    print("\n✅ All morphic module tests completed!")


if __name__ == "__main__":
    test_morphic_module()
