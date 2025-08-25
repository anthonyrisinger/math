#!/usr/bin/env python3
"""
Morphic Mathematics
===================

Morphic polynomials, golden ratio operations, and geometric stability
calculations. Implements the polynomial families that create stable
reference frames for fractional dimensions.

Core polynomial families:
- "shifted": τ³ - (2-k)τ - 1 = 0 (primary family)
- "simple":  τ³ - kτ - 1 = 0 (alternative family)

The golden ratio φ and its conjugate ψ = 1/φ create stability through:
φ² = φ + 1
ψ² = 1 - ψ
"""

import numpy as np
from .constants import PHI, PSI, NUMERICAL_EPSILON

def morphic_polynomial_roots(k, mode="shifted"):
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
        coeffs = [1.0, 0.0, -k, -1.0]           # τ³ + 0τ² - kτ - 1
    else:
        raise ValueError("mode must be 'shifted' or 'simple'")

    # Find all roots
    roots = np.roots(coeffs)

    # Extract real roots (filter out complex with small imaginary part)
    real_roots = roots.real[np.abs(roots.imag) < NUMERICAL_EPSILON]

    # Sort in descending order
    return np.sort(real_roots)[::-1]

def discriminant(k, mode="shifted"):
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
        return -4*a*(c**3) - 27*(a**2)*(d**2)
    elif mode == "simple":
        # τ³ - kτ - 1 = 0
        # a = 1, b = 0, c = -k, d = -1
        a, c, d = 1.0, -k, -1.0
        return -4*a*(c**3) - 27*(a**2)*(d**2)
    else:
        raise ValueError("mode must be 'shifted' or 'simple'")

def k_perfect_circle(mode="shifted"):
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

def k_discriminant_zero(mode="shifted"):
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
        return 2.0 - (27.0/4.0)**(1.0/3.0)
    elif mode == "simple":
        # Solve: -4(-k)³ - 27 = 0
        # 4k³ = 27
        # k³ = 27/4
        # k = (27/4)^(1/3)
        return (27.0/4.0)**(1.0/3.0)
    else:
        raise ValueError("mode must be 'shifted' or 'simple'")

def golden_ratio_properties():
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
        'phi': phi,
        'psi': psi,
        'phi_squared': phi**2,
        'phi_plus_one': phi + 1,
        'phi_squared_equals_phi_plus_one': abs(phi**2 - (phi + 1)) < NUMERICAL_EPSILON,
        'psi_squared': psi**2,
        'one_minus_psi': 1 - psi,
        'psi_squared_equals_one_minus_psi': abs(psi**2 - (1 - psi)) < NUMERICAL_EPSILON,
        'phi_times_psi': phi * psi,
        'phi_times_psi_equals_one': abs(phi * psi - 1) < NUMERICAL_EPSILON,
        'phi_minus_psi': phi - psi,
        'phi_minus_psi_equals_one': abs((phi - psi) - 1.0) < NUMERICAL_EPSILON,
        'phi_plus_psi_equals_sqrt5': abs((phi + psi) - np.sqrt(5)) < NUMERICAL_EPSILON
    }

def morphic_scaling_factor(phi=PHI):
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
        Scaling factor
    """
    return phi**(1/phi)  # φ^(1/φ) ≈ 1.465

def stability_regions(mode="shifted", k_min=-5, k_max=5, num_points=1000):
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
        'k_values': k_values,
        'discriminants': [],
        'num_real_roots': [],
        'stable_regions': [],
        'critical_points': {}
    }

    for k in k_values:
        disc = discriminant(k, mode)
        roots = morphic_polynomial_roots(k, mode)

        results['discriminants'].append(disc)
        results['num_real_roots'].append(len(roots))

    results['discriminants'] = np.array(results['discriminants'])
    results['num_real_roots'] = np.array(results['num_real_roots'])

    # Find critical points
    results['critical_points']['perfect_circle'] = k_perfect_circle(mode)
    results['critical_points']['discriminant_zero'] = k_discriminant_zero(mode)

    # Stable regions (where discriminant > 0, indicating 3 real roots)
    stable_mask = results['discriminants'] > 0
    results['stable_regions'] = k_values[stable_mask]

    return results

def morphic_transformation_matrix(tau, phi=PHI):
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
    scale = tau * phi**(1/phi)
    angle = np.pi / phi  # Golden angle related rotation

    cos_a, sin_a = np.cos(angle), np.sin(angle)

    return scale * np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])

def generate_morphic_sequence(n_terms, phi=PHI):
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
        sequence[i] = phi * sequence[i-1] + psi * sequence[i-2]

    return sequence

def curvature_peak_estimate(tau, dtheta=0.1):
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
    theta = np.arange(0, 2*np.pi, dtheta)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)

    # Apply morphic transformation
    transform = morphic_transformation_matrix(tau)
    circle_points = np.column_stack([x_circle, y_circle])
    transformed = circle_points @ transform.T

    # Compute curvature
    x, y = transformed[:, 0], transformed[:, 1]

    with np.errstate(all="ignore"):
        dx = np.gradient(x, dtheta)
        dy = np.gradient(y, dtheta)
        ddx = np.gradient(dx, dtheta)
        ddy = np.gradient(dy, dtheta)

        # Curvature formula: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**(3/2)

        # Avoid division by zero
        denominator[denominator == 0] = np.nan
        kappa = numerator / denominator

        # Return maximum finite curvature
        finite_kappa = kappa[np.isfinite(kappa)]
        return float(np.max(finite_kappa)) if len(finite_kappa) > 0 else 0.0

class MorphicAnalyzer:
    """
    Complete morphic mathematics analyzer.

    Provides comprehensive analysis of morphic polynomial families,
    stability regions, and geometric transformations.
    """

    def __init__(self, mode="shifted"):
        self.mode = mode
        self.phi = PHI
        self.psi = PSI

    def analyze_parameter(self, k):
        """Comprehensive analysis of parameter k."""
        roots = morphic_polynomial_roots(k, self.mode)
        disc = discriminant(k, self.mode)

        analysis = {
            'k': k,
            'discriminant': disc,
            'num_real_roots': len(roots),
            'real_roots': roots,
            'is_stable': disc > 0,
            'critical_distances': {}
        }

        # Distances to critical points
        k_circle = k_perfect_circle(self.mode)
        k_disc_zero = k_discriminant_zero(self.mode)

        analysis['critical_distances']['to_perfect_circle'] = abs(k - k_circle)
        analysis['critical_distances']['to_discriminant_zero'] = abs(k - k_disc_zero)

        return analysis

    def find_optimal_parameters(self, criterion='max_stability'):
        """Find optimal parameter values."""
        if criterion == 'max_stability':
            # Parameters with maximum discriminant
            stability = stability_regions(self.mode)
            if len(stability['stable_regions']) > 0:
                max_disc_idx = np.argmax(stability['discriminants'])
                return stability['k_values'][max_disc_idx]

        return None

if __name__ == "__main__":
    print("MORPHIC MATHEMATICS TEST")
    print("=" * 50)

    # Golden ratio properties
    props = golden_ratio_properties()
    print("Golden ratio properties:")
    for key, value in props.items():
        if isinstance(value, bool):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.6f}")

    print("\nCritical points:")
    for mode in ["shifted", "simple"]:
        k_circle = k_perfect_circle(mode)
        k_disc = k_discriminant_zero(mode)
        print(f"  {mode}: perfect circle k={k_circle:.3f}, discriminant zero k={k_disc:.3f}")

    # Test polynomial roots
    print(f"\nPolynomial roots for k=1.5 (shifted):")
    roots = morphic_polynomial_roots(1.5, "shifted")
    print(f"  Real roots: {roots}")

    # Stability analysis
    analyzer = MorphicAnalyzer("shifted")
    analysis = analyzer.analyze_parameter(1.5)
    print(f"\nParameter k=1.5 analysis:")
    for key, value in analysis.items():
        if key != 'real_roots':
            print(f"  {key}: {value}")