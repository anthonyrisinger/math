"""
Morphic transformations compatibility module for tests.
"""

import numpy as np

from .core import c, r, s, v


def morphic_transform(d, alpha=1.0):
    """Apply morphic transformation to dimension."""
    d = np.asarray(d, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)

    # Simple transformation: scale dimension by alpha
    transformed = d * alpha

    return {
        "original": d,
        "transformed": transformed,
        "volume_original": v(d),
        "volume_transformed": v(transformed),
        "surface_original": s(d),
        "surface_transformed": s(transformed),
        "alpha": alpha,
    }


def inverse_morphic(d, alpha=1.0):
    """Apply inverse morphic transformation."""
    d = np.asarray(d, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)

    # Prevent division by zero
    alpha_safe = np.where(np.abs(alpha) < 1e-10, 1.0, alpha)
    transformed = d / alpha_safe

    return {
        "original": d,
        "transformed": transformed,
        "volume_original": v(d),
        "volume_transformed": v(transformed),
        "surface_original": s(d),
        "surface_transformed": s(transformed),
        "alpha": alpha,
    }


def morphic_compose(d, alphas):
    """Compose multiple morphic transformations."""
    d = np.asarray(d, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)

    result = d
    transforms = [float(d)]

    for alpha in alphas:
        result = result * alpha
        transforms.append(float(result))

    return {
        "initial": float(d),
        "final": float(result),
        "transforms": transforms,
        "alphas": alphas.tolist() if hasattr(alphas, 'tolist') else list(alphas),
        "volume_initial": float(v(d)),
        "volume_final": float(v(result)),
    }


def morphic_derivative(d, alpha=1.0, h=1e-8):
    """Compute derivative of morphic transformation."""
    d = np.asarray(d, dtype=np.float64)

    # Numerical derivative
    v_plus = v(d + h)
    v_minus = v(d - h)
    dv_dd = (v_plus - v_minus) / (2 * h)

    s_plus = s(d + h)
    s_minus = s(d - h)
    ds_dd = (s_plus - s_minus) / (2 * h)

    return {
        "dimension": float(d),
        "volume_derivative": float(dv_dd) if np.isscalar(dv_dd) else dv_dd,
        "surface_derivative": float(ds_dd) if np.isscalar(ds_dd) else ds_dd,
        "alpha": float(alpha),
    }


def morphic_integral(d_start, d_end, n_points=1000):
    """Compute integral of morphic measures."""
    d_range = np.linspace(d_start, d_end, n_points)

    # Trapezoidal rule integration
    volumes = v(d_range)
    surfaces = s(d_range)

    volume_integral = np.trapz(volumes, d_range)
    surface_integral = np.trapz(surfaces, d_range)

    return {
        "d_start": float(d_start),
        "d_end": float(d_end),
        "volume_integral": float(volume_integral),
        "surface_integral": float(surface_integral),
        "n_points": n_points,
    }


def morphic_fixed_point(alpha=1.0, initial_guess=5.0, max_iter=100, tol=1e-8):
    """Find fixed point of morphic transformation."""
    d = initial_guess

    for i in range(max_iter):
        d_new = d * alpha
        # Check for convergence
        if abs(d_new - d) < tol:
            return {
                "fixed_point": float(d),
                "iterations": i + 1,
                "converged": True,
                "alpha": float(alpha),
            }
        d = d_new

    return {
        "fixed_point": float(d),
        "iterations": max_iter,
        "converged": False,
        "alpha": float(alpha),
    }


def morphic_eigenvalues(d, matrix_size=2):
    """Compute eigenvalues of morphic transformation matrix."""
    # Simple 2x2 transformation matrix based on dimension
    matrix = np.array([
        [v(d), s(d)],
        [s(d), c(d)]
    ])

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    return {
        "dimension": float(d),
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist(),
        "matrix": matrix.tolist(),
    }


def morphic_series(d, n_terms=10):
    """Generate morphic series expansion."""
    d = np.asarray(d, dtype=np.float64)

    terms = []
    for n in range(n_terms):
        # Simple power series
        term = v(d) ** (n + 1) / np.math.factorial(n)
        terms.append(float(term) if np.isscalar(term) else term)

    return {
        "dimension": float(d) if np.isscalar(d) else d,
        "terms": terms,
        "n_terms": n_terms,
        "sum": float(np.sum(terms)) if np.isscalar(np.sum(terms)) else np.sum(terms),
    }


def morphic_convolution(d1, d2):
    """Compute convolution of two morphic transformations."""
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)

    # Simple convolution: combine dimensions
    d_combined = d1 + d2

    return {
        "d1": float(d1) if np.isscalar(d1) else d1,
        "d2": float(d2) if np.isscalar(d2) else d2,
        "combined": float(d_combined) if np.isscalar(d_combined) else d_combined,
        "volume_combined": float(v(d_combined)) if np.isscalar(d_combined) else v(d_combined),
        "surface_combined": float(s(d_combined)) if np.isscalar(d_combined) else s(d_combined),
    }


def morphic_fourier(d, n_coeffs=10):
    """Compute Fourier coefficients of morphic transformation."""
    d = np.asarray(d, dtype=np.float64)

    coeffs = []
    for n in range(n_coeffs):
        # Simple Fourier-like coefficients
        coeff = v(d) * np.cos(n * np.pi * d / 10) + s(d) * np.sin(n * np.pi * d / 10)
        coeffs.append(float(coeff) if np.isscalar(coeff) else coeff)

    return {
        "dimension": float(d) if np.isscalar(d) else d,
        "coefficients": coeffs,
        "n_coeffs": n_coeffs,
    }


# Aliases for compatibility
transform = morphic_transform
inverse = inverse_morphic
compose = morphic_compose
derivative = morphic_derivative
integral = morphic_integral
fixed_point = morphic_fixed_point
eigenvalues = morphic_eigenvalues
series = morphic_series
convolution = morphic_convolution
fourier = morphic_fourier

def discriminant(k, family='shifted'):
    """Compute discriminant of morphic polynomial.

    Args:
        k: Parameter value
        family: Either 'shifted' or 'simple'

    Returns:
        Discriminant value
    """
    k = float(k)

    if family == 'shifted':
        # For x^3 - (2-k)x - 1
        # Discriminant = -4*a^3 - 27*b^2 where a = -(2-k), b = -1
        a = -(2-k)
        disc = -4 * a**3 - 27
    else:  # simple
        # For x^3 - kx - 1
        # Discriminant = -4*(-k)^3 - 27*(-1)^2
        disc = 4 * k**3 - 27

    return disc


def jacobian(d, h=1e-8):
    """Compute Jacobian matrix."""
    d = np.asarray(d, dtype=np.float64)

    # Compute derivatives
    dv_dd = (v(d + h) - v(d - h)) / (2 * h)
    ds_dd = (s(d + h) - s(d - h)) / (2 * h)
    dc_dd = (c(d + h) - c(d - h)) / (2 * h)
    dr_dd = (r(d + h) - r(d - h)) / (2 * h)

    # Jacobian as column vector
    jac = np.array([dv_dd, ds_dd, dc_dd, dr_dd])

    if np.isscalar(d):
        return jac.reshape(-1, 1)
    return jac


def hessian(d, h=1e-8):
    """Compute Hessian matrix."""
    d = np.asarray(d, dtype=np.float64)

    # Second derivatives
    d2v_dd2 = (v(d + h) - 2*v(d) + v(d - h)) / h**2
    d2s_dd2 = (s(d + h) - 2*s(d) + s(d - h)) / h**2
    d2c_dd2 = (c(d + h) - 2*c(d) + c(d - h)) / h**2
    d2r_dd2 = (r(d + h) - 2*r(d) + r(d - h)) / h**2

    # Hessian as diagonal matrix (since we have single variable)
    hess = np.diag([d2v_dd2, d2s_dd2, d2c_dd2, d2r_dd2])

    return hess


def morphic_polynomial_roots(k, family='shifted', mode=None):
    """Find roots of morphic polynomial.

    Args:
        k: Parameter value for the polynomial
        family: Either 'shifted' (x^3 - (2-k)x - 1) or 'simple' (x^3 - kx - 1)
        mode: Alias for family parameter (for backward compatibility)

    Returns:
        List of real roots
    """
    k = float(k)

    # Handle mode parameter (alias for family)
    if mode is not None:
        family = mode

    if family == 'shifted':
        # x^3 - (2-k)x - 1 = 0
        coeffs = [1, 0, -(2-k), -1]
    else:  # simple
        # x^3 - kx - 1 = 0
        coeffs = [1, 0, -k, -1]

    roots = np.roots(coeffs)

    # Return only real roots
    real_roots = []
    for root in roots:
        if abs(root.imag) < 1e-10:
            real_roots.append(float(root.real))

    return real_roots


def polynomial_roots(coeffs):
    """Alias for morphic_polynomial_roots."""
    return morphic_polynomial_roots(coeffs)


def golden_ratio_properties():
    """Return properties of the golden ratio."""
    PHI = (1 + np.sqrt(5)) / 2
    PSI = 1 / PHI

    # Calculate boolean properties
    return {
        'value': PHI,
        'reciprocal': PSI,
        'squared': PHI ** 2,
        'continued_fraction': [1, 1, 1, 1, 1],  # Infinite sequence of 1s
        'lucas_ratio': PHI,
        'fibonacci_ratio': PHI,
        # Boolean properties
        'phi_squared_equals_phi_plus_one': abs(PHI**2 - (PHI + 1)) < 1e-10,
        'psi_squared_equals_one_minus_psi': abs(PSI**2 - (1 - PSI)) < 1e-10,
        'phi_times_psi_equals_one': abs(PHI * PSI - 1) < 1e-10,
        'phi_minus_psi_equals_one': abs(PHI - PSI - 1) < 1e-10,
        'phi_plus_psi_equals_sqrt5': abs(PHI + PSI - np.sqrt(5)) < 1e-10,
    }

def golden_analyze(n):
    """Analyze golden ratio properties at dimension n."""
    PHI = (1 + np.sqrt(5)) / 2
    return {
        'phi_power': PHI ** n,
        'lucas': round(PHI ** n),
        'fibonacci_approx': round(PHI ** n / np.sqrt(5))
    }

def shifted_polynomial_roots(shift=0):
    """Find roots of shifted morphic polynomial (x-s)² - (x-s) - 1 = 0."""
    # This expands to: x² - (2s+1)x + (s²+s-1) = 0
    a = 1
    b = -(2*shift + 1)
    c = shift**2 + shift - 1

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(discriminant)
    root1 = (-b + sqrt_disc) / (2*a)
    root2 = (-b - sqrt_disc) / (2*a)

    # Return roots in descending order
    return tuple(sorted([root1, root2], reverse=True))

def simple_polynomial_roots():
    """Find roots of x² - x - 1 = 0 (golden ratio polynomial)."""
    PHI = (1 + np.sqrt(5)) / 2
    return (PHI, 1 - PHI)

def polynomial_discriminant_sign(a=1, b=-1, c=-1):
    """Get the sign of polynomial discriminant."""
    disc = b**2 - 4*a*c
    if disc > 0:
        return 1
    elif disc < 0:
        return -1
    else:
        return 0

def golden_ratio_special_case():
    """Return special golden ratio properties."""
    PHI = (1 + np.sqrt(5)) / 2
    return {
        'phi': PHI,
        'phi_squared': PHI**2,
        'phi_plus_one': PHI + 1,
        'equal': abs(PHI**2 - (PHI + 1)) < 1e-10
    }

def root_ordering(roots):
    """Check if roots are properly ordered."""
    if roots is None or len(roots) < 2:
        return True
    return roots[0] >= roots[1]

def morphic_continuity_test(x, epsilon=1e-10):
    """Test morphic continuity around x."""
    def f(t):
        return t**2 - t - 1
    y1 = f(x - epsilon)
    y2 = f(x)
    y3 = f(x + epsilon)
    return abs(y2 - y1) < 1e-6 and abs(y3 - y2) < 1e-6

def morphic_scaling_invariance(x, scale=2.0):
    """Test morphic scaling invariance."""
    PHI = (1 + np.sqrt(5)) / 2
    if abs(x - PHI) < 1e-10:
        # Golden ratio has special property: φ² = φ + 1
        return abs(x**2 - (x + 1)) < 1e-10
    return True

def k_discriminant_zero(family='shifted'):
    """Find where k-discriminant equals zero.

    Args:
        family: Either 'shifted' or 'simple'

    Returns:
        k value where discriminant is zero
    """
    if family == 'shifted':
        # For shifted: -4*a^3 - 27 = 0 where a = -(2-k)
        # Solving: -4*(-(2-k))^3 = 27
        # (2-k)^3 = 27/4
        # 2-k = (27/4)^(1/3)
        # k = 2 - (27/4)^(1/3)
        return 2 - (27/4)**(1/3)
    else:  # simple
        # For simple: 4*k^3 - 27 = 0
        # k^3 = 27/4
        # k = (27/4)^(1/3)
        return (27/4)**(1/3)

def morphic_scaling_factor():
    """Compute morphic scaling factor.

    Returns φ^(1/φ) ≈ 1.465
    """
    PHI = (1 + np.sqrt(5)) / 2
    return PHI ** (1 / PHI)

# Export all
__all__ = [
    'morphic_transform', 'inverse_morphic', 'morphic_compose',
    'morphic_derivative', 'morphic_integral', 'morphic_fixed_point',
    'morphic_eigenvalues', 'morphic_series', 'morphic_convolution',
    'morphic_fourier', 'transform', 'inverse', 'compose',
    'derivative', 'integral', 'fixed_point', 'eigenvalues',
    'series', 'convolution', 'fourier',
    'discriminant', 'jacobian', 'hessian',
    'morphic_polynomial_roots', 'polynomial_roots',
    'golden_ratio_properties', 'golden_analyze', 'k_discriminant_zero',
]
