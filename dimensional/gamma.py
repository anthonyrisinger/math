"""
Gamma function compatibility module for tests.
"""

import numpy as np
from scipy.special import beta as scipy_beta
from scipy.special import digamma as scipy_digamma
from scipy.special import gamma as scipy_gamma
from scipy.special import gammaln as scipy_gammaln

# Import from core
from .core import c, r, s, v


def gamma(z):
    """Gamma function with array support."""
    z = np.asarray(z, dtype=np.float64)
    result = scipy_gamma(z)
    return float(result) if np.isscalar(z) else result


def gammaln(z):
    """Log-gamma function."""
    z = np.asarray(z, dtype=np.float64)
    result = scipy_gammaln(z)
    return float(result) if np.isscalar(z) else result


def digamma(z):
    """Digamma function."""
    z = np.asarray(z, dtype=np.float64)
    result = scipy_digamma(z)
    return float(result) if np.isscalar(z) else result


def beta_function(a, b):
    """Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    result = scipy_beta(a, b)
    if np.isscalar(a) and np.isscalar(b):
        return float(result)
    return result


def factorial_extension(n):
    """Factorial extension n! = Γ(n+1)."""
    n = np.asarray(n, dtype=np.float64)
    result = scipy_gamma(n + 1)
    # Handle negative integers
    neg_int_mask = (n < 0) & (np.abs(n - np.round(n)) < 1e-10)
    if np.any(neg_int_mask):
        result = np.where(neg_int_mask, np.inf, result)
    return float(result) if np.isscalar(n) else result


def batch_gamma_operations(z):
    """Compute multiple gamma functions at once."""
    z = np.asarray(z, dtype=np.float64)
    return {
        "gamma": gamma(z),
        "ln_gamma": gammaln(z),
        "digamma": digamma(z),
        "factorial": gamma(z + 1)  # n! = Γ(n+1)
    }


def clear_cache():
    """Clear computation cache."""
    pass  # No-op for compatibility

def get_cache_info():
    """Get cache information."""
    return {}  # Empty dict for compatibility

def explore(d, use_cache=True):
    """Explore dimensional properties."""
    if np.isscalar(d):
        d_float = float(d)
        return {
            "dimension": d_float,
            "volume": float(v(d_float)),
            "surface": float(s(d_float)),
            "complexity": float(c(d_float)),
            "ratio": float(r(d_float)),
            "density": 1.0 / float(v(d_float)) if v(d_float) != 0 else np.inf,
            "gamma": float(gamma(d_float)) if d_float > 0 else None,
        }

    # Array input
    d_array = np.asarray(d, dtype=np.float64)
    volumes = v(d_array)
    surfaces = s(d_array)
    complexities = c(d_array)
    ratios = r(d_array)

    results = []
    for i, dim in enumerate(d_array):
        results.append({
            "dimension": float(dim),
            "volume": float(volumes[i]),
            "surface": float(surfaces[i]),
            "complexity": float(complexities[i]),
            "ratio": float(ratios[i]),
            "density": 1.0 / float(volumes[i]) if volumes[i] != 0 else np.inf,
            "gamma": float(gamma(dim)) if dim > 0 else None,
        })

    return results


def convergence_diagnostics(d_range=None, measure="volume", threshold=1e-10):
    """Analyze convergence behavior."""
    if d_range is None:
        d_range = np.arange(1, 101, 1)
    else:
        d_range = np.asarray(d_range)

    measure_func = {"volume": v, "surface": s, "complexity": c}.get(measure, v)
    values = measure_func(d_range)

    zero_idx = np.where(values < threshold)[0]
    converge_dim = float(d_range[zero_idx[0]]) if len(zero_idx) > 0 else None

    if len(values) > 1:
        log_values = np.log(values[values > 0])
        if len(log_values) > 1:
            dims_positive = d_range[values > 0]
            decay_rate = np.polyfit(dims_positive, log_values, 1)[0]
        else:
            decay_rate = None
    else:
        decay_rate = None

    return {
        "measure": measure,
        "threshold": threshold,
        "convergence_dimension": converge_dim,
        "decay_rate": decay_rate,
        "final_value": float(values[-1]),
        "max_value": float(np.max(values)),
        "max_dimension": float(d_range[np.argmax(values)]),
    }


def find_peak(measure_func):
    """Find peak of a measure function."""
    from scipy.optimize import minimize_scalar

    if measure_func == v:
        bounds = (1, 10)
    elif measure_func == s:
        bounds = (1, 12)
    elif measure_func == c:
        bounds = (1, 15)
    else:
        bounds = (0.1, 20)

    result = minimize_scalar(lambda d: -measure_func(d), bounds=bounds, method='bounded')
    return (result.x, -result.fun)


def v_peak():
    """Find volume peak."""
    return find_peak(v)


def s_peak():
    """Find surface peak."""
    return find_peak(s)


def c_peak():
    """Find complexity peak."""
    return find_peak(c)


def qplot(*functions, labels=None, d_range=None):
    """Quick plot of functions."""
    if d_range is None:
        d_range = np.linspace(0.1, 20, 100)
    else:
        d_range = np.asarray(d_range)

    result = {}
    for i, func in enumerate(functions):
        label = labels[i] if labels and i < len(labels) else f"Function {i+1}"
        result[label] = func(d_range) if callable(func) else np.full_like(d_range, func)

    return result

def peaks():
    """Find all peaks."""
    return {
        "volume_peak": v_peak(),
        "surface_peak": s_peak(),
        "complexity_peak": c_peak(),
        "volume": v_peak(),
        "surface": s_peak(),
        "complexity": c_peak(),
    }


# Aliases for compatibility
gamma_safe = gamma
gammaln_safe = gammaln
factorial = factorial_extension
beta = beta_function

# Stub functions for CLI compatibility
def demo():
    print("Gamma demonstration")
    for i in range(1, 6):
        print(f"gamma({i}) = {gamma(i)}")

def lab(d=None, interactive=False):
    if d is None:
        d = 4.0
    return {
        "current": explore(d),
        "peaks": peaks(),
        "convergence": convergence_diagnostics(),
        "interactive": interactive,
    }

def instant(d_range=None):
    if d_range is None:
        d_range = np.linspace(0.1, 20, 200)
    else:
        d_range = np.asarray(d_range)

    return {
        "dimensions": d_range.tolist(),
        "volume": v(d_range).tolist(),
        "surface": s(d_range).tolist(),
        "complexity": c(d_range).tolist(),
        "ratio": r(d_range).tolist(),
    }

def live(expr_file):
    print(f"Live mode for {expr_file}")


def quick_gamma_analysis(z):
    return {
        'dimension': float(z),
        'gamma_value': float(gamma(z)),
        'log_gamma': float(gammaln(z)),
        'digamma': float(digamma(z)),
        'factorial': float(factorial_extension(z-1)) if z > 0 else None,
        'gamma': float(gamma(z)),
        'ln_gamma': float(gammaln(z)),
    }

def fractional_domain_validation(z):
    """Validate fractional domain inputs."""
    z = np.asarray(z, dtype=np.float64)

    # Check for negative values
    if np.any(z <= 0):
        invalid = z[z <= 0]
        return {
            'valid': False,
            'invalid_values': invalid.tolist() if hasattr(invalid, 'tolist') else [float(invalid)],
            'message': f"Gamma function undefined for non-positive values: {invalid}"
        }

    return {
        'valid': True,
        'invalid_values': [],
        'message': "All values valid"
    }


def special_function_accuracy(z, reference_func=scipy_gamma):
    """Check accuracy of special function computation."""
    z = np.asarray(z, dtype=np.float64)

    computed = gamma(z)
    reference = reference_func(z)

    abs_error = np.abs(computed - reference)
    rel_error = abs_error / (np.abs(reference) + 1e-10)

    return {
        'absolute_error': float(np.max(abs_error)),
        'relative_error': float(np.max(rel_error)),
        'mean_abs_error': float(np.mean(abs_error)),
        'mean_rel_error': float(np.mean(rel_error)),
    }


# Export all
__all__ = [
    'gamma', 'gammaln', 'digamma', 'beta_function', 'factorial_extension',
    'batch_gamma_operations', 'explore', 'convergence_diagnostics',
    'find_peak', 'v_peak', 's_peak', 'c_peak', 'peaks',
    'gamma_safe', 'gammaln_safe', 'factorial', 'beta',
    'demo', 'lab', 'instant', 'live', 'qplot',
    'clear_cache', 'get_cache_info', 'quick_gamma_analysis',
    'fractional_domain_validation', 'special_function_accuracy',
    'v', 's', 'c', 'r'
]
