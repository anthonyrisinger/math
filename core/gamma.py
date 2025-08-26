#!/usr/bin/env python3
"""
Gamma Function Family
=====================

Safe, numerically stable implementations of the gamma function family.
Handles edge cases, large values, and provides unified interface for
all gamma-related calculations used throughout the framework.
"""

import numpy as np
from scipy.special import digamma, gamma, gammaln, polygamma

from .constants import GAMMA_OVERFLOW_THRESHOLD, LOG_SPACE_THRESHOLD, NUMERICAL_EPSILON


def gamma_safe(z):
    """
    Numerically stable gamma function.

    Parameters
    ----------
    z : float or array-like
        Input values

    Returns
    -------
    float or array
        Γ(z) with proper handling of edge cases and overflow
    """
    z = np.asarray(z)

    # Handle edge cases
    if np.any(z == 0):
        # Γ(0) is undefined (pole)
        result = np.full_like(z, np.inf, dtype=float)
        mask = z != 0
        if np.any(mask):
            result[mask] = gamma_safe(z[mask])
        return result if z.ndim > 0 else float(result)

    # Handle negative integers (poles)
    if np.any((z < 0) & (np.abs(z - np.round(z)) < NUMERICAL_EPSILON)):
        result = np.full_like(z, np.inf, dtype=float)
        mask = ~((z < 0) & (np.abs(z - np.round(z)) < NUMERICAL_EPSILON))
        if np.any(mask):
            result[mask] = gamma_safe(z[mask])
        return result if z.ndim > 0 else float(result)

    # Use log-space for large values to avoid overflow
    if np.any(np.abs(z) > GAMMA_OVERFLOW_THRESHOLD):
        large_mask = np.abs(z) > GAMMA_OVERFLOW_THRESHOLD
        result = np.zeros_like(z, dtype=float)

        # Small values: direct computation
        if np.any(~large_mask):
            result[~large_mask] = gamma(z[~large_mask])

        # Large values: exp(log(gamma))
        if np.any(large_mask):
            log_gamma_vals = gammaln(z[large_mask])
            # Only exponentiate if the log value isn't too large
            exp_mask = log_gamma_vals < LOG_SPACE_THRESHOLD
            if np.any(exp_mask):
                if large_mask.ndim > 0:
                    large_indices = np.where(large_mask)[0]
                    safe_indices = large_indices[exp_mask]
                    result[safe_indices] = np.exp(log_gamma_vals[exp_mask])
                else:
                    result[()] = np.exp(log_gamma_vals)

            # For extremely large values, return inf
            inf_mask = log_gamma_vals >= LOG_SPACE_THRESHOLD
            if np.any(inf_mask):
                if large_mask.ndim > 0:
                    large_indices = np.where(large_mask)[0]
                    inf_indices = large_indices[inf_mask]
                    result[inf_indices] = np.inf
                else:
                    result[()] = np.inf

        return result if z.ndim > 0 else float(result)

    # Normal case
    return gamma(z)


def gammaln_safe(z):
    """
    Safe log-gamma function.

    Parameters
    ----------
    z : float or array-like
        Input values

    Returns
    -------
    float or array
        log(Γ(z)) with proper handling of edge cases
    """
    z = np.asarray(z)

    # Handle poles (zeros and negative integers)
    if np.any(z <= 0):
        if np.any(np.abs(z - np.round(z)) < NUMERICAL_EPSILON):
            # At poles, return -inf
            result = np.full_like(z, -np.inf, dtype=float)
            mask = ~(np.abs(z - np.round(z)) < NUMERICAL_EPSILON)
            if np.any(mask):
                result[mask] = gammaln_safe(z[mask])
            return result if z.ndim > 0 else float(result)

    return gammaln(z)


def digamma_safe(z):
    """
    Safe digamma function (psi function).
    ψ(z) = d/dz log(Γ(z)) = Γ'(z)/Γ(z)

    Parameters
    ----------
    z : float or array-like
        Input values

    Returns
    -------
    float or array
        ψ(z) with proper handling of edge cases
    """
    z = np.asarray(z)

    # Handle poles
    if np.any(z <= 0):
        if np.any(np.abs(z - np.round(z)) < NUMERICAL_EPSILON):
            result = np.full_like(z, -np.inf, dtype=float)
            mask = ~(np.abs(z - np.round(z)) < NUMERICAL_EPSILON)
            if np.any(mask):
                result[mask] = digamma_safe(z[mask])
            return result if z.ndim > 0 else float(result)

    return digamma(z)


def polygamma_safe(n, z):
    """
    Safe polygamma function.
    ψ^(n)(z) = d^(n+1)/dz^(n+1) log(Γ(z))

    Parameters
    ----------
    n : int
        Order of derivative (0 gives digamma)
    z : float or array-like
        Input values

    Returns
    -------
    float or array
        ψ^(n)(z) with proper handling of edge cases
    """
    z = np.asarray(z)

    # Handle poles
    if np.any(z <= 0):
        if np.any(np.abs(z - np.round(z)) < NUMERICAL_EPSILON):
            result = np.full_like(z, (-1) ** (n + 1) * np.inf, dtype=float)
            mask = ~(np.abs(z - np.round(z)) < NUMERICAL_EPSILON)
            if np.any(mask):
                result[mask] = polygamma_safe(n, z[mask])
            return result if z.ndim > 0 else float(result)

    return polygamma(n, z)


def gamma_ratio_safe(a, b):
    """
    Compute Γ(a)/Γ(b) safely using log-space when needed.

    Parameters
    ----------
    a, b : float or array-like
        Gamma function arguments

    Returns
    -------
    float or array
        Γ(a)/Γ(b)
    """
    # Use log-space for large values
    if np.any(np.abs(np.asarray(a)) > GAMMA_OVERFLOW_THRESHOLD / 2) or np.any(
        np.abs(np.asarray(b)) > GAMMA_OVERFLOW_THRESHOLD / 2
    ):
        log_ratio = gammaln_safe(a) - gammaln_safe(b)
        return np.exp(log_ratio)

    return gamma_safe(a) / gamma_safe(b)


def factorial_extension(n):
    """
    Factorial extension for non-negative real numbers.
    n! = Γ(n+1)

    Parameters
    ----------
    n : float or array-like
        Non-negative real numbers

    Returns
    -------
    float or array
        n! = Γ(n+1)
    """
    return gamma_safe(np.asarray(n) + 1)


def double_factorial_extension(n):
    """
    Double factorial extension for real numbers.
    n!! = Γ(n/2 + 1) * 2^(n/2) / √π (for n ≥ 0)

    Parameters
    ----------
    n : float or array-like
        Real numbers

    Returns
    -------
    float or array
        n!! extended to real numbers
    """
    n = np.asarray(n)
    if np.any(n < 0):
        raise ValueError("Double factorial extension only defined for n ≥ 0")

    from .constants import SQRT_PI

    return gamma_safe(n / 2 + 1) * (2 ** (n / 2)) / SQRT_PI


# Convenience functions for common gamma expressions
def gamma_half_integer(n):
    """Γ(n + 1/2) for integer n."""
    from .constants import SQRT_PI

    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return SQRT_PI

    # Use the relation Γ(n + 1/2) = (2n-1)!! * √π / 2^n
    from math import factorial

    return (
        factorial(2 * n - 1) * SQRT_PI / (2**n * factorial(n - 1)) if n > 0 else SQRT_PI
    )


def beta_function(a, b):
    """
    Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b).

    Parameters
    ----------
    a, b : float or array-like
        Beta function parameters

    Returns
    -------
    float or array
        B(a,b)
    """
    return gamma_ratio_safe(a, a + b) * gamma_safe(b)


if __name__ == "__main__":
    # Test the functions - library code should never print
    test_values = [0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]

    # Validate gamma function implementation
    for val in test_values:
        result = gamma_safe(val)
        assert np.isfinite(result), f"Gamma function failed for {val}"

    # Validate known mathematical identities
    sqrt_pi = np.sqrt(np.pi)
    assert abs(gamma_safe(0.5) - sqrt_pi) < 1e-10, "Γ(1/2) ≠ √π"
    assert abs(gamma_safe(1.0) - 1.0) < 1e-10, "Γ(1) ≠ 1"
    assert abs(gamma_safe(2.0) - 1.0) < 1e-10, "Γ(2) ≠ 1!"
    assert abs(gamma_safe(3.0) - 2.0) < 1e-10, "Γ(3) ≠ 2!"
