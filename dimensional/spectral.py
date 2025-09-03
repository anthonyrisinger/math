#!/usr/bin/env python3
"""Spectral analysis - simplified module."""


import numpy as np


def dimensional_spectral_density(d: float, freq: float = 1.0) -> float:
    """Compute spectral density at dimension d and frequency."""
    return np.exp(-freq * d / 20.0) * (1 + np.sin(2 * np.pi * freq * d))


def spectral_signature(d: float, n_freqs: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Get spectral signature for dimension d."""
    freqs = np.linspace(0, 10, n_freqs)
    densities = np.array([dimensional_spectral_density(d, f) for f in freqs])
    return freqs, densities


def peak_frequency(d: float) -> float:
    """Find peak frequency for dimension d."""
    freqs, densities = spectral_signature(d, 1000)
    return freqs[np.argmax(densities)]
