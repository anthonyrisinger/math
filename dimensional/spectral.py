"""
Spectral analysis compatibility module.
"""

import numpy as np
from scipy import signal

from .core import c, r, s, v


def dimensional_spectral_density(d_range, measure='volume'):
    """Compute spectral density of dimensional measure."""
    d_range = np.asarray(d_range)

    # Get measure values
    if measure == 'volume':
        values = v(d_range)
    elif measure == 'surface':
        values = s(d_range)
    elif measure == 'complexity':
        values = c(d_range)
    elif measure == 'ratio':
        values = r(d_range)
    else:
        values = v(d_range)

    # Compute power spectral density
    freqs, psd = signal.periodogram(values)

    return {
        'frequencies': freqs.tolist(),
        'power_spectral_density': psd.tolist(),
        'dimensions': d_range.tolist(),
        'measure': measure,
    }


def spectral_peak(d_range, measure='volume'):
    """Find spectral peak."""
    result = dimensional_spectral_density(d_range, measure)
    psd = np.array(result['power_spectral_density'])
    freqs = np.array(result['frequencies'])

    peak_idx = np.argmax(psd)

    return {
        'peak_frequency': float(freqs[peak_idx]),
        'peak_power': float(psd[peak_idx]),
        'measure': measure,
    }


def spectral_bandwidth(d_range, measure='volume', level_db=-3):
    """Compute spectral bandwidth."""
    result = dimensional_spectral_density(d_range, measure)
    psd = np.array(result['power_spectral_density'])
    freqs = np.array(result['frequencies'])

    # Find peak
    peak_idx = np.argmax(psd)
    peak_power = psd[peak_idx]

    # Find bandwidth at level_db below peak
    threshold = peak_power * (10 ** (level_db / 10))
    above_threshold = psd >= threshold

    if np.any(above_threshold):
        indices = np.where(above_threshold)[0]
        bandwidth = float(freqs[indices[-1]] - freqs[indices[0]])
    else:
        bandwidth = 0.0

    return {
        'bandwidth': bandwidth,
        'center_frequency': float(freqs[peak_idx]),
        'level_db': level_db,
        'measure': measure,
    }


def spectral_coherence(d_range, measure1='volume', measure2='surface'):
    """Compute spectral coherence between two measures."""
    d_range = np.asarray(d_range)

    # Get measure values
    measures = {
        'volume': v,
        'surface': s,
        'complexity': c,
        'ratio': r,
    }

    values1 = measures.get(measure1, v)(d_range)
    values2 = measures.get(measure2, s)(d_range)

    # Compute coherence
    freqs, coherence = signal.coherence(values1, values2)

    return {
        'frequencies': freqs.tolist(),
        'coherence': coherence.tolist(),
        'measure1': measure1,
        'measure2': measure2,
        'mean_coherence': float(np.mean(coherence)),
    }


def spectral_entropy(d_range, measure='volume'):
    """Compute spectral entropy."""
    result = dimensional_spectral_density(d_range, measure)
    psd = np.array(result['power_spectral_density'])

    # Normalize to probability distribution
    psd_norm = psd / np.sum(psd)

    # Shannon entropy
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

    return {
        'spectral_entropy': float(entropy),
        'measure': measure,
        'n_frequencies': len(psd),
    }


# Export
__all__ = [
    'dimensional_spectral_density', 'spectral_peak',
    'spectral_bandwidth', 'spectral_coherence', 'spectral_entropy',
]
