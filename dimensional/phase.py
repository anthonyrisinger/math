#!/usr/bin/env python3
"""Phase dynamics - simplified module."""

from typing import Any

import numpy as np


def phase_evolution(d: float, t: float = 1.0) -> float:
    """Compute phase evolution at dimension d and time t."""
    return np.exp(-d * t / 10.0) * np.cos(2 * np.pi * d / 5.26)


def critical_points() -> list[float]:
    """Return critical dimension points."""
    return [5.26, 10.52, 15.78, 21.04]


def analyze(d: float) -> dict[str, Any]:
    """Analyze phase at dimension d."""
    return {
        "dimension": d,
        "phase": phase_evolution(d),
        "is_critical": any(abs(d - cp) < 0.1 for cp in critical_points()),
    }
