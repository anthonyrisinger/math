"""
Phase Dynamics Module
=====================

Refactored from monolithic phase.py (1532 lines) into focused submodules.
"""

from .analysis import (
    advanced_emergence_detection,
    emergence_threshold,
    quick_phase_analysis,
)
from .core import (
    dimensional_time,
    phase_coherence,
    phase_evolution_step,
    sap_rate,
    total_phase_energy,
)
from .dynamics import PhaseDynamicsEngine

__all__ = [
    # Engine
    'PhaseDynamicsEngine',
    # Core functions
    'phase_evolution_step',
    'sap_rate',
    'total_phase_energy',
    'phase_coherence',
    'dimensional_time',
    # Analysis functions
    'emergence_threshold',
    'advanced_emergence_detection',
    'quick_phase_analysis',
]
