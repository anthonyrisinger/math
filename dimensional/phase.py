#!/usr/bin/env python3
"""
Dimensional Phase Dynamics
==========================

Convenience module that imports from core.phase and provides additional
high-level utilities for phase dynamics analysis and visualization.

This module preserves API compatibility while delegating core functionality
to the robust implementations in core.phase.
"""

# Import all core functionality from the authoritative implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.phase import *  # noqa: F401,F403
from core.phase import (
    PhaseDynamicsEngine,
    phase_evolution_step,
    sap_rate,
    emergence_threshold,
    total_phase_energy,
    phase_coherence,
    dimensional_time,
    rk45_step,
    ConvergenceDiagnostics,
    TopologicalInvariants,
)

import numpy as np

# ============================================================================  
# ADDITIONAL CONVENIENCE FUNCTIONS
# ============================================================================


def quick_emergence_analysis(max_dimensions=8, time_steps=500):
    """
    Perform a quick analysis of dimensional emergence patterns.

    Parameters
    ----------
    max_dimensions : int
        Maximum dimensions to simulate
    time_steps : int
        Number of evolution steps

    Returns
    -------
    dict
        Analysis results including emergence times and patterns
    """
    engine = PhaseDynamicsEngine(max_dimensions=max_dimensions)
    
    results = []
    for _ in range(time_steps):
        state = engine.get_state()
        results.append({
            'time': state['time'],
            'emerged': len(state['emerged_dimensions']),
            'energy': state['total_energy'],
            'coherence': state['coherence']
        })
        engine.step(0.01)
    
    return {
        'evolution': results,
        'final_state': engine.get_state(),
        'convergence': engine.diagnostics.get_diagnostics()
    }


def dimensional_explorer(start_dim=0, end_dim=8, resolution=100):
    """
    Explore dimensional space and identify key transition points.

    Parameters
    ----------
    start_dim : float
        Starting dimension
    end_dim : float
        Ending dimension
    resolution : int
        Number of points to sample

    Returns
    -------
    dict
        Exploration results with transition points and properties
    """
    dimensions = np.linspace(start_dim, end_dim, resolution)
    
    results = {
        'dimensions': dimensions,
        'capacities': [],
        'thresholds': [],
        'sapping_rates': []
    }
    
    for d in dimensions:
        try:
            from core.measures import phase_capacity
            capacity = phase_capacity(d)
            results['capacities'].append(capacity)
            results['thresholds'].append(0.9 * capacity)
            
            # Sample sapping rate from d to d+1
            rate = sap_rate(d, d + 1, None)
            results['sapping_rates'].append(rate)
            
        except (ValueError, OverflowError):
            results['capacities'].append(np.nan)
            results['thresholds'].append(np.nan)
            results['sapping_rates'].append(0.0)
    
    return results


# The PhaseDynamicsEngine and other core functions are imported from core.phase
# This provides a clean consolidated API while maintaining compatibility
