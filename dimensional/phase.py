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
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from core.phase import *  # noqa: F401,F403
from core.phase import (
    PhaseDynamicsEngine,
    sap_rate,
)

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
    for step in range(time_steps):
        try:
            state = engine.get_state()
            results.append(
                {
                    "time": step,
                    "dimensions": len(state),
                    "active_phases": sum(1 for phase in state if phase > 0.1),
                    "total_phase": sum(state),
                    "max_dimension": np.argmax(state),
                }
            )
            engine.step(dt=0.01)
        except Exception:
            break

    return {
        "time_steps": len(results),
        "max_dimensions": max_dimensions,
        "emergence_sequence": results,
        "final_state": engine.get_state(),
        "convergence": engine.diagnostics.get_diagnostics(),
    }


def quick_phase_analysis(dimension=4.0, time_steps=100):
    """
    Perform quick phase analysis around a specific dimension.
    
    Parameters
    ----------
    dimension : float
        Target dimension to analyze
    time_steps : int
        Number of evolution steps
        
    Returns
    -------
    dict
        Analysis results including phase properties and evolution
    """
    from core.measures import phase_capacity
    
    try:
        engine = PhaseDynamicsEngine(max_dimensions=int(dimension) + 2)
        
        # Set initial state at target dimension
        engine.set_phase(int(dimension), 1.0)
        
        # Run brief evolution
        results = []
        for step in range(time_steps):
            state = engine.get_state()
            results.append({
                'time': step,
                'phases': state.copy(),
                'effective_dimension': sum(d * p for d, p in enumerate(state) if p > 0.01)
            })
            engine.step(dt=0.01)
            
        return {
            'target_dimension': dimension,
            'time_steps': time_steps,
            'phase_capacity': phase_capacity(dimension),
            'evolution': results,
            'final_state': results[-1] if results else {'effective_dimension': dimension}
        }
        
    except Exception as e:
        # Fallback analysis if engine fails
        return {
            'target_dimension': dimension,
            'phase_capacity': phase_capacity(dimension) if dimension >= 0 else 0.0,
            'final_state': {'effective_dimension': dimension},
            'error': str(e)
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
        "dimensions": dimensions,
        "capacities": [],
        "thresholds": [],
        "sapping_rates": [],
    }

    for d in dimensions:
        try:
            from core.measures import phase_capacity

            capacity = phase_capacity(d)
            results["capacities"].append(capacity)
            results["thresholds"].append(0.9 * capacity)

            # Sample sapping rate from d to d+1
            rate = sap_rate(d, d + 1, None)
            results["sapping_rates"].append(rate)

        except (ValueError, OverflowError):
            results["capacities"].append(np.nan)
            results["thresholds"].append(np.nan)
            results["sapping_rates"].append(0.0)

    return results


# The PhaseDynamicsEngine and other core functions are imported from core.phase
# This provides a clean consolidated API while maintaining compatibility