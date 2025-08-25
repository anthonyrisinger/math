#!/usr/bin/env python3
"""
Dimensional Analysis Package
============================

Unified package for dimensional emergence theory, geometric measures,
and reality framework analysis. Consolidates and extends the dimensional
analysis scattered across dim0-dim5 modules.

Core concepts:
- Dimension as fundamental parameter
- Time emergence from dimensional change  
- Phase coherence and emergence
- Geometric measure optimization
- Reality framework modeling

Quick start:
    from analysis import *
    analyzer = DimensionalAnalyzer()
    analyzer.explore_dimension(4)
"""

# Import core functionality
from .geometric_measures import *
from .emergence_framework import *
from .reality_modeling import *

# Import test suite
from .test_analysis import run_all_tests

# Package metadata
__version__ = "1.0.0"
__author__ = "Dimensional Analysis Project"
__description__ = "Unified dimensional analysis and emergence theory tools"

# Convenience function
def quick_analysis(dimension):
    """Quick dimensional analysis of a given dimension."""
    from .geometric_measures import DimensionalAnalyzer
    analyzer = DimensionalAnalyzer()
    return analyzer.analyze_dimension(dimension)

def explore_emergence():
    """Explore dimensional emergence interactively."""
    from .emergence_framework import EmergenceFramework
    framework = EmergenceFramework()
    framework.interactive_exploration()

def analyze_reality():
    """Analyze reality through dimensional lens."""
    from .reality_modeling import RealityModeler
    reality = RealityModeler()
    return reality.complete_analysis()

def test_package():
    """Run complete test suite for the package."""
    return run_all_tests()
