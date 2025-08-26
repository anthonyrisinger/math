#!/usr/bin/env python3
"""
Modern Visualization Backends
============================

PHASE 2: AGGRESSIVE MODERNIZATION
- Kingdon geometric algebra visualization
- Plotly interactive dashboards
- Matplotlib replacement with extreme prejudice
"""

from .base_backend import VisualizationBackend
from .kingdon_backend import KingdonRenderer
from .plotly_backend import PlotlyDashboard

__all__ = ["KingdonRenderer", "PlotlyDashboard", "VisualizationBackend"]
