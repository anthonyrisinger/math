#!/usr/bin/env python3
"""
View-Preserving 3D Base Class
==============================

Base class for 3D visualizations that preserves view angles during animation.
Solves the problem of view resetting during updates.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from core_measures import setup_3d_axis, VIEW_ELEV, VIEW_AZIM

class ViewPreserving3D:
    """Base class for 3D visualizations that preserves viewing angles."""

    def __init__(self):
        self.current_elev = VIEW_ELEV
        self.current_azim = VIEW_AZIM
        self.view_changed = False

    def setup_3d_axis(self, ax, title=""):
        """Setup 3D axis with view preservation."""
        ax.set_proj_type('ortho')
        ax.view_init(elev=self.current_elev, azim=self.current_azim)
        ax.set_box_aspect((1, 1, 1))

        if title:
            ax.set_title(title, fontsize=12, pad=15)

        ax.grid(True, alpha=0.3)

        # Connect mouse events for view tracking
        ax.mouse_init()

    def save_view(self, ax):
        """Save current view angles before clearing."""
        if hasattr(ax, 'elev'):
            self.current_elev = ax.elev
        if hasattr(ax, 'azim'):
            self.current_azim = ax.azim

    def restore_view(self, ax):
        """Restore saved view angles after clearing."""
        ax.view_init(elev=self.current_elev, azim=self.current_azim)

    def update_plot_with_preserved_view(self, ax, update_func):
        """Update plot while preserving view angles."""
        # Save current view
        self.save_view(ax)

        # Clear and update
        ax.clear()

        # Setup axis with preserved view
        self.setup_3d_axis(ax)

        # Call the actual update function
        update_func()

        # Ensure view is restored
        self.restore_view(ax)