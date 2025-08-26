"""
Visualization Themes
====================

Standardized themes and styling for mathematical visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np

THEMES = {
    'mathematical': {
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    },
    'dark': {
        'figure.facecolor': '#1a1a1a',
        'axes.facecolor': '#2d2d2d', 
        'axes.edgecolor': 'white',
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.grid': True,
        'grid.color': 'gray',
        'grid.alpha': 0.3
    },
    'golden_ratio': {
        'figure.facecolor': '#fdf6e3',
        'axes.facecolor': '#fdf6e3',
        'axes.edgecolor': '#b58900',
        'text.color': '#586e75',
        'axes.labelcolor': '#586e75',
        'xtick.color': '#657b83',
        'ytick.color': '#657b83'
    }
}

def get_theme(name='mathematical'):
    """Get theme configuration by name."""
    return THEMES.get(name, THEMES['mathematical'])

def apply_theme(name='mathematical'):
    """Apply theme to matplotlib."""
    theme = get_theme(name)
    plt.rcParams.update(theme)