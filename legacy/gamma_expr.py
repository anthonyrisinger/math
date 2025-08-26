#!/usr/bin/env python3
"""
Edit this file and save to see live updates!
Define a function called plot(fig, d) that creates your visualization.
The parameter d is the current dimension value.
"""

import numpy as np
from scipy.special import gamma

π = np.pi
φ = (1 + np.sqrt(5)) / 2


def plot(fig, d=4.0):
    """Your visualization here - edit and save!"""

    # Example: V×S complexity around dimension d
    fig.clear()
    ax = fig.add_subplot(111, facecolor="#1a1a1a")

    # Define measures
    def v(x):
        return π ** (x / 2) / gamma(x / 2 + 1)
    def s(x):
        return 2 * π ** (x / 2) / gamma(x / 2)
    def c(x):
        return v(x) * s(x)

    # Plot range around current d
    x = np.linspace(max(0.1, d - 3), d + 3, 500)

    # Plot complexity
    y = [c(xi) for xi in x]
    ax.plot(x, y, "lime", lw=3, alpha=0.8)

    # Mark current point
    ax.axvline(d, color="yellow", lw=2, alpha=0.5)
    ax.plot(d, c(d), "yo", markersize=10)
    ax.text(d, c(d), f"  d={d:.2f}, C={c(d):.3f}", color="yellow")

    # Style
    ax.set_title(f"Complexity C = V×S around d={d:.2f}", color="white", fontsize=14)
    ax.set_xlabel("Dimension", color="white")
    ax.set_ylabel("V×S", color="white")
    ax.grid(True, alpha=0.2)

    # Try different things!
    # - Plot gamma directly: y = [gamma(xi) for xi in x]
    # - Show poles: x = np.linspace(-5, 5, 1000)
    # - Complex plane: use meshgrid and contourf
    # - Multiple subplots: use fig.add_subplot(2,2,1) etc
