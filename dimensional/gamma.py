#!/usr/bin/env python3
"""
Dimensional Gamma Functions
===========================

Complete gamma function family with numerical stability, interactive exploration,
and quick analysis tools. Consolidates all gamma functionality into a single,
comprehensive module.

Features:
- Robust numerical implementations with overflow protection
- Quick one-liner exploration tools
- Interactive keyboard-controlled lab
- Live editing and hot-reload capabilities
- Complex plane visualization
- Peak finding and analysis
- Export and visualization utilities
"""

import importlib.util
import os
import time
import traceback

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, gamma, gammaln, polygamma

# ============================================================================
# CONSTANTS
# ============================================================================

PI = π = np.pi
PHI = φ = (1 + np.sqrt(5)) / 2
PSI = ψ = 1 / φ
E = np.e
SQRT_PI = np.sqrt(π)

# Numerical stability constants
NUMERICAL_EPSILON = 1e-15
GAMMA_OVERFLOW_THRESHOLD = 171.0
LOG_SPACE_THRESHOLD = 700.0

# ============================================================================
# CORE ROBUST GAMMA FUNCTIONS
# ============================================================================


def gamma_safe(z):
    """
    Numerically stable gamma function with proper edge case handling.

    Parameters
    ----------
    z : float or array-like
        Input values

    Returns
    -------
    float or array
        Γ(z) with overflow protection and pole handling
    """
    z = np.asarray(z)

    # Handle edge cases
    if np.any(z == 0):
        result = np.full_like(z, np.inf, dtype=float)
        mask = z != 0
        if np.any(mask):
            result[mask] = gamma_safe(z[mask])
        return result if z.ndim > 0 else float(result)

    # Handle negative integers (poles)
    negative_int_mask = (z < 0) & (np.abs(z - np.round(z)) < NUMERICAL_EPSILON)
    if np.any(negative_int_mask):
        result = np.full_like(z, np.inf, dtype=float)
        mask = ~negative_int_mask
        if np.any(mask):
            result[mask] = gamma_safe(z[mask])
        return result if z.ndim > 0 else float(result)

    # Use log-space for large values
    if np.any(np.abs(z) > GAMMA_OVERFLOW_THRESHOLD):
        large_mask = np.abs(z) > GAMMA_OVERFLOW_THRESHOLD
        result = np.zeros_like(z, dtype=float)

        # Small values: direct computation
        if np.any(~large_mask):
            result[~large_mask] = gamma(z[~large_mask])

        # Large values: exp(log(gamma))
        if np.any(large_mask):
            log_gamma_vals = gammaln(z[large_mask])
            exp_mask = log_gamma_vals < LOG_SPACE_THRESHOLD
            if np.any(exp_mask):
                large_indices = np.where(large_mask)[0]
                safe_indices = large_indices[exp_mask]
                result[safe_indices] = np.exp(log_gamma_vals[exp_mask])

            # Extremely large values
            inf_mask = log_gamma_vals >= LOG_SPACE_THRESHOLD
            if np.any(inf_mask):
                large_indices = np.where(large_mask)[0]
                inf_indices = large_indices[inf_mask]
                result[inf_indices] = np.inf

        return result if z.ndim > 0 else float(result)

    # Normal case
    return gamma(z)


def gammaln_safe(z):
    """Safe log-gamma function."""
    z = np.asarray(z)

    # Handle poles
    if np.any(z <= 0):
        pole_mask = np.abs(z - np.round(z)) < NUMERICAL_EPSILON
        if np.any(pole_mask):
            result = np.full_like(z, -np.inf, dtype=float)
            mask = ~pole_mask
            if np.any(mask):
                result[mask] = gammaln_safe(z[mask])
            return result if z.ndim > 0 else float(result)

    return gammaln(z)


def digamma_safe(z):
    """Safe digamma function (psi function)."""
    z = np.asarray(z)

    # Handle poles
    if np.any(z <= 0):
        pole_mask = np.abs(z - np.round(z)) < NUMERICAL_EPSILON
        if np.any(pole_mask):
            result = np.full_like(z, -np.inf, dtype=float)
            mask = ~pole_mask
            if np.any(mask):
                result[mask] = digamma_safe(z[mask])
            return result if z.ndim > 0 else float(result)

    return digamma(z)


def polygamma_safe(n, z):
    """Safe polygamma function."""
    z = np.asarray(z)

    # Handle poles
    if np.any(z <= 0):
        pole_mask = np.abs(z - np.round(z)) < NUMERICAL_EPSILON
        if np.any(pole_mask):
            result = np.full_like(z, (-1) ** (n + 1) * np.inf, dtype=float)
            mask = ~pole_mask
            if np.any(mask):
                result[mask] = polygamma_safe(n, z[mask])
            return result if z.ndim > 0 else float(result)

    return polygamma(n, z)


# ============================================================================
# DIMENSIONAL MEASURES (Quick Tools)
# ============================================================================

# One-liner geometric measures
def v(d):
    return π ** (d / 2) / gamma_safe(d / 2 + 1)  # Volume of d-ball
def s(d):
    return 2 * π ** (d / 2) / gamma_safe(d / 2)  # Surface of (d-1)-sphere
def c(d):
    return v(d) * s(d)  # Complexity V×S
def r(d):
    return s(d) / v(d) if v(d) > 0 else float("inf")  # Ratio S/V
def ρ(d):
    return gamma_safe(d / 2 + 1) / π ** (d / 2)  # Density 1/V

# Quick peak finders
def find_peak(f, a=0, b=15):
    return max([(f(d), d) for d in np.linspace(a, b, 1000)])[1]
def v_peak():
    return find_peak(v, 0.1, 10)  # ≈ 5.26
def s_peak():
    return find_peak(s, 0.1, 10)  # ≈ 7.26
def c_peak():
    return find_peak(c, 0.1, 10)  # ≈ 6.05

# Gamma derivatives
ψ0 = digamma_safe  # ψ(x) = Γ'(x)/Γ(x)
def ψ1(x):
    return polygamma_safe(1, x)  # Trigamma
def ψ2(x):
    return polygamma_safe(2, x)  # Tetragamma
def ψn(n, x):
    return polygamma_safe(n, x)  # n-th derivative

# Complex gamma
def γ(z):
    return gamma_safe(z)  # Works with complex z
def log_γ(z):
    return gammaln_safe(z) if np.isreal(z) else np.log(gamma_safe(z))
def abs_γ(z):
    return np.abs(gamma_safe(z))

# Special formulas
def reflect(z):
    return π / (np.sin(π * z) * gamma_safe(1 - z))  # Reflection formula
def duplicate(z):
    return (np.sqrt(π) * 2 ** (1 - 2 * z) * gamma_safe(2 * z) / gamma_safe(z + 0.5))  # Duplication

# ============================================================================
# QUICK VISUALIZATION TOOLS
# ============================================================================


def qplot(*funcs, x=None, labels=None, title="", figsize=(10, 6)):
    """Quick plot multiple functions with dark theme."""
    if x is None:
        x = np.linspace(0.1, 10, 1000)

    fig = plt.figure(figsize=figsize, facecolor="#0a0a0a")
    ax = plt.gca()
    ax.set_facecolor("#1a1a1a")

    colors = ["cyan", "magenta", "lime", "yellow", "orange", "red", "blue"]
    for i, f in enumerate(funcs):
        try:
            y = [f(xi) for xi in x]
            label = (
                labels[i]
                if labels
                else f.__name__ if hasattr(f, "__name__") else f"f{i}"
            )
            ax.plot(x, y, colors[i % len(colors)], lw=2, label=label, alpha=0.8)
        except Exception as e:
            print(f"Error plotting function {i}: {e}")

    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    ax.set_title(title, color="white", fontsize=14)
    plt.show()
    return fig, ax


def qsurf(f, x_range=(-3, 3), y_range=(-3, 3), res=100):
    """Quick surface plot of complex function."""
    fig = plt.figure(figsize=(10, 8), facecolor="#0a0a0a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a1a")

    x = np.linspace(*x_range, res)
    y = np.linspace(*y_range, res)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    W = np.zeros_like(Z, dtype=float)
    for i in range(res):
        for j in range(res):
            try:
                W[i, j] = np.abs(f(Z[i, j]))
            except:
                W[i, j] = np.nan

    surf = ax.plot_surface(X, Y, W, cmap="plasma", alpha=0.8)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_zlabel("|f(z)|")
    plt.colorbar(surf)
    plt.show()
    return fig, ax


def explore(d=4):
    """Quick exploration at dimension d."""
    print(f"\n🌟 DIMENSION d = {d}")
    print(f"  Volume V_d     = {v(d):.6f}")
    print(f"  Surface S_d    = {s(d):.6f}")
    print(f"  Complexity C_d = {c(d):.6f}")
    print(f"  Ratio S/V      = {r(d):.6f}")
    print(f"  Density ρ_d    = {ρ(d):.6f}")
    print(f"  Gamma(d/2)     = {gamma_safe(d/2):.6f}")
    print(f"  Digamma(d/2)   = {digamma_safe(d/2):.6f}")


def peaks():
    """Find all peaks."""
    print("\n🎯 CRITICAL PEAKS:")
    print(f"  Volume peak:     d = {v_peak():.6f}")
    print(f"  Surface peak:    d = {s_peak():.6f}")
    print(f"  Complexity peak: d = {c_peak():.6f}")
    print(f"  π boundary:      d = {π:.6f}")
    print(f"  2π boundary:     d = {2*π:.6f}")
    print(f"  φ golden:        d = {φ:.6f}")


def instant():
    """Instant 4-panel visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="#0a0a0a")
    fig.suptitle("GAMMA GEOMETRIC MEASURES", color="white", fontsize=16)

    d = np.linspace(0.1, 15, 1000)

    # Volume
    ax = axes[0, 0]
    ax.set_facecolor("#1a1a1a")
    ax.plot(d, [v(x) for x in d], "cyan", lw=2)
    ax.set_title("Volume V_d", color="cyan")
    ax.grid(True, alpha=0.2)

    # Surface
    ax = axes[0, 1]
    ax.set_facecolor("#1a1a1a")
    ax.plot(d, [s(x) for x in d], "magenta", lw=2)
    ax.set_title("Surface S_d", color="magenta")
    ax.grid(True, alpha=0.2)

    # Complexity
    ax = axes[1, 0]
    ax.set_facecolor("#1a1a1a")
    ax.plot(d, [c(x) for x in d], "lime", lw=2)
    ax.set_title("Complexity V×S", color="lime")
    ax.grid(True, alpha=0.2)

    # Gamma function
    ax = axes[1, 1]
    ax.set_facecolor("#1a1a1a")
    x = np.linspace(0.1, 8, 1000)
    ax.plot(x, gamma_safe(x), "yellow", lw=2)
    ax.set_title("Γ(x)", color="yellow")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 10)

    plt.tight_layout()
    plt.show()
    return fig, axes


# ============================================================================
# INTERACTIVE GAMMA LAB
# ============================================================================


class GammaLab:
    """Interactive keyboard-controlled gamma function laboratory."""

    def __init__(self, start_d=4.0):
        self.d = start_d
        self.mode = 0
        self.modes = ["measures", "gamma", "complex", "derivatives", "poles"]
        self.show_complex = False

        # Setup figure
        self.fig = plt.figure(figsize=(14, 10), facecolor="#0a0a0a")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.update()

    def on_key(self, event):
        """Handle keyboard input."""
        if event.key == "up":
            self.d += 0.1
        elif event.key == "down":
            self.d -= 0.1
        elif event.key == "right":
            self.d += 1.0
        elif event.key == "left":
            self.d -= 1.0
        elif event.key == " ":
            self.mode = (self.mode + 1) % len(self.modes)
        elif event.key == "r":
            self.d = 4.0
        elif event.key == "c":
            self.show_complex = not self.show_complex
        elif event.key == "p":
            self.find_peaks()
        elif event.key in ["q", "escape"]:
            plt.close("all")
            return

        self.update()

    def update(self):
        """Refresh all plots."""
        self.fig.clear()

        # Title
        self.fig.text(
            0.5,
            0.96,
            f"GAMMA LAB | d = {self.d:.3f} | Mode: {self.modes[self.mode].upper()}",
            ha="center",
            fontsize=16,
            color="white",
            weight="bold",
        )

        if self.modes[self.mode] == "measures":
            self.plot_measures()
        elif self.modes[self.mode] == "gamma":
            self.plot_gamma()
        elif self.modes[self.mode] == "complex":
            self.plot_complex_gamma()
        elif self.modes[self.mode] == "derivatives":
            self.plot_derivatives()
        elif self.modes[self.mode] == "poles":
            self.plot_poles()

        plt.draw()

    def plot_measures(self):
        """Plot V, S, and C=V×S."""
        gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)

        d_range = np.linspace(-1, 15, 1000)

        # Volume
        ax1 = self.fig.add_subplot(gs[0, 0], facecolor="#1a1a1a")
        v_vals = [v(d) for d in d_range]
        ax1.plot(d_range, v_vals, "cyan", lw=2, alpha=0.8)
        ax1.axvline(self.d, color="yellow", lw=2, alpha=0.5)
        ax1.axhline(v(self.d), color="yellow", lw=1, alpha=0.3)
        ax1.set_title(
            f"Volume V_d = π^(d/2)/Γ(d/2+1) | V_{self.d:.1f} = {v(self.d):.4f}",
            color="cyan",
        )
        ax1.grid(True, alpha=0.2)
        ax1.set_xlim(-1, 15)

        # Surface
        ax2 = self.fig.add_subplot(gs[0, 1], facecolor="#1a1a1a")
        s_vals = [s(d) for d in d_range]
        ax2.plot(d_range, s_vals, "magenta", lw=2, alpha=0.8)
        ax2.axvline(self.d, color="yellow", lw=2, alpha=0.5)
        ax2.axhline(s(self.d), color="yellow", lw=1, alpha=0.3)
        ax2.set_title(
            f"Surface S_d = 2π^(d/2)/Γ(d/2) | S_{self.d:.1f} = {s(self.d):.4f}",
            color="magenta",
        )
        ax2.grid(True, alpha=0.2)
        ax2.set_xlim(-1, 15)

        # Complexity
        ax3 = self.fig.add_subplot(gs[1, :], facecolor="#1a1a1a")
        c_vals = [c(d) for d in d_range]
        ax3.plot(d_range, c_vals, "lime", lw=3, alpha=0.8)
        ax3.axvline(self.d, color="yellow", lw=2, alpha=0.5)
        ax3.axhline(c(self.d), color="yellow", lw=1, alpha=0.3)

        # Mark the peak
        peak_idx = np.argmax(c_vals)
        peak_d = d_range[peak_idx]
        ax3.plot(peak_d, c_vals[peak_idx], "ro", markersize=10)
        ax3.text(
            peak_d,
            c_vals[peak_idx],
            f"  Peak: d={peak_d:.3f}",
            color="red",
            fontsize=10,
        )

        ax3.set_title(
            f"Complexity C_d = V_d × S_d | C_{self.d:.1f} = {c(self.d):.4f}",
            color="lime",
        )
        ax3.set_xlabel("Dimension d", color="white")
        ax3.grid(True, alpha=0.2)
        ax3.set_xlim(-1, 15)

        # Mark critical dimensions
        for ax in [ax1, ax2, ax3]:
            ax.axvline(π, color="orange", lw=1, alpha=0.3, linestyle="--")
            ax.axvline(2 * π, color="orange", lw=1, alpha=0.3, linestyle="--")
            ax.axvline(φ, color="gold", lw=1, alpha=0.3, linestyle=":")

    def plot_gamma(self):
        """Direct gamma function visualization."""
        gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)

        # Real gamma
        ax1 = self.fig.add_subplot(gs[0, :], facecolor="#1a1a1a")
        x = np.linspace(-5, 5, 1000)
        y = []
        for xi in x:
            try:
                val = gamma_safe(xi)
                if abs(val) > 100:
                    y.append(np.nan)
                else:
                    y.append(val)
            except:
                y.append(np.nan)

        ax1.plot(x, y, "cyan", lw=2)
        ax1.axvline(self.d / 2, color="yellow", lw=2, alpha=0.5)
        ax1.axhline(0, color="white", lw=0.5, alpha=0.3)
        ax1.axvline(0, color="white", lw=0.5, alpha=0.3)
        ax1.set_title(
            f"Γ(x) | Γ({self.d/2:.2f}) = {gamma_safe(self.d/2):.4f}", color="cyan"
        )
        ax1.set_ylim(-10, 10)
        ax1.grid(True, alpha=0.2)

        # Log gamma
        ax2 = self.fig.add_subplot(gs[1, 0], facecolor="#1a1a1a")
        x_pos = np.linspace(0.01, 10, 1000)
        ax2.plot(x_pos, gammaln_safe(x_pos), "magenta", lw=2)
        ax2.axvline(self.d / 2, color="yellow", lw=2, alpha=0.5)
        ax2.set_title("log Γ(x)", color="magenta")
        ax2.grid(True, alpha=0.2)

        # Digamma (derivative of log gamma)
        ax3 = self.fig.add_subplot(gs[1, 1], facecolor="#1a1a1a")
        ax3.plot(x_pos, digamma_safe(x_pos), "lime", lw=2)
        ax3.axvline(self.d / 2, color="yellow", lw=2, alpha=0.5)
        ax3.set_title("ψ(x) = d/dx log Γ(x)", color="lime")
        ax3.grid(True, alpha=0.2)

    def plot_complex_gamma(self):
        """Complex plane gamma visualization."""
        ax = self.fig.add_subplot(111, facecolor="#1a1a1a")

        # Create complex grid
        re = np.linspace(-4, 4, 200)
        im = np.linspace(-4, 4, 200)
        Re, Im = np.meshgrid(re, im)
        Z = Re + 1j * Im

        # Compute |Γ(z)|
        Gamma_abs = np.abs(gamma_safe(Z))
        Gamma_abs = np.clip(Gamma_abs, 0, 10)

        # Plot
        c = ax.contourf(Re, Im, Gamma_abs, levels=50, cmap="plasma")
        plt.colorbar(c, ax=ax, label="|Γ(z)|")

        # Mark current point
        ax.plot(self.d / 2, 0, "yo", markersize=10)
        ax.text(self.d / 2, 0.5, f"d/2={self.d/2:.2f}", color="yellow")

        # Mark poles
        for n in range(-3, 1):
            ax.plot(n, 0, "rx", markersize=8)

        ax.set_title("Complex Gamma |Γ(z)|", color="white", fontsize=14)
        ax.set_xlabel("Re(z)", color="white")
        ax.set_ylabel("Im(z)", color="white")
        ax.grid(True, alpha=0.2)

    def plot_derivatives(self):
        """Gamma derivatives - polygamma functions."""
        gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)

        x = np.linspace(0.1, 10, 1000)

        for n, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            ax = self.fig.add_subplot(gs[row, col], facecolor="#1a1a1a")

            if n == 0:
                y = digamma_safe(x)
                title = "ψ⁰(x) = Γ'(x)/Γ(x)"
                color = "cyan"
            else:
                y = polygamma_safe(n - 1, x)
                title = f"ψ^{n}(x)"
                color = ["magenta", "lime", "orange"][n - 1]

            ax.plot(x, y, color=color, lw=2)
            ax.axvline(self.d / 2, color="yellow", lw=2, alpha=0.5)
            ax.set_title(title, color=color)
            ax.grid(True, alpha=0.2)
            ax.set_xlim(0.1, 10)

            if n == 0:
                ax.set_ylim(-5, 5)

    def plot_poles(self):
        """Gamma function poles and residues."""
        ax = self.fig.add_subplot(111, projection="3d", facecolor="#1a1a1a")

        # Create surface around poles
        x = np.linspace(-4, 2, 400)
        y = np.linspace(-2, 2, 400)
        X, Y = np.meshgrid(x, y)
        Z_complex = X + 1j * Y

        # Compute gamma (handle poles)
        Gamma_vals = np.zeros_like(Z_complex)
        for i in range(Z_complex.shape[0]):
            for j in range(Z_complex.shape[1]):
                try:
                    val = gamma_safe(Z_complex[i, j])
                    Gamma_vals[i, j] = np.log(np.abs(val) + 1)
                except:
                    Gamma_vals[i, j] = 10

        # Cap values for visualization
        Gamma_vals = np.clip(Gamma_vals, -5, 5)

        # Plot surface
        ax.plot_surface(X, Y, Gamma_vals, cmap="plasma", alpha=0.8)

        # Mark poles
        for n in range(-3, 1):
            ax.scatter([n], [0], [5], color="red", s=50, marker="x")

        ax.set_title("log|Γ(z)| showing poles at negative integers", color="white")
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        ax.set_zlabel("log|Γ(z)|")
        ax.view_init(elev=30, azim=45)

    def find_peaks(self):
        """Find and display peaks."""
        d_range = np.linspace(0.1, 15, 5000)

        v_vals = [v(d) for d in d_range]
        s_vals = [s(d) for d in d_range]
        c_vals = [c(d) for d in d_range]

        v_peak_d = d_range[np.argmax(v_vals)]
        s_peak_d = d_range[np.argmax(s_vals)]
        c_peak_d = d_range[np.argmax(c_vals)]

        print("\n🎯 PEAKS FOUND:")
        print(f"  Volume peak:     d = {v_peak_d:.6f}")
        print(f"  Surface peak:    d = {s_peak_d:.6f}")
        print(f"  Complexity peak: d = {c_peak_d:.6f}")
        print(f"  Current d:       d = {self.d:.6f}")

    def show(self):
        """Display the lab."""
        plt.show()


# ============================================================================
# LIVE EDITING SYSTEM
# ============================================================================

EXPR_TEMPLATE = '''#!/usr/bin/env python3
"""
Live Gamma Editor - Edit and save to see updates!
Define a function called plot(fig, d) that creates your visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from dimensional.gamma import *

def plot(fig, d=4.0):
    """Your visualization here - edit and save!"""

    fig.clear()
    ax = fig.add_subplot(111, facecolor='#1a1a1a')

    # Example: Complexity around dimension d
    x = np.linspace(max(0.1, d-3), d+3, 500)
    y = [c(xi) for xi in x]

    ax.plot(x, y, 'lime', lw=3, alpha=0.8)
    ax.axvline(d, color='yellow', lw=2, alpha=0.5)
    ax.plot(d, c(d), 'yo', markersize=10)
    ax.text(d, c(d), f'  d={d:.2f}, C={c(d):.3f}', color='yellow')

    ax.set_title(f'Complexity C = V×S around d={d:.2f}', color='white', fontsize=14)
    ax.set_xlabel('Dimension', color='white')
    ax.set_ylabel('V×S', color='white')
    ax.grid(True, alpha=0.2)
'''


class LiveGamma:
    """Live editing system for gamma exploration."""

    def __init__(self, expr_file="gamma_expr.py"):
        self.expr_file = expr_file
        self.d = 4.0
        self.last_mtime = 0
        self.module = None

        # Create expression file if it doesn't exist
        if not os.path.exists(self.expr_file):
            with open(self.expr_file, "w") as f:
                f.write(EXPR_TEMPLATE)
            print(
                f"✨ Created {self.expr_file} - edit it and save to see live updates!"
            )

        # Setup figure
        self.fig = plt.figure(figsize=(12, 8), facecolor="#0a0a0a")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Initial load
        self.reload_module()

    def on_key(self, event):
        """Handle keyboard input."""
        if event.key == "up":
            self.d += 0.1
            self.update_plot()
        elif event.key == "down":
            self.d -= 0.1
            self.update_plot()
        elif event.key == "right":
            self.d += 1.0
            self.update_plot()
        elif event.key == "left":
            self.d -= 1.0
            self.update_plot()
        elif event.key == "r":
            self.d = 4.0
            self.update_plot()
        elif event.key == "space":
            self.reload_module()
        elif event.key in ["q", "escape"]:
            plt.close("all")
            return

    def reload_module(self):
        """Reload the expression module."""
        try:
            spec = importlib.util.spec_from_file_location("gamma_expr", self.expr_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, "plot"):
                self.show_error("No plot() function found in expression file")
                return

            self.module = module
            self.update_plot()

        except Exception:
            self.show_error(f"Error loading module:\n{traceback.format_exc()}")

    def update_plot(self):
        """Update the plot with current module."""
        if self.module is None:
            return

        try:
            self.fig.clear()
            self.fig.text(
                0.5,
                0.98,
                f"LIVE GAMMA | d = {self.d:.3f} | ↑↓←→: change d | SPACE: reload | Q: quit",
                ha="center",
                fontsize=12,
                color="white",
                weight="bold",
            )

            self.module.plot(self.fig, self.d)
            self.fig.canvas.draw_idle()

        except Exception:
            self.show_error(f"Error in plot():\n{traceback.format_exc()}")

    def show_error(self, msg):
        """Display error message."""
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor="#1a1a1a")
        ax.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            color="red",
            fontsize=10,
            family="monospace",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.fig.canvas.draw_idle()

    def watch(self):
        """Watch for file changes and auto-reload."""
        print("\n🔥 LIVE MODE ACTIVE")
        print(f"   Watching: {self.expr_file}")
        print("   Edit the file and save to see changes!")
        print("\n   Controls: ↑↓←→ (change d), SPACE (reload), Q (quit)")

        while plt.fignum_exists(self.fig.number):
            try:
                current_mtime = os.path.getmtime(self.expr_file)
                if current_mtime > self.last_mtime:
                    self.last_mtime = current_mtime
                    print("📝 Detected change, reloading...")
                    self.reload_module()

                plt.pause(0.1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Watch error: {e}")
                time.sleep(1)

    def show(self):
        """Start live editing mode."""
        plt.show()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def lab(start_d=4.0):
    """Start interactive gamma lab."""
    lab = GammaLab(start_d)
    lab.show()


def live(expr_file="gamma_expr.py"):
    """Start live editing mode."""
    live = LiveGamma(expr_file)
    live.watch()


def demo():
    """Run a comprehensive demo."""
    print("🚀 DIMENSIONAL GAMMA DEMO")
    print("=" * 50)

    # Quick exploration
    explore(4)
    peaks()

    # Instant visualization
    instant()

    print("\n🎮 Try these interactive modes:")
    print("  lab()     # Interactive keyboard lab")
    print("  live()    # Live editing mode")
    print("  qplot(v, s, c)  # Quick multi-plot")


# Export all functions for easy access
__all__ = [
    # Core functions
    "gamma_safe",
    "gammaln_safe",
    "digamma_safe",
    "polygamma_safe",
    # Quick tools
    "v",
    "s",
    "c",
    "r",
    "ρ",
    "find_peak",
    "v_peak",
    "s_peak",
    "c_peak",
    "ψ0",
    "ψ1",
    "ψ2",
    "ψn",
    "γ",
    "log_γ",
    "abs_γ",
    "reflect",
    "duplicate",
    # Visualization
    "qplot",
    "qsurf",
    "explore",
    "peaks",
    "instant",
    # Interactive
    "GammaLab",
    "LiveGamma",
    "lab",
    "live",
    "demo",
    # Constants
    "PI",
    "PHI",
    "PSI",
    "E",
    "SQRT_PI",
    "π",
    "φ",
    "ψ",
]

if __name__ == "__main__":
    demo()
