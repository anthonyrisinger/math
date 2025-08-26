#!/usr/bin/env python3
"""
GAMMA QUICK - Ultra-compact exploration tools
==============================================
Copy-paste these into any Python session for instant exploration.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, gamma, gammaln, polygamma

# Core constants
π = np.pi
φ = (1 + np.sqrt(5)) / 2
ψ = 1 / φ
e = np.e

# One-liner geometric measures
def v(d):
    return π ** (d / 2) / gamma(d / 2 + 1)  # Volume of d-ball
def s(d):
    return 2 * π ** (d / 2) / gamma(d / 2)  # Surface of (d-1)-sphere
def c(d):
    return v(d) * s(d)  # Complexity V×S
def r(d):
    return s(d) / v(d) if v(d) > 0 else float("inf")  # Ratio S/V
def ρ(d):
    return gamma(d / 2 + 1) / π ** (d / 2)  # Density 1/V

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
ψ0 = digamma  # ψ(x) = Γ'(x)/Γ(x)
def ψ1(x):
    return polygamma(1, x)  # Trigamma
def ψ2(x):
    return polygamma(2, x)  # Tetragamma
def ψn(n, x):
    return polygamma(n, x)  # n-th derivative

# Complex gamma
def γ(z):
    return gamma(z)  # Works with complex z
def log_γ(z):
    return gammaln(z) if np.isreal(z) else np.log(gamma(z))
def abs_γ(z):
    return np.abs(gamma(z))

# Reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
def reflect(z):
    return π / (np.sin(π * z) * gamma(1 - z))

# Duplication formula: Γ(z)Γ(z+1/2) = √π * 2^(1-2z) * Γ(2z)
def duplicate(z):
    return np.sqrt(π) * 2 ** (1 - 2 * z) * gamma(2 * z) / gamma(z + 0.5)


# Quick plots
def qplot(*funcs, x=None, labels=None, title=""):
    """Quick plot multiple functions."""
    if x is None:
        x = np.linspace(0.1, 10, 1000)
    plt.figure(figsize=(10, 6), facecolor="#0a0a0a")
    ax = plt.gca()
    ax.set_facecolor("#1a1a1a")

    colors = ["cyan", "magenta", "lime", "yellow", "orange"]
    for i, f in enumerate(funcs):
        y = [f(xi) for xi in x]
        label = (
            labels[i] if labels else f.__name__ if hasattr(f, "__name__") else f"f{i}"
        )
        ax.plot(x, y, colors[i % len(colors)], lw=2, label=label, alpha=0.8)

    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    ax.set_title(title, color="white", fontsize=14)
    plt.show()


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


def explore(d=4):
    """Quick exploration at dimension d."""
    print(f"\n🌟 DIMENSION d = {d}")
    print(f"  Volume V_d     = {v(d):.6f}")
    print(f"  Surface S_d    = {s(d):.6f}")
    print(f"  Complexity C_d = {c(d):.6f}")
    print(f"  Ratio S/V      = {r(d):.6f}")
    print(f"  Density ρ_d    = {ρ(d):.6f}")
    print(f"  Gamma(d/2)     = {gamma(d/2):.6f}")
    print(f"  Digamma(d/2)   = {digamma(d/2):.6f}")


def peaks():
    """Find all peaks."""
    print("\n🎯 CRITICAL PEAKS:")
    print(f"  Volume peak:     d = {v_peak():.6f}")
    print(f"  Surface peak:    d = {s_peak():.6f}")
    print(f"  Complexity peak: d = {c_peak():.6f}")
    print(f"  π boundary:      d = {π:.6f}")
    print(f"  2π boundary:     d = {2*π:.6f}")
    print(f"  φ golden:        d = {φ:.6f}")


# Instant visualizations
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
    ax.plot(x, gamma(x), "yellow", lw=2)
    ax.set_title("Γ(x)", color="yellow")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 10)

    plt.tight_layout()
    plt.show()


# Usage examples
if __name__ == "__main__":
    print(__doc__)
    print("\n💫 Quick examples:")
    print("  explore(4)        # Explore dimension 4")
    print("  peaks()           # Find all peaks")
    print("  instant()         # Instant visualization")
    print("  qplot(v, s, c)    # Quick plot V, S, C")
    print("  qsurf(gamma)      # Surface plot of gamma")
    print("\n🚀 One-liners:")
    print(f"  v(4) = {v(4):.4f}    # Volume at d=4")
    print(f"  c_peak() = {c_peak():.4f}  # Find complexity peak")

    # Run instant visualization
    instant()
