#!/usr/bin/env python3
"""
GAMMA QUICK - Ultra-compact exploration tools
==============================================
Copy-paste these into any Python session for instant exploration.
"""

import numpy as np
from scipy.special import gamma, gammaln, digamma, polygamma
import matplotlib.pyplot as plt

# Core constants
π = np.pi
φ = (1 + np.sqrt(5))/2
ψ = 1/φ
e = np.e

# One-liner geometric measures
v = lambda d: π**(d/2) / gamma(d/2 + 1)                    # Volume of d-ball
s = lambda d: 2*π**(d/2) / gamma(d/2)                      # Surface of (d-1)-sphere
c = lambda d: v(d) * s(d)                                  # Complexity V×S
r = lambda d: s(d) / v(d) if v(d) > 0 else float('inf')   # Ratio S/V
ρ = lambda d: gamma(d/2 + 1) / π**(d/2)                   # Density 1/V

# Quick peak finders
find_peak = lambda f, a=0, b=15: max([(f(d), d) for d in np.linspace(a, b, 1000)])[1]
v_peak = lambda: find_peak(v, 0.1, 10)  # ≈ 5.26
s_peak = lambda: find_peak(s, 0.1, 10)  # ≈ 7.26
c_peak = lambda: find_peak(c, 0.1, 10)  # ≈ 6.05

# Gamma derivatives
ψ0 = digamma                            # ψ(x) = Γ'(x)/Γ(x)
ψ1 = lambda x: polygamma(1, x)          # Trigamma
ψ2 = lambda x: polygamma(2, x)          # Tetragamma
ψn = lambda n, x: polygamma(n, x)       # n-th derivative

# Complex gamma
γ = lambda z: gamma(z)                                     # Works with complex z
log_γ = lambda z: gammaln(z) if np.isreal(z) else np.log(gamma(z))
abs_γ = lambda z: np.abs(gamma(z))

# Reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
reflect = lambda z: π / (np.sin(π*z) * gamma(1-z))

# Duplication formula: Γ(z)Γ(z+1/2) = √π * 2^(1-2z) * Γ(2z)
duplicate = lambda z: np.sqrt(π) * 2**(1-2*z) * gamma(2*z) / gamma(z+0.5)

# Quick plots
def qplot(*funcs, x=None, labels=None, title=""):
    """Quick plot multiple functions."""
    if x is None:
        x = np.linspace(0.1, 10, 1000)
    plt.figure(figsize=(10, 6), facecolor='#0a0a0a')
    ax = plt.gca()
    ax.set_facecolor('#1a1a1a')

    colors = ['cyan', 'magenta', 'lime', 'yellow', 'orange']
    for i, f in enumerate(funcs):
        y = [f(xi) for xi in x]
        label = labels[i] if labels else f.__name__ if hasattr(f, '__name__') else f"f{i}"
        ax.plot(x, y, colors[i % len(colors)], lw=2, label=label, alpha=0.8)

    ax.grid(True, alpha=0.2)
    ax.legend(loc='best')
    ax.set_title(title, color='white', fontsize=14)
    plt.show()

def qsurf(f, x_range=(-3, 3), y_range=(-3, 3), res=100):
    """Quick surface plot of complex function."""
    fig = plt.figure(figsize=(10, 8), facecolor='#0a0a0a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a1a')

    x = np.linspace(*x_range, res)
    y = np.linspace(*y_range, res)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y

    W = np.zeros_like(Z, dtype=float)
    for i in range(res):
        for j in range(res):
            try:
                W[i,j] = np.abs(f(Z[i,j]))
            except:
                W[i,j] = np.nan

    surf = ax.plot_surface(X, Y, W, cmap='plasma', alpha=0.8)
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_zlabel('|f(z)|')
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='#0a0a0a')
    fig.suptitle('GAMMA GEOMETRIC MEASURES', color='white', fontsize=16)

    d = np.linspace(0.1, 15, 1000)

    # Volume
    ax = axes[0,0]
    ax.set_facecolor('#1a1a1a')
    ax.plot(d, [v(x) for x in d], 'cyan', lw=2)
    ax.set_title('Volume V_d', color='cyan')
    ax.grid(True, alpha=0.2)

    # Surface
    ax = axes[0,1]
    ax.set_facecolor('#1a1a1a')
    ax.plot(d, [s(x) for x in d], 'magenta', lw=2)
    ax.set_title('Surface S_d', color='magenta')
    ax.grid(True, alpha=0.2)

    # Complexity
    ax = axes[1,0]
    ax.set_facecolor('#1a1a1a')
    ax.plot(d, [c(x) for x in d], 'lime', lw=2)
    ax.set_title('Complexity V×S', color='lime')
    ax.grid(True, alpha=0.2)

    # Gamma function
    ax = axes[1,1]
    ax.set_facecolor('#1a1a1a')
    x = np.linspace(0.1, 8, 1000)
    ax.plot(x, gamma(x), 'yellow', lw=2)
    ax.set_title('Γ(x)', color='yellow')
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