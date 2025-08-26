#!/usr/bin/env python3
"""
GAMMA LAB - Ultra-fast interactive exploration
===============================================
Keyboard controls:
  ↑/↓     : Change dimension by ±0.1
  ←/→     : Change dimension by ±1.0
  SPACE   : Cycle through plot modes
  R       : Reset to d=4
  C       : Toggle complex plane
  P       : Find peaks
  Q/ESC   : Quit
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, gamma, gammaln, polygamma

# The essentials
π = np.pi
φ = (1 + np.sqrt(5)) / 2
ψ = 1 / φ


class GammaLab:
    def __init__(self):
        self.d = 4.0  # Start at our dimension
        self.mode = 0  # Plot mode
        self.modes = ["measures", "gamma", "complex", "derivatives", "poles"]
        self.show_complex = False

        # Setup figure
        self.fig = plt.figure(figsize=(14, 10), facecolor="#0a0a0a")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.update()
        plt.show()

    def v(self, d):
        """Volume of unit d-ball"""
        if abs(d) < 1e-10:
            return 1.0
        return π ** (d / 2) / gamma(d / 2 + 1)

    def s(self, d):
        """Surface of unit (d-1)-sphere"""
        if abs(d) < 1e-10:
            return 2.0
        return 2 * π ** (d / 2) / gamma(d / 2)

    def c(self, d):
        """Complexity V×S"""
        return self.v(d) * self.s(d)

    def on_key(self, event):
        """Handle keyboard input"""
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
        """Refresh all plots"""
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
        """Plot V, S, and C=V×S"""
        gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)

        d_range = np.linspace(-1, 15, 1000)

        # Volume
        ax1 = self.fig.add_subplot(gs[0, 0], facecolor="#1a1a1a")
        v_vals = [self.v(d) for d in d_range]
        ax1.plot(d_range, v_vals, "cyan", lw=2, alpha=0.8)
        ax1.axvline(self.d, color="yellow", lw=2, alpha=0.5)
        ax1.axhline(self.v(self.d), color="yellow", lw=1, alpha=0.3)
        ax1.set_title(
            f"Volume V_d = π^(d/2)/Γ(d/2+1) | V_{self.d:.1f} = {self.v(self.d):.4f}",
            color="cyan",
        )
        ax1.grid(True, alpha=0.2)
        ax1.set_xlim(-1, 15)

        # Surface
        ax2 = self.fig.add_subplot(gs[0, 1], facecolor="#1a1a1a")
        s_vals = [self.s(d) for d in d_range]
        ax2.plot(d_range, s_vals, "magenta", lw=2, alpha=0.8)
        ax2.axvline(self.d, color="yellow", lw=2, alpha=0.5)
        ax2.axhline(self.s(self.d), color="yellow", lw=1, alpha=0.3)
        ax2.set_title(
            f"Surface S_d = 2π^(d/2)/Γ(d/2) | S_{self.d:.1f} = {self.s(self.d):.4f}",
            color="magenta",
        )
        ax2.grid(True, alpha=0.2)
        ax2.set_xlim(-1, 15)

        # Complexity
        ax3 = self.fig.add_subplot(gs[1, :], facecolor="#1a1a1a")
        c_vals = [self.c(d) for d in d_range]
        ax3.plot(d_range, c_vals, "lime", lw=3, alpha=0.8)
        ax3.axvline(self.d, color="yellow", lw=2, alpha=0.5)
        ax3.axhline(self.c(self.d), color="yellow", lw=1, alpha=0.3)

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
            f"Complexity C_d = V_d × S_d | C_{self.d:.1f} = {self.c(self.d):.4f}",
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
        """Direct gamma function visualization"""
        gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)

        # Real gamma
        ax1 = self.fig.add_subplot(gs[0, :], facecolor="#1a1a1a")
        x = np.linspace(-5, 5, 1000)
        y = []
        for xi in x:
            try:
                val = gamma(xi)
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
        ax1.set_title(f"Γ(x) | Γ({self.d/2:.2f}) = {gamma(self.d/2):.4f}", color="cyan")
        ax1.set_ylim(-10, 10)
        ax1.grid(True, alpha=0.2)

        # Log gamma
        ax2 = self.fig.add_subplot(gs[1, 0], facecolor="#1a1a1a")
        x_pos = np.linspace(0.01, 10, 1000)
        ax2.plot(x_pos, gammaln(x_pos), "magenta", lw=2)
        ax2.axvline(self.d / 2, color="yellow", lw=2, alpha=0.5)
        ax2.set_title("log Γ(x)", color="magenta")
        ax2.grid(True, alpha=0.2)

        # Digamma (derivative of log gamma)
        ax3 = self.fig.add_subplot(gs[1, 1], facecolor="#1a1a1a")
        ax3.plot(x_pos, digamma(x_pos), "lime", lw=2)
        ax3.axvline(self.d / 2, color="yellow", lw=2, alpha=0.5)
        ax3.set_title("ψ(x) = d/dx log Γ(x)", color="lime")
        ax3.grid(True, alpha=0.2)

    def plot_complex_gamma(self):
        """Complex plane gamma visualization"""
        ax = self.fig.add_subplot(111, facecolor="#1a1a1a")

        # Create complex grid
        re = np.linspace(-4, 4, 200)
        im = np.linspace(-4, 4, 200)
        Re, Im = np.meshgrid(re, im)
        Z = Re + 1j * Im

        # Compute |Γ(z)|
        Gamma_abs = np.abs(gamma(Z))
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
        """Gamma derivatives - polygamma functions"""
        gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)

        x = np.linspace(0.1, 10, 1000)

        for n, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            ax = self.fig.add_subplot(gs[row, col], facecolor="#1a1a1a")

            if n == 0:
                y = digamma(x)
                title = "ψ⁰(x) = Γ'(x)/Γ(x)"
                color = "cyan"
            else:
                y = polygamma(n - 1, x)
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
        """Gamma function poles and residues"""
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
                    val = gamma(Z_complex[i, j])
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
        """Find and display peaks"""
        d_range = np.linspace(0.1, 15, 5000)

        v_vals = [self.v(d) for d in d_range]
        s_vals = [self.s(d) for d in d_range]
        c_vals = [self.c(d) for d in d_range]

        v_peak = d_range[np.argmax(v_vals)]
        s_peak = d_range[np.argmax(s_vals)]
        c_peak = d_range[np.argmax(c_vals)]

        print("\n🎯 PEAKS FOUND:")
        print(f"  Volume peak:     d = {v_peak:.6f}")
        print(f"  Surface peak:    d = {s_peak:.6f}")
        print(f"  Complexity peak: d = {c_peak:.6f}")
        print(f"  Current d:       d = {self.d:.6f}")


if __name__ == "__main__":
    print(__doc__)
    lab = GammaLab()
