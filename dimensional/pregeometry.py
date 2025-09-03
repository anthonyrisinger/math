#!/usr/bin/env python3
"""
Pre-Geometry Visualizer (n = -1) - MATPLOTLIB ELIMINATED
=========================================================

DEPRECATED: Interactive visualization moved to modern dashboard system.
Use: python -m dimensional --pregeometry or modern dashboard for visualization.

Mathematical core functions preserved.
"""

# MATPLOTLIB ELIMINATED - visualization moved to modern dashboard
import warnings

import numpy as np
from scipy.special import gamma

from .core import PHI, PI


# Mock matplotlib components for deprecated code
class _MockMatplotlib:
    def axes(self, *args, **kwargs):
        return None

    def show(self):
        pass


class _MockWidget:
    def __init__(self, *args, **kwargs):
        pass

    def on_changed(self, func):
        pass

    def on_clicked(self, func):
        pass


plt = _MockMatplotlib()
Slider = _MockWidget
Button = _MockWidget


def _deprecated_viz_warning():
    warnings.warn(
        "PreGeometry matplotlib visualization DEPRECATED. "
        "Use 'python -m dimensional --pregeometry' or modern dashboard instead.",
        DeprecationWarning,
        stacklevel=2,
    )


class PreGeometry:
    """Model the n=-1 pre-geometric primordial state."""

    def __init__(self):
        # Pre-geometric parameters
        self.n_range = np.linspace(-1, 1, 1000)
        self.time = 0.0
        self.oscillation_rate = PHI

        # Compute extended gamma for negative dimensions
        self.compute_negative_measures()

        # Wave function of pre-geometry
        self.psi = np.zeros(len(self.n_range), dtype=complex)
        self.initialize_wavefunction()

    def compute_negative_measures(self):
        """Compute measures for negative and fractional dimensions."""
        self.volumes = []
        self.surfaces = []

        for n in self.n_range:
            if n > -0.5 and n < 0.5:
                # Near zero, use special handling
                if abs(n) < 0.01:
                    v = 1.0
                    s = 2.0
                else:
                    # Use careful evaluation
                    try:
                        v = PI ** (n / 2) / gamma(n / 2 + 1)
                        s = 2 * PI ** (n / 2) / gamma(n / 2)
                    except BaseException:
                        v = 1.0
                        s = 2.0
            else:
                # For negative dimensions, we get oscillating infinities
                # We'll represent this as modulated complex values
                try:
                    # Complex extension
                    phase = PI * n
                    magnitude = (
                        abs(1 / gamma(n / 2 + 1))
                        if n < 0
                        else PI ** (n / 2) / abs(gamma(n / 2 + 1))
                    )
                    v = magnitude * np.exp(1j * phase)

                    magnitude_s = (
                        abs(2 / gamma(n / 2))
                        if n < 0
                        else 2 * PI ** (n / 2) / abs(gamma(n / 2))
                    )
                    s = magnitude_s * np.exp(1j * phase)
                except BaseException:
                    v = 0j
                    s = 0j

            self.volumes.append(v)
            self.surfaces.append(s)

        self.volumes = np.array(self.volumes)
        self.surfaces = np.array(self.surfaces)

    def initialize_wavefunction(self):
        """Initialize the pre-geometric wavefunction."""
        # Gaussian packet centered at n=-1 with golden ratio width
        center = -1.0
        width = 1 / PHI

        for i, n in enumerate(self.n_range):
            # Oscillating phase based on position
            phase = np.exp(1j * PHI * n)

            # Gaussian envelope
            envelope = np.exp(-((n - center) ** 2) / (2 * width**2))

            self.psi[i] = envelope * phase

        # Normalize
        norm = np.sqrt(np.sum(np.abs(self.psi) ** 2))
        self.psi /= norm

    def evolve_wavefunction(self, dt):
        """Evolve the pre-geometric wavefunction."""
        self.time += dt

        # The fundamental equation: ∂Ψ/∂n = iφΨ
        for i in range(len(self.psi)):
            # Dimensional gradient term
            if i > 0 and i < len(self.psi) - 1:
                grad2 = self.psi[i + 1] - 2 * self.psi[i] + self.psi[i - 1]
            else:
                grad2 = 0

            # Evolution with golden ratio coupling
            self.psi[i] *= np.exp(1j * PHI * dt)

            # Diffusion term for stability
            self.psi[i] += 0.01 * grad2 * dt

        # Renormalize
        norm = np.sqrt(np.sum(np.abs(self.psi) ** 2))
        if norm > 0:
            self.psi /= norm

    def get_probability_density(self):
        """Get probability density |Ψ|²."""
        return np.abs(self.psi) ** 2

    def get_phase_field(self):
        """Get the phase field arg(Ψ)."""
        return np.angle(self.psi)


class PreGeometryVisualizer:
    """Interactive visualizer for pre-geometric state."""

    def __init__(self):
        self.pregeom = PreGeometry()
        self.auto_evolve = False
        self.anim = None

        # Store view angles
        self.elev = 36.87  # Golden angle
        self.azim = -45

    def create_figure(self):
        """Create the interactive visualization."""
        # DEPRECATED: Use modern dashboard instead
        _deprecated_viz_warning()
        return self._get_mathematical_results()
        self.fig.suptitle(
            "Pre-Geometry: The n=-1 Primordial State Before Dimension",
            fontsize=14,
            fontweight="bold",
        )

        # Create subplots
        self.ax3d = self.fig.add_subplot(121, projection="3d")
        self.ax2d = self.fig.add_subplot(122)

        # Basic 3D axis setup (replacing setup_3d_axis functionality)
        self.ax3d.set_xlabel("x")
        self.ax3d.set_ylabel("y")
        self.ax3d.set_zlabel("Pre-Geometric Potential")
        self.ax3d.set_title("Pre-Geometric Potential Field")

        # Controls
        self._create_controls()

        # Initial plo
        self.update_plots()

        print("PRE-GEOMETRY VISUALIZER (n = -1)")
        print("=" * 60)
        print("CONCEPT: Before dimension emerges, there is the pre-geometric")
        print("         void where n=-1. This is the primordial field from")
        print("         which dimension itself crystallizes.")
        print()
        print("KEY INSIGHTS:")
        print("• Γ(-1/2) = -2√π (negative fractional factorial)")
        print("• Oscillating infinities suggest fundamental instability")
        print("• Golden ratio φ governs the emergence dynamics")
        print("• Time emerges as t = ∫φ dn from dimensional change")
        print()
        print("VISUALIZATION:")
        print("• Left: 3D potential field in pre-geometric space")
        print("• Right: Wavefunction evolution and probability density")
        print("• Watch how dimension wants to 'crystallize' from n=-1")

    def _create_controls(self):
        """Create interactive controls."""
        # Time slider
        ax_time = plt.axes([0.15, 0.02, 0.3, 0.03])
        self.time_slider = Slider(ax_time, "Evolution", 0, 10, valinit=0)
        self.time_slider.on_changed(self.on_time_change)

        # Buttons
        ax_auto = plt.axes([0.5, 0.02, 0.06, 0.03])
        ax_reset = plt.axes([0.57, 0.02, 0.06, 0.03])
        ax_pulse = plt.axes([0.64, 0.02, 0.06, 0.03])

        self.btn_auto = Button(ax_auto, "Auto")
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_pulse = Button(ax_pulse, "Pulse")

        self.btn_auto.on_clicked(self.toggle_auto)
        self.btn_reset.on_clicked(self.reset)
        self.btn_pulse.on_clicked(self.add_pulse)

    def on_time_change(self, val):
        """Handle time slider change."""
        target = val
        while self.pregeom.time < target:
            self.pregeom.evolve_wavefunction(0.01)
        self.update_plots()

    def toggle_auto(self, event):
        """Toggle automatic evolution."""
        self.auto_evolve = not self.auto_evolve
        self.btn_auto.label.set_text("Stop" if self.auto_evolve else "Auto")

        if self.auto_evolve and self.anim is None:
            self.anim = None  # animation.FuncAnimation deprecated
        elif not self.auto_evolve and self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None

    def animate_frame(self, frame):
        """Animation frame."""
        if self.auto_evolve:
            self.pregeom.evolve_wavefunction(0.02)
            self.time_slider.set_val(self.pregeom.time)
            self.update_plots()

    def reset(self, event):
        """Reset to initial state."""
        self.pregeom = PreGeometry()
        self.time_slider.set_val(0)
        self.update_plots()

    def add_pulse(self, event):
        """Add a perturbation pulse."""
        # Add a gaussian pulse at n=0
        for i, n in enumerate(self.pregeom.n_range):
            pulse = 0.3 * np.exp(-((n - 0) ** 2) / 0.1) * np.exp(1j * PI / 4)
            self.pregeom.psi[i] += pulse

        # Renormalize
        norm = np.sqrt(np.sum(np.abs(self.pregeom.psi) ** 2))
        if norm > 0:
            self.pregeom.psi /= norm

        self.update_plots()

    def update_plots(self):
        """Update both 3D and 2D plots."""
        # Save view angles before clearing
        if hasattr(self.ax3d, "elev"):
            self.elev = self.ax3d.elev
        if hasattr(self.ax3d, "azim"):
            self.azim = self.ax3d.azim

        # Update 3D plo
        self.ax3d.clear()
        self.ax3d.set_proj_type("ortho")
        self.ax3d.view_init(elev=self.elev, azim=self.azim)
        self.ax3d.set_box_aspect((1, 1, 1))

        # Create 3D surface showing the pre-geometric potential
        n_mesh = self.pregeom.n_range[::10]
        theta = np.linspace(0, 2 * PI, 50)
        N, Theta = np.meshgrid(n_mesh, theta)

        # Get probability density and phase
        prob_density = self.pregeom.get_probability_density()[::10]

        # Create surface using probability as radius
        R = np.outer(np.ones_like(theta), prob_density * 5 + 0.1)
        X = N
        Y = R * np.cos(Theta)
        Z = R * np.sin(Theta)

        self.ax3d.plot_surface(
            X,
            Y,
            Z,
            facecolors=None,  # cm.hsv deprecated
            alpha=0.8,
            linewidth=0,
            antialiased=True,
        )

        # Mark critical dimensions
        for n_crit, label, color in [
            (-1, "n=-1", "white"),
            (0, "void", "black"),
            (1, "d=1", "blue"),
        ]:
            if -1 <= n_crit <= 1:
                self.ax3d.plot(
                    [n_crit, n_crit],
                    [0, 0],
                    [-1, 1],
                    color=color,
                    linewidth=3,
                    alpha=0.7,
                )
                self.ax3d.text(n_crit, 0, 1.2, label, color=color, fontsize=10)

        # Add oscillating infinity indicators at n=-1
        t = self.pregeom.time
        osc_height = np.sin(PHI * t) * 0.5
        self.ax3d.scatter(
            [-1],
            [0],
            [osc_height],
            c="gold",
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
        )

        self.ax3d.set_xlabel("Dimension n")
        self.ax3d.set_ylabel("Re(Ψ)")
        self.ax3d.set_zlabel("Im(Ψ)")
        self.ax3d.set_xlim([-1, 1])
        self.ax3d.set_ylim([-1, 1])
        self.ax3d.set_zlim([-1, 1])
        self.ax3d.set_title("Pre-Geometric Potential Field")

        # Update 2D plo
        self.ax2d.clear()

        # Plot wavefunction components
        self.ax2d.plot(
            self.pregeom.n_range,
            np.real(self.pregeom.psi),
            "b-",
            alpha=0.7,
            label="Re(Ψ)",
        )
        self.ax2d.plot(
            self.pregeom.n_range,
            np.imag(self.pregeom.psi),
            "r-",
            alpha=0.7,
            label="Im(Ψ)",
        )
        self.ax2d.plot(
            self.pregeom.n_range,
            np.abs(self.pregeom.psi),
            "k-",
            linewidth=2,
            label="|Ψ|",
        )

        # Fill probability density
        prob = self.pregeom.get_probability_density()
        self.ax2d.fill_between(
            self.pregeom.n_range,
            0,
            prob,
            alpha=0.3,
            color="purple",
            label="|Ψ|²",
        )

        # Mark critical points
        self.ax2d.axvline(
            -1, color="gold", linestyle="--", alpha=0.5, label="n=-1"
        )
        self.ax2d.axvline(
            0, color="black", linestyle="--", alpha=0.5, label="void"
        )
        self.ax2d.axvline(
            PHI - 1, color="green", linestyle="--", alpha=0.5, label="φ-1"
        )

        self.ax2d.set_xlabel("Dimension n")
        self.ax2d.set_ylabel("Wavefunction Ψ")
        self.ax2d.set_title(
            f"Pre-Geometric Wavefunction (t={self.pregeom.time:.2f})"
        )
        self.ax2d.legend(loc="upper right", fontsize=8)
        self.ax2d.grid(True, alpha=0.3)
        self.ax2d.set_xlim([-1, 1])
        self.ax2d.set_ylim([-0.5, 1])

        # Add info tex
        info_text = (
            f"Time: {self.pregeom.time:.2f}\n"
            f"⟨n⟩ = {np.sum(self.pregeom.n_range * prob):.3f}\n"
            f"σ_n = {np.sqrt(np.sum((self.pregeom.n_range**2) * prob)):.3f}"
        )

        self.ax2d.text(
            0.02,
            0.98,
            info_text,
            transform=self.ax2d.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        self.fig.canvas.draw_idle()

    def run(self):
        """Run the visualization."""
        self.create_figure()
        plt.show()


def main():
    """Launch the pre-geometry visualizer."""
    visualizer = PreGeometryVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()

    def _get_mathematical_results(self):
        """Get mathematical results without matplotlib visualization."""
        return {
            "pre_geometric_state": self.compute_pre_geometric_state(),
            "oscillation_data": self.oscillation_data,
            "potential_field": self.potential_field,
            "dimensional_seeds": self.dimensional_seeds,
            "time": self.time,
        }

    def compute_pre_geometric_state(self):
        """Compute pre-geometric state mathematically."""
        # Primordial oscillation
        oscillation = np.sin(self.oscillation_rate * self.time)

        # Potential field computation
        self.potential_field = np.array(
            [self._potential_at_n(n) for n in self.n_range]
        )

        # Dimensional seeds
        self.dimensional_seeds = self._find_dimensional_seeds()

        # Oscillation data
        self.oscillation_data = {
            "amplitude": oscillation,
            "frequency": self.oscillation_rate,
            "phase": self.time % (2 * PI / self.oscillation_rate),
        }

        return {
            "potential_range": [
                np.min(self.potential_field),
                np.max(self.potential_field),
            ],
            "dimensional_seed_count": len(self.dimensional_seeds),
            "oscillation_amplitude": oscillation,
            "mathematical_state": "pre_geometric",
        }
