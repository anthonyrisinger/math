#!/usr/bin/env python3
"""
Emergence Cascade Visualizer
============================

Interactive 3D visualization of the sequential emergence of dimensions
from the void (d=0) through phase accumulation and critical transitions.
Shows the lemniscate manifold structure and dimensional birth events.

Run: python emergence_cascade.py
Controls: Time evolution, emergence thresholds, dimensional birthing
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.widgets import Button, CheckButtons, Slider

from core_measures import PHI, PI, PSI, DimensionalMeasures, setup_3d_axis


class EmergenceCascade:
    """Model the sequential emergence of dimensions."""

    def __init__(self, max_dimensions=8):
        self.max_dim = max_dimensions
        self.measures = DimensionalMeasures()

        # Emergence state
        self.current_time = 0.0
        self.phase_densities = np.zeros(max_dimensions, dtype=complex)
        self.phase_densities[0] = 1.0  # Start with void

        # Emergence events - when each dimension "crystallized"
        self.emergence_times = {}
        self.emergence_times[0] = 0.0  # Void always exists

        # Critical thresholds for emergence
        self.emergence_thresholds = {}
        for d in range(max_dimensions):
            self.emergence_thresholds[d] = self.measures.ball_volume(max(d, 0.01)) * 0.9

        # Lemniscate parameters for visualization
        self.lemniscate_scale = 2.0
        self.manifold_resolution = 100

        # Animation state
        self.auto_evolve = False
        self.emergence_speed = 0.1

    def evolve_emergence(self, dt):
        """Evolve the emergence process by time step dt."""
        self.current_time += dt

        # Phase accumulation follows golden ratio dynamics
        for d in range(1, self.max_dim):
            if d not in self.emergence_times:
                # Accumulate phase from lower dimensions
                inflow = 0.0
                for source_d in range(d):
                    if source_d in self.emergence_times:
                        # Phase flows with dimensional distance scaling
                        distance_factor = 1 / (d - source_d + PHI)
                        source_strength = abs(self.phase_densities[source_d])
                        rate = distance_factor * source_strength * dt
                        inflow += rate

                # Add phase with dimensional signature
                phase_rotation = np.exp(1j * PI * d / 6)
                self.phase_densities[d] += inflow * phase_rotation

                # Check for emergence
                current_magnitude = abs(self.phase_densities[d])
                threshold = self.emergence_thresholds[d]

                if current_magnitude >= threshold:
                    self.emergence_times[d] = self.current_time
                    print(f"🌟 Dimension {d} EMERGED at time {self.current_time:.2f}!")

                    # Normalize and stabilize
                    self.phase_densities[d] = (
                        threshold * phase_rotation / abs(phase_rotation)
                    )

                    # Seed higher dimensions
                    for higher_d in range(d + 1, min(d + 3, self.max_dim)):
                        self.phase_densities[higher_d] += 0.05 * np.exp(1j * PI / 4)

    def get_lemniscate_manifold(self, dimension_level):
        """Generate lemniscate surface for a specific dimensional level."""
        # Lemniscate: x = a*cos(t)/(1+sin²(s)), y = a*sin(t)*cos(s)/(1+sin²(s))
        t = np.linspace(0, 2 * PI, self.manifold_resolution)
        s = np.linspace(0, 2 * PI, self.manifold_resolution // 2)
        T, S = np.meshgrid(t, s)

        a = self.lemniscate_scale * (1 + dimension_level * 0.3)

        # Lemniscate equations
        denominator = 1 + np.sin(S) ** 2
        X = a * np.cos(T) / denominator
        Y = a * np.sin(T) * np.cos(S) / denominator
        Z = np.sqrt(np.abs(T)) * PSI + dimension_level * 0.5

        return X, Y, Z

    def get_emergence_trajectory(self, dimension):
        """Get the trajectory for a specific dimension's emergence."""
        if dimension not in self.emergence_times:
            return None

        emergence_time = self.emergence_times[dimension]

        # Parametric trajectory leading to emergence
        tau = np.linspace(0, emergence_time, 50)

        # Spiral trajectory with golden ratio scaling
        theta = tau * PHI
        r = tau / (emergence_time + 1) * 2

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = dimension + 0.1 * np.sin(tau * PI)

        return x, y, z

    def reset_cascade(self):
        """Reset to initial state."""
        self.current_time = 0.0
        self.phase_densities = np.zeros(self.max_dim, dtype=complex)
        self.phase_densities[0] = 1.0
        self.emergence_times = {0: 0.0}


class EmergenceCascadeVisualizer:
    """Interactive visualizer for dimensional emergence cascade."""

    def __init__(self):
        self.cascade = EmergenceCascade()
        self.show_manifolds = True
        self.show_trajectories = True
        self.show_phase_field = True
        self.anim = None

    def create_figure(self):
        """Create the interactive figure."""
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle(
            "Dimensional Emergence Cascade: The Birth of Reality",
            fontsize=14,
            fontweight="bold",
        )

        # Main 3D plot
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        setup_3d_axis(self.ax3d, "The Sequential Birth of Dimensions")

        # Controls
        self._create_controls()

        # Initial plot
        self.update_plot()

        print("EMERGENCE CASCADE VISUALIZER")
        print("=" * 60)
        print("THEORY: Dimensions emerge sequentially from the void through")
        print("        phase accumulation and critical transitions.")
        print()
        print("CONTROLS:")
        print("• Time slider: Manual evolution of emergence process")
        print("• Auto button: Automatic time evolution")
        print("• Speed slider: Control evolution rate")
        print("• Inject: Force emergence at specific dimension")
        print("• Reset: Return to primordial void")
        print("• Checkboxes: Toggle visualization elements")
        print()
        print("VISUALIZATION:")
        print("• Lemniscate surfaces: Dimensional manifolds")
        print("• Spiral trajectories: Paths to emergence")
        print("• Phase field: Complex phase evolution")
        print("• Color coding: Time of emergence")

    def _create_controls(self):
        """Create interactive controls."""
        # Time slider
        ax_time = plt.axes([0.15, 0.02, 0.35, 0.03])
        self.time_slider = Slider(ax_time, "Time", 0, 20, valinit=0)
        self.time_slider.on_changed(self.on_time_change)

        # Speed slider
        ax_speed = plt.axes([0.55, 0.02, 0.15, 0.03])
        self.speed_slider = Slider(ax_speed, "Speed", 0.01, 0.5, valinit=0.1)
        self.speed_slider.on_changed(self.on_speed_change)

        # Control buttons
        ax_auto = plt.axes([0.72, 0.02, 0.05, 0.03])
        ax_reset = plt.axes([0.78, 0.02, 0.05, 0.03])
        ax_inject = plt.axes([0.84, 0.02, 0.05, 0.03])

        self.btn_auto = Button(ax_auto, "Auto")
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_inject = Button(ax_inject, "Inject")

        self.btn_auto.on_clicked(self.toggle_auto)
        self.btn_reset.on_clicked(self.reset_cascade)
        self.btn_inject.on_clicked(self.inject_dimension)

        # Visualization toggles
        ax_checks = plt.axes([0.02, 0.5, 0.12, 0.15])
        self.checkboxes = CheckButtons(
            ax_checks,
            ["Manifolds", "Trajectories", "Phase Field"],
            [self.show_manifolds, self.show_trajectories, self.show_phase_field],
        )
        self.checkboxes.on_clicked(self.on_toggle_display)

    def on_time_change(self, val):
        """Handle time slider change."""
        target_time = val
        while self.cascade.current_time < target_time:
            self.cascade.evolve_emergence(0.05)
        self.update_plot()

    def on_speed_change(self, val):
        """Handle speed change."""
        self.cascade.emergence_speed = val

    def toggle_auto(self, event):
        """Toggle automatic evolution."""
        self.cascade.auto_evolve = not self.cascade.auto_evolve
        self.btn_auto.label.set_text("Stop" if self.cascade.auto_evolve else "Auto")

        if self.cascade.auto_evolve and self.anim is None:
            self.anim = animation.FuncAnimation(
                self.fig, self.animate_frame, interval=50, blit=False
            )
        elif not self.cascade.auto_evolve and self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None

    def animate_frame(self, frame):
        """Animation frame update."""
        if self.cascade.auto_evolve:
            self.cascade.evolve_emergence(self.cascade.emergence_speed)
            self.time_slider.set_val(self.cascade.current_time)
            self.update_plot()

    def reset_cascade(self, event):
        """Reset the cascade."""
        self.cascade.reset_cascade()
        self.time_slider.set_val(0)
        self.update_plot()

    def inject_dimension(self, event):
        """Force emergence of next dimension."""
        # Find next unemerged dimension
        for d in range(1, self.cascade.max_dim):
            if d not in self.cascade.emergence_times:
                self.cascade.phase_densities[d] = (
                    self.cascade.emergence_thresholds[d] * 1.1
                )
                print(f"💉 Forced emergence of dimension {d}")
                break
        self.update_plot()

    def on_toggle_display(self, label):
        """Handle display toggles."""
        if label == "Manifolds":
            self.show_manifolds = not self.show_manifolds
        elif label == "Trajectories":
            self.show_trajectories = not self.show_trajectories
        elif label == "Phase Field":
            self.show_phase_field = not self.show_phase_field
        self.update_plot()

    def update_plot(self):
        """Update the 3D visualization."""
        self.ax3d.clear()
        setup_3d_axis(self.ax3d, "The Sequential Birth of Dimensions")

        # Color map for emergence times
        if len(self.cascade.emergence_times) > 1:
            max_emergence_time = max(self.cascade.emergence_times.values())
            time_colormap = cm.plasma
        else:
            max_emergence_time = 1.0
            time_colormap = cm.plasma

        # Draw dimensional manifolds (lemniscate surfaces)
        if self.show_manifolds:
            for d in range(min(6, self.cascade.max_dim)):
                if d in self.cascade.emergence_times:
                    X, Y, Z = self.cascade.get_lemniscate_manifold(d)

                    # Color by emergence time
                    emergence_time = self.cascade.emergence_times[d]
                    color_value = emergence_time / max(max_emergence_time, 1.0)
                    surface_color = time_colormap(color_value)

                    # Alpha based on how long ago it emerged
                    time_since_emergence = self.cascade.current_time - emergence_time
                    alpha = 0.3 + 0.4 * np.exp(-time_since_emergence / 5)
                    alpha = min(0.7, alpha)  # Cap alpha for visibility

                    # Create facecolors array
                    facecolors = np.full(X.shape + (4,), list(surface_color))
                    facecolors[:, :, 3] = alpha

                    self.ax3d.plot_surface(
                        X, Y, Z, facecolors=facecolors, linewidth=0, antialiased=True
                    )

        # Draw emergence trajectories
        if self.show_trajectories:
            for d in range(1, min(6, self.cascade.max_dim)):
                if d in self.cascade.emergence_times:
                    traj = self.cascade.get_emergence_trajectory(d)
                    if traj is not None:
                        x, y, z = traj

                        # Color by dimension
                        color = cm.viridis(d / 8)

                        self.ax3d.plot(x, y, z, color=color, linewidth=3, alpha=0.8)

                        # Mark emergence point
                        self.ax3d.scatter(
                            [x[-1]],
                            [y[-1]],
                            [z[-1]],
                            c=[color],
                            s=150,
                            marker="*",
                            edgecolors="black",
                            linewidth=2,
                        )

                        # Label
                        self.ax3d.text(
                            x[-1],
                            y[-1],
                            z[-1] + 0.2,
                            f"D{d}",
                            fontsize=10,
                            color="black",
                            ha="center",
                        )

        # Draw phase field
        if self.show_phase_field:
            # Create a grid showing phase density at each dimensional level
            for d in range(min(8, self.cascade.max_dim)):
                phase_mag = abs(self.cascade.phase_densities[d])
                if phase_mag > 0.01:
                    # Position in space
                    theta = 2 * PI * d / 8
                    r = 1 + d * 0.3
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    z = d * 0.5

                    # Phase magnitude as sphere size
                    size = max(50, min(500, phase_mag * 500))

                    # Color by phase angle
                    phase_angle = np.angle(self.cascade.phase_densities[d])
                    color = cm.hsv((phase_angle + PI) / (2 * PI))

                    # Emergence status
                    if d in self.cascade.emergence_times:
                        marker = "o"
                        edge_color = "gold"
                        edge_width = 3
                    else:
                        marker = "o"
                        edge_color = "gray"
                        edge_width = 1

                    self.ax3d.scatter(
                        [x],
                        [y],
                        [z],
                        c=[color],
                        s=size,
                        marker=marker,
                        alpha=0.7,
                        edgecolors=edge_color,
                        linewidth=edge_width,
                    )

                    # Phase bar
                    bar_height = min(2, phase_mag * 2)
                    self.ax3d.plot(
                        [x, x],
                        [y, y],
                        [z, z + bar_height],
                        color="blue",
                        linewidth=4,
                        alpha=0.5,
                    )

        # Draw the void (d=0) specially
        self.ax3d.scatter(
            [0],
            [0],
            [0],
            c="white",
            s=300,
            marker="o",
            edgecolors="black",
            linewidth=3,
            alpha=0.9,
        )
        self.ax3d.text(
            0,
            0,
            0.3,
            "VOID\n(d=0)",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

        # Draw critical boundaries
        for d_crit, label, color in [
            (PI, "π-boundary", "red"),
            (2 * PI, "2π-boundary", "orange"),
        ]:
            if d_crit < 8:
                # Boundary plane
                xx, yy = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
                zz = np.full_like(xx, d_crit * 0.5)
                self.ax3d.plot_surface(xx, yy, zz, alpha=0.1, color=color)
                self.ax3d.text(0, 3, d_crit * 0.5, label, color=color, fontsize=10)

        # Information display
        emerged_dims = sorted(list(self.cascade.emergence_times.keys()))
        next_dim = None
        for d in range(1, self.cascade.max_dim):
            if d not in self.cascade.emergence_times:
                next_dim = d
                break

        info_text = f"Time: {self.cascade.current_time:.2f}\n"
        info_text += f"Emerged: {emerged_dims}\n"
        if next_dim is not None:
            phase_progress = abs(self.cascade.phase_densities[next_dim])
            threshold = self.cascade.emergence_thresholds[next_dim]
            progress_pct = (phase_progress / threshold) * 100
            info_text += f"Next: D{next_dim} ({progress_pct:.1f}%)"

        self.ax3d.text2D(
            0.02,
            0.98,
            info_text,
            transform=self.ax3d.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.9),
        )

        # Set limits
        self.ax3d.set_xlim([-4, 4])
        self.ax3d.set_ylim([-4, 4])
        self.ax3d.set_zlim([0, 6])

        self.ax3d.set_xlabel("Spatial X")
        self.ax3d.set_ylabel("Spatial Y")
        self.ax3d.set_zlabel("Dimensional Level")

        self.fig.canvas.draw_idle()

    def run(self):
        """Run the interactive visualization."""
        self.create_figure()
        plt.show()


def main():
    """Launch the emergence cascade visualizer."""
    visualizer = EmergenceCascadeVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()
