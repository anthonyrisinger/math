#!/usr/bin/env python3
"""
Dimensional Emergence: Interactive Framework
Organic visualization with 3D focus and proper geometry
"""

import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, cm
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d
from scipy.special import gamma, gammaln

warnings.filterwarnings("ignore")

# Core constants
PHI = (1 + np.sqrt(5)) / 2
PSI = 1 / PHI
VARPI = gamma(0.25) ** 2 / (2 * np.sqrt(2 * np.pi))
PI = np.pi

# Viewing parameters
ELEV = np.degrees(PHI - 1)
AZIM = -45


class GeometricMeasures:
    """
    Correct geometric formulas with proper handling of edge cases

    Key clarification:
    - 0-sphere (S^0) = two points {-1, +1}, measure = 2
    - 1-ball (B^1) = interval [-1, 1], volume = 2
    - 1-sphere (S^1) = circle, circumference = 2π
    """

    @staticmethod
    def ball_volume(d):
        """Volume of unit d-ball"""
        if abs(d) < 1e-10:
            return 1.0  # B^0 = single point

        # Avoid numerical issues
        if d > 170:
            log_vol = (d / 2) * np.log(PI) - gammaln(d / 2 + 1)
            return np.exp(np.real(log_vol))

        return PI ** (d / 2) / gamma(d / 2 + 1)

    @staticmethod
    def sphere_surface(d):
        """Surface area of unit (d-1)-sphere embedded in R^d"""
        if abs(d) < 1e-10:
            # Limit as d→0 is undefined; we set a finite value for visualization
            return 2.0

        if abs(d - 1) < 1e-10:
            return 2.0  # S^0 = two points

        # General formula
        if d > 170:
            log_surf = np.log(2) + (d / 2) * np.log(PI) - gammaln(d / 2)
            return np.exp(np.real(log_surf))

        return 2 * PI ** (d / 2) / gamma(d / 2)


class InteractiveDimensionalFramework:
    """
    Interactive visualization with organic flow
    """

    def __init__(self):
        self.time = 0
        self.paused = False
        self.current_view = 0

        # Precompute dimensional landscape
        self.compute_landscape()

        # Phase dynamics state
        self.phase_state = np.zeros(13, dtype=complex)
        self.phase_state[0] = 1.0

    def compute_landscape(self):
        """Compute the dimensional landscape once"""
        # Continuous dimension range
        self.d_range = np.linspace(0.1, 15, 3000)

        # Compute measures
        self.volumes = np.array(
            [GeometricMeasures.ball_volume(d) for d in self.d_range]
        )
        self.surfaces = np.array(
            [GeometricMeasures.sphere_surface(d) for d in self.d_range]
        )

        # Smooth any numerical artifacts
        self.volumes = np.clip(self.volumes, 0, 100)
        self.surfaces = np.clip(self.surfaces, 0, 100)

        # Find peaks organically (no forced values)
        self.vol_peak_idx = np.argmax(self.volumes)
        self.surf_peak_idx = np.argmax(self.surfaces)

        # Compute derived quantities
        self.ratios = self.surfaces / np.maximum(self.volumes, 1e-10)
        self.log_volumes = np.log(1 + self.volumes)
        self.log_surfaces = np.log(1 + self.surfaces)

        # Phase field (complex-valued landscape)
        self.phase_field = self.volumes * np.exp(1j * self.d_range * PI / 6)

    def create_visualization(self):
        """Create interactive 4-panel visualization"""

        # Create figure with 2x2 layout
        self.fig = plt.figure(figsize=(16, 14))
        gs = gridspec.GridSpec(2, 2, hspace=0.25, wspace=0.2)

        # ========== Panel 1: Main 3D Landscape ==========
        self.ax1 = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.ax1.set_proj_type("ortho")
        self.ax1.view_init(elev=ELEV, azim=AZIM)
        self.ax1.set_box_aspect((1, 1, 1))

        # ========== Panel 2: Phase Space 3D ==========
        self.ax2 = self.fig.add_subplot(gs[0, 1], projection="3d")
        self.ax2.set_proj_type("ortho")
        self.ax2.view_init(elev=ELEV, azim=AZIM + 90)
        self.ax2.set_box_aspect((1, 1, 1))

        # ========== Panel 3: Energy Flow 3D ==========
        self.ax3 = self.fig.add_subplot(gs[1, 0], projection="3d")
        self.ax3.set_proj_type("ortho")
        self.ax3.view_init(elev=ELEV + 20, azim=AZIM)
        self.ax3.set_box_aspect((1, 1, 1))

        # ========== Panel 4: Emergence Dynamics 3D ==========
        self.ax4 = self.fig.add_subplot(gs[1, 1], projection="3d")
        self.ax4.set_proj_type("ortho")
        self.ax4.view_init(elev=ELEV - 20, azim=AZIM + 45)
        self.ax4.set_box_aspect((1, 1, 1))

        # Add control sliders
        ax_time = plt.axes([0.15, 0.02, 0.3, 0.02])
        ax_rotation = plt.axes([0.55, 0.02, 0.3, 0.02])

        self.time_slider = Slider(ax_time, "Phase", 0, 2 * PI, valinit=0)
        self.rotation_slider = Slider(ax_rotation, "Rotation", -180, 180, valinit=AZIM)

        self.time_slider.on_changed(self.update_phase)
        self.rotation_slider.on_changed(self.update_rotation)

        # Initial plot
        self.update_all_panels()

    def update_phase(self, val):
        """Update based on phase slider"""
        self.time = val
        self.update_all_panels()

    def update_rotation(self, val):
        """Update viewing angle"""
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.view_init(elev=ELEV, azim=val)
        self.fig.canvas.draw_idle()

    def update_all_panels(self):
        """Update all four panels organically"""

        # Clear all axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()

        # Time-dependent modulation
        t = self.time
        modulation = np.sin(t) * 0.5 + 1.0

        # ========== Panel 1: Dimensional Landscape ==========
        # Create surface mesh
        d_mesh = self.d_range[::30]  # Subsample for mesh
        theta_mesh = np.linspace(0, 2 * PI, 40)
        D, Theta = np.meshgrid(d_mesh, theta_mesh)

        # Interpolate measures onto mesh
        vol_interp = interp1d(self.d_range, self.volumes, kind="cubic")
        surf_interp = interp1d(self.d_range, self.surfaces, kind="cubic")

        V_mesh = vol_interp(d_mesh)
        S_mesh = surf_interp(d_mesh)

        # Create 3D surface (dimension, volume*cos(θ), volume*sin(θ))
        X = D
        Y = np.outer(np.cos(Theta[:, 0]), V_mesh)
        Z = np.outer(np.sin(Theta[:, 0]), V_mesh)

        # Color by surface area (4th dimension)
        colors = np.outer(np.ones(len(theta_mesh)), S_mesh)

        self.ax1.plot_surface(
            X,
            Y,
            Z,
            facecolors=cm.viridis(colors / np.max(colors)),
            alpha=0.7,
            linewidth=0,
            antialiased=True,
        )

        # Add trajectory
        self.ax1.plot(
            self.d_range[::10],
            self.volumes[::10] * np.cos(t),
            self.volumes[::10] * np.sin(t),
            "w-",
            linewidth=2,
            alpha=0.8,
        )

        self.ax1.set_xlabel("Dimension")
        self.ax1.set_ylabel("V·cos(φ)")
        self.ax1.set_zlabel("V·sin(φ)")
        self.ax1.set_title("Dimensional Landscape (colored by surface area)")

        # ========== Panel 2: Phase Space ==========
        # Complex phase representation
        phase_real = np.real(self.phase_field) * modulation
        phase_imag = np.imag(self.phase_field) * modulation
        phase_mag = np.abs(self.phase_field)

        # Create spiral embedding
        spiral_x = self.d_range
        spiral_y = phase_real
        spiral_z = phase_imag

        # Plot as continuous surface
        n_spiral = 50
        for i in range(0, len(self.d_range) - n_spiral, n_spiral):
            segment = slice(i, i + n_spiral)
            colors_seg = cm.plasma(phase_mag[segment] / np.max(phase_mag))

            for j in range(n_spiral - 1):
                idx = i + j
                self.ax2.plot(
                    [spiral_x[idx], spiral_x[idx + 1]],
                    [spiral_y[idx], spiral_y[idx + 1]],
                    [spiral_z[idx], spiral_z[idx + 1]],
                    color=colors_seg[j],
                    linewidth=2,
                    alpha=0.8,
                )

        self.ax2.set_xlabel("Dimension")
        self.ax2.set_ylabel("Re(φ)")
        self.ax2.set_zlabel("Im(φ)")
        self.ax2.set_title("Phase Space Evolution")

        # ========== Panel 3: Energy Flow Surface ==========
        # Create energy flow field
        energy = self.volumes * self.surfaces * modulation
        flow_gradient = np.gradient(energy)

        # Parametric surface
        u = np.linspace(0, 2 * PI, 30)
        v = self.d_range[::50]
        U, V = np.meshgrid(u, v)

        # Map to 3D
        energy_v = interp1d(self.d_range, energy, kind="cubic")(v)
        X3 = V
        Y3 = np.outer(energy_v, np.cos(U[0]))
        Z3 = np.outer(energy_v, np.sin(U[0]))

        # Color by gradient (phase velocity)
        grad_v = interp1d(self.d_range, flow_gradient, kind="cubic")(v)
        colors3 = np.outer(grad_v, np.ones(len(u)))

        self.ax3.plot_surface(
            X3,
            Y3,
            Z3,
            facecolors=cm.coolwarm(0.5 + colors3 / np.max(np.abs(colors3)) / 2),
            alpha=0.6,
            linewidth=0,
        )

        self.ax3.set_xlabel("Dimension")
        self.ax3.set_ylabel("E·cos(θ)")
        self.ax3.set_zlabel("E·sin(θ)")
        self.ax3.set_title("Energy Flow (V×S product)")

        # ========== Panel 4: Emergence Dynamics ==========
        # Lemniscate-inspired parametric surface
        s = np.linspace(0, 2 * PI, 100)
        t_param = np.linspace(0, 4 * PI, 100)
        S, T = np.meshgrid(s, t_param)

        # Lemniscate modulation
        a = 3 * modulation
        X4 = a * np.cos(T) / (1 + np.sin(S) ** 2)
        Y4 = a * np.sin(T) * np.cos(S) / (1 + np.sin(S) ** 2)
        Z4 = np.sqrt(np.abs(T)) * PSI

        # Color by dimensional density
        density = np.exp(-T / (2 * PI))

        self.ax4.plot_surface(
            X4, Y4, Z4, facecolors=cm.twilight(density), alpha=0.7, linewidth=0
        )

        # Add emergence trajectory
        traj_t = np.linspace(0, t, 50)
        traj_x = a * np.cos(traj_t) / (1 + np.sin(traj_t) ** 2)
        traj_y = a * np.sin(traj_t) * np.cos(traj_t) / (1 + np.sin(traj_t) ** 2)
        traj_z = np.sqrt(np.abs(traj_t)) * PSI

        self.ax4.plot(traj_x, traj_y, traj_z, "gold", linewidth=3, alpha=0.9)

        self.ax4.set_xlabel("Phase X")
        self.ax4.set_ylabel("Phase Y")
        self.ax4.set_zlabel("√d")
        self.ax4.set_title("Emergence Manifold (∞-surface)")

        # Update canvas
        self.fig.canvas.draw_idle()

    def animate(self):
        """Create animation"""

        def update(frame):
            self.time = (frame * 0.05) % (2 * PI)
            self.time_slider.set_val(self.time)
            self.update_all_panels()
            return []

        self.anim = animation.FuncAnimation(
            self.fig, update, frames=200, interval=50, blit=False
        )
        return self.anim


def run_interactive():
    """Launch interactive framework"""

    print("DIMENSIONAL EMERGENCE - INTERACTIVE")
    print("=" * 50)
    print(f"Constants: ϖ={VARPI:.6f}, φ={PHI:.6f}")
    print()
    print("Controls:")
    print("  • Phase slider: Modulates the dimensional landscape")
    print("  • Rotation slider: Rotates all 3D views")
    print("  • Each panel shows different aspect of emergence")
    print()
    print("Geometry clarification:")
    print("  • 0-sphere S⁰ = two points {-1,+1}")
    print("  • 1-ball B¹ = interval [-1,1], volume = 2")
    print("  • 1-sphere S¹ = circle, circumference = 2π")
    print()

    framework = InteractiveDimensionalFramework()
    framework.create_visualization()

    # Optional: uncomment to enable animation
    # anim = framework.animate()

    plt.show()

    return framework


if __name__ == "__main__":
    framework = run_interactive()
