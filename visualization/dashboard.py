#!/usr/bin/env python3
"""
Dimensional Dashboard Core
=========================

Unified visualization dashboard for the Dimensional Emergence Framework.
Event-driven architecture that orchestrates multiple visualization modules.
"""

import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Import existing modules
from dimensional_landscape import DimensionalLandscape
from emergence_cascade import EmergenceCascade, EmergenceCascadeVisualizer
from matplotlib.widgets import Slider

# Import topo_viz functionality
try:
    import topo_viz

    TOPO_VIZ_AVAILABLE = True
    AVAILABLE_SCENES = topo_viz.list_scenes()
except ImportError as e:
    TOPO_VIZ_AVAILABLE = False
    AVAILABLE_SCENES = []
    warnings.warn(f"topo_viz.py not available: {e}. Using fallback topology view.")


class EventBus:
    """Simple event bus for parameter synchronization."""

    def __init__(self):
        self._subscribers = defaultdict(list)

    def publish(self, event_name, data):
        """Publish an event to all subscribers."""
        for callback in self._subscribers[event_name]:
            try:
                callback(data)
            except Exception as e:
                warnings.warn(f"Event handler failed for {event_name}: {e}")

    def subscribe(self, event_name, callback):
        """Subscribe to an event."""
        self._subscribers[event_name].append(callback)


class DashboardState:
    """Centralized state management for dashboard parameters."""

    def __init__(self, dimension=4.0, time=0.0, max_dimensions=8):
        self.dimension = dimension
        self.time = time
        self.max_dimensions = max_dimensions
        self.auto_evolve = False
        self.evolution_speed = 0.1

        # Visualization flags
        self.show_volume = True
        self.show_surface = True
        self.show_complexity = True
        self.show_critical_points = True


class LayoutManager:
    """Manages figure layout and subplot organization."""

    def __init__(self, layout="grid"):
        self.layout = layout
        self.main_fig = None
        self.subplots = {}

    def create_dashboard_figure(self):
        """Create the main dashboard figure with subplot grid."""
        if self.layout == "grid":
            self.main_fig = plt.figure(figsize=(16, 12))
            self.main_fig.suptitle(
                "Dimensional Emergence Framework Dashboard",
                fontsize=16,
                fontweight="bold",
            )

            # Create 2x3 grid layout
            self.subplots = {
                "landscape": self.main_fig.add_subplot(2, 3, 1, projection="3d"),
                "cascade": self.main_fig.add_subplot(2, 3, 2),
                "topology": self.main_fig.add_subplot(2, 3, 3, projection="3d"),
                "morphic": self.main_fig.add_subplot(2, 3, 4),
                "phase_flow": self.main_fig.add_subplot(2, 3, 5),
                "controls": self.main_fig.add_subplot(2, 3, 6),
            }

            # Adjust layout
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        return self.main_fig, self.subplots


class TopologyViewController:
    """Enhanced controller that integrates topo_viz.py functionality."""

    def __init__(self):
        self.current_scene = "gamma_volume" if TOPO_VIZ_AVAILABLE else "basic_torus"
        self.scene_params = {"n_max": 10.0, "steps": 301}
        self.ax = None
        self.available_scenes = AVAILABLE_SCENES

        # Scene categories for better organization
        self.scene_categories = {
            "Gamma Functions": ["gamma_volume", "gamma_area"],
            "QWZ Models": [
                "qwz_curvature",
                "bz_torus_curvature_cloud",
                "bz_torus_bulged",
                "qwz_cylinder_bands",
            ],
            "Weierstrass": ["wp_domain", "wp_sphere", "wp_sphere_cuts"],
            "Topology": ["hopf_link", "torus_engine", "torus_defects_ribbon"],
            "Pump Family": ["pump_bloch", "pump_loops", "pump_chern"],
            "Other": ["simplex_orthant", "simplex_regular", "padic_tree"],
            "Interactive": ["live_qwz", "live_bz_torus_bulged", "live_pump", "live_wp"],
        }

    def set_axis(self, ax):
        """Set the matplotlib axis for rendering."""
        self.ax = ax
        # Configure for orthographic projection if 3D
        if hasattr(ax, "set_proj_type"):
            ax.set_proj_type("ortho")
            ax.set_box_aspect((1, 1, 1))
            # Use golden ratio view angle like topo_viz
            (1 + 5**0.5) / 2.0
            ax.view_init(elev=35.4, azim=-45.0)  # deg(phi-1), -45°

    def set_scene(self, scene_name, **params):
        """Set the current scene and parameters."""
        if TOPO_VIZ_AVAILABLE and scene_name in AVAILABLE_SCENES:
            self.current_scene = scene_name
            self.scene_params.update(params)
        elif scene_name == "basic_torus":
            self.current_scene = scene_name
            self.scene_params.update(params)
        else:
            warnings.warn(f"Scene '{scene_name}' not available. Using fallback.")

    def render_scene(self, dimension=4.0):
        """Render the current topology scene."""
        if self.ax is None:
            return

        if TOPO_VIZ_AVAILABLE and self.current_scene in AVAILABLE_SCENES:
            self._render_topo_viz_scene(dimension)
        else:
            self._render_fallback_scene(dimension)

    def _render_topo_viz_scene(self, dimension=4.0):
        """Render using topo_viz functionality."""
        try:
            # Clear the axis
            self.ax.clear()

            # Map dimension to scene parameters where applicable
            params = self.scene_params.copy()

            # Dimension-dependent parameter mapping
            if self.current_scene in ["gamma_volume", "gamma_area"]:
                params["n_max"] = max(dimension * 2, 10.0)
            elif self.current_scene in ["qwz_curvature", "bz_torus_bulged"]:
                params["m"] = (dimension - 2.0) * 0.5  # Map d=4 -> m=1.0
                params["Nk"] = min(81, max(41, int(dimension * 20)))
            elif self.current_scene == "padic_tree":
                params["depth"] = max(3, min(8, int(dimension)))

            # Handle interactive vs static scenes
            if self.current_scene.startswith("live_"):
                # For interactive scenes in dashboard, we render static version
                static_scene = self.current_scene.replace("live_", "")
                if static_scene in AVAILABLE_SCENES:
                    scene_func = topo_viz.SCENES[static_scene]
                else:
                    scene_func = topo_viz.SCENES[self.current_scene]
            else:
                scene_func = topo_viz.SCENES[self.current_scene]

            # Create temporary figure for scene rendering
            temp_fig = plt.figure(figsize=(6, 6))
            if self.current_scene in ["qwz_curvature"]:
                temp_ax = temp_fig.add_subplot(111)
            else:
                temp_ax = temp_fig.add_subplot(111, projection="3d")
                if hasattr(temp_ax, "set_proj_type"):
                    temp_ax.set_proj_type("ortho")
                    temp_ax.set_box_aspect((1, 1, 1))

            # Render scene to temporary figure
            scene_func(show=False, **params)

            # Transfer content to dashboard axis
            self._transfer_plot_content(temp_ax, self.ax)

            plt.close(temp_fig)

            # Set title
            scene_title = self.current_scene.replace("_", " ").title()
            self.ax.set_title(f"{scene_title} (d={dimension:.1f})")

        except Exception as e:
            warnings.warn(f"Error rendering topo_viz scene '{self.current_scene}': {e}")
            self._render_fallback_scene(dimension)

    def _transfer_plot_content(self, source_ax, target_ax):
        """Transfer plot content from source to target axis."""
        # This is a simplified transfer - in practice, you'd need more sophisticated
        # content transfer depending on the plot type
        try:
            # Copy basic properties
            target_ax.set_title(source_ax.get_title())
            target_ax.set_xlabel(source_ax.get_xlabel())
            target_ax.set_ylabel(source_ax.get_ylabel())
            if hasattr(source_ax, "get_zlabel"):
                target_ax.set_zlabel(source_ax.get_zlabel())
        except Exception:
            pass  # Ignore transfer errors

    def _render_fallback_scene(self, dimension=4.0):
        """Render a basic topology scene as fallback."""
        self.ax.clear()
        self.ax.set_title(f"Topological View (d={dimension:.1f})")

        # Create a dimension-dependent torus visualization
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, 2 * np.pi, 30)
        u, v = np.meshgrid(u, v)

        # Map dimension to torus parameters
        R = 1.0 + 0.3 * np.sin(dimension)  # Dimension affects major radius
        r = 0.3 + 0.1 * np.cos(dimension)  # Dimension affects minor radius

        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)

        # Color by dimension for visual feedback
        colors = plt.cm.viridis(dimension / 10.0)
        self.ax.plot_surface(x, y, z, alpha=0.7, color=colors)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

    def get_available_scenes(self):
        """Get list of available scenes organized by category."""
        if not TOPO_VIZ_AVAILABLE:
            return {"Basic": ["basic_torus"]}

        organized = {}
        for category, scenes in self.scene_categories.items():
            available = [s for s in scenes if s in AVAILABLE_SCENES]
            if available:
                organized[category] = available

        # Add any uncategorized scenes
        categorized_scenes = set()
        for scenes in self.scene_categories.values():
            categorized_scenes.update(scenes)

        uncategorized = [s for s in AVAILABLE_SCENES if s not in categorized_scenes]
        if uncategorized:
            organized["Other"] = uncategorized

        return organized


class MorphicController:
    """Controller for morphic transformations visualization."""

    def __init__(self):
        self.ax = None

    def set_axis(self, ax):
        """Set the matplotlib axis for rendering."""
        self.ax = ax

    def render_morphic_view(self, dimension=4.0):
        """Render morphic transformation view."""
        if self.ax is None:
            return

        self.ax.clear()
        self.ax.set_title(f"Morphic Transformations (d={dimension:.1f})")

        # Create morphic visualization based on golden ratio
        phi = (1 + np.sqrt(5)) / 2
        t = np.linspace(0, 4 * np.pi, 1000)

        # Dimension affects the morphic scaling
        scale = dimension / 4.0

        x = scale * np.cos(t) * (1 + 0.2 * np.cos(phi * t))
        y = scale * np.sin(t) * (1 + 0.2 * np.sin(phi * t))

        self.ax.plot(x, y, linewidth=2, alpha=0.8)
        self.ax.set_xlabel("Real")
        self.ax.set_ylabel("Imaginary")
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)


class DimensionalDashboard:
    """Main dashboard class that orchestrates all visualizations."""

    def __init__(self, layout="grid"):
        # Core state and event management
        self.state = DashboardState()
        self.event_bus = EventBus()
        self.layout_manager = LayoutManager(layout)

        # Visualization modules
        self.landscape = DimensionalLandscape(d_max=12)
        self.cascade = EmergenceCascade(max_dimensions=8)
        self.cascade_viz = EmergenceCascadeVisualizer()
        self.topo_controller = TopologyViewController()
        self.morphic_controller = MorphicController()

        # UI elements
        self.dimension_slider = None
        self.time_slider = None
        self.play_button = None

        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Set up event handling for parameter synchronization."""
        self.event_bus.subscribe("dimension_changed", self._on_dimension_change)
        self.event_bus.subscribe("time_evolved", self._on_time_evolution)

    def _on_dimension_change(self, event_data):
        """Handle dimension parameter changes."""
        new_d = event_data["d"]
        self.state.dimension = new_d

        # Update scene parameters based on dimension
        if TOPO_VIZ_AVAILABLE:
            scene = self.topo_controller.current_scene
            if scene in ["gamma_volume", "gamma_area"]:
                self.topo_controller.scene_params["n_max"] = max(new_d * 2, 10.0)
            elif scene in ["qwz_curvature", "bz_torus_bulged"]:
                self.topo_controller.scene_params["m"] = (new_d - 2.0) * 0.5

        # Update all visualization modules
        self._update_landscape_view()
        self._update_topology_view()
        self._update_morphic_view()
        self._render_controls_info()

        # Redraw
        self.layout_manager.main_fig.canvas.draw()

    def _on_time_evolution(self, event_data):
        """Handle time evolution updates."""
        new_t = event_data["t"]
        self.state.time = new_t

        # Evolve the cascade
        dt = new_t - self.cascade.current_time
        if dt > 0:
            self.cascade.evolve_emergence(dt)

        self._update_cascade_view()
        self.layout_manager.main_fig.canvas.draw()

    def _add_interactive_controls(self):
        """Add sliders and buttons for interactive control."""
        # Make room for controls at bottom
        plt.subplots_adjust(bottom=0.15)

        # Dimension slider
        ax_dim = plt.axes([0.1, 0.05, 0.3, 0.03])
        self.dimension_slider = Slider(
            ax_dim, "Dimension", 0.1, 12.0, valinit=self.state.dimension, valfmt="%.1f"
        )
        self.dimension_slider.on_changed(self._on_dimension_slider_change)

        # Time slider
        ax_time = plt.axes([0.5, 0.05, 0.3, 0.03])
        self.time_slider = Slider(
            ax_time, "Time", 0.0, 10.0, valinit=self.state.time, valfmt="%.1f"
        )
        self.time_slider.on_changed(self._on_time_slider_change)

    def _on_dimension_slider_change(self, val):
        """Handle dimension slider changes."""
        self.event_bus.publish("dimension_changed", {"d": val, "source": "slider"})

    def _on_time_slider_change(self, val):
        """Handle time slider changes."""
        self.event_bus.publish("time_evolved", {"t": val, "source": "slider"})

    def _update_landscape_view(self):
        """Update the dimensional landscape view."""
        ax = self.layout_manager.subplots["landscape"]
        ax.clear()

        # Use simplified landscape rendering
        d_range = np.linspace(0.1, 12, 100)
        volumes = np.array([self.landscape.measures.ball_volume(d) for d in d_range])
        surfaces = np.array(
            [self.landscape.measures.sphere_surface(d) for d in d_range]
        )
        complexity = volumes * surfaces

        # Plot the curves
        ax.plot(d_range, volumes, label="Volume", linewidth=2)
        ax.plot(d_range, surfaces, label="Surface", linewidth=2)
        ax.plot(d_range, complexity, label="Complexity", linewidth=2)

        # Mark current dimension
        current_v = self.landscape.measures.ball_volume(self.state.dimension)
        ax.axvline(self.state.dimension, color="red", linestyle="--", alpha=0.7)
        ax.scatter([self.state.dimension], [current_v], color="red", s=100, zorder=5)

        ax.set_xlabel("Dimension")
        ax.set_ylabel("Measure")
        ax.set_title(f"Dimensional Landscape (d={self.state.dimension:.1f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _update_cascade_view(self):
        """Update the emergence cascade view."""
        ax = self.layout_manager.subplots["cascade"]
        ax.clear()

        # Simplified cascade visualization
        dimensions = range(self.cascade.max_dim)
        densities = np.abs(self.cascade.phase_densities)

        bars = ax.bar(dimensions, densities, alpha=0.7)

        # Color bars based on emergence status
        for i, bar in enumerate(bars):
            if i in self.cascade.emergence_times:
                bar.set_color("green")
            else:
                bar.set_color("gray")

        ax.set_xlabel("Dimension")
        ax.set_ylabel("Phase Density")
        ax.set_title(f"Emergence Cascade (t={self.state.time:.1f})")
        ax.grid(True, alpha=0.3)

    def _update_topology_view(self):
        """Update the topology view using enhanced controller."""
        self.topo_controller.render_scene(self.state.dimension)

    def _update_morphic_view(self):
        """Update the morphic transformations view."""
        self.morphic_controller.render_morphic_view(self.state.dimension)

    def _render_all_views(self):
        """Render all visualization components."""
        self._update_landscape_view()
        self._update_cascade_view()
        self._update_topology_view()
        self._update_morphic_view()

        # Add placeholder for phase flow
        ax = self.layout_manager.subplots["phase_flow"]
        ax.clear()
        ax.set_title("Phase Flow Dynamics")
        ax.text(
            0.5,
            0.5,
            "Phase Flow\n(Coming Soon)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        # Render controls info
        self._render_controls_info()

    def _render_controls_info(self):
        """Render the controls information panel."""
        ax = self.layout_manager.subplots["controls"]
        ax.clear()
        ax.axis("off")

        # Topology scene info
        ax.text(
            0.1,
            0.9,
            "Topology Scene:",
            fontweight="bold",
            transform=ax.transAxes,
            fontsize=10,
        )
        scene_display = self.topo_controller.current_scene.replace("_", " ").title()
        ax.text(
            0.1, 0.8, f"Current: {scene_display}", transform=ax.transAxes, fontsize=9
        )

        if TOPO_VIZ_AVAILABLE:
            ax.text(
                0.1,
                0.7,
                f"Available: {len(AVAILABLE_SCENES)} scenes",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.text(
                0.1,
                0.6,
                "Categories:",
                fontweight="bold",
                transform=ax.transAxes,
                fontsize=9,
            )
            categories = list(self.topo_controller.get_available_scenes().keys())[:3]
            for i, cat in enumerate(categories):
                ax.text(
                    0.1, 0.55 - i * 0.08, f"• {cat}", transform=ax.transAxes, fontsize=8
                )

        # Current parameters
        ax.text(
            0.1,
            0.35,
            "Parameters:",
            fontweight="bold",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.text(
            0.1,
            0.28,
            f"d = {self.state.dimension:.1f}",
            transform=ax.transAxes,
            fontsize=8,
        )
        if (
            hasattr(self.topo_controller, "scene_params")
            and self.topo_controller.scene_params
        ):
            params_str = ", ".join(
                [
                    f"{k}={v:.1f}" if isinstance(v, (int, float)) else f"{k}={v}"
                    for k, v in list(self.topo_controller.scene_params.items())[:2]
                ]
            )
            ax.text(0.1, 0.21, params_str, transform=ax.transAxes, fontsize=8)

        # Interactive controls
        ax.text(
            0.1,
            0.15,
            "Controls:",
            fontweight="bold",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.text(0.1, 0.08, "• n/p: Switch scenes", transform=ax.transAxes, fontsize=8)
        ax.text(
            0.1, 0.01, "• Sliders: Adjust params", transform=ax.transAxes, fontsize=8
        )

    def add_scene_selector(self):
        """Add topology scene selection functionality."""
        if TOPO_VIZ_AVAILABLE and len(AVAILABLE_SCENES) > 1:
            # Create scene selector (simplified - in full implementation would be a dropdown)
            self.current_scene_index = 0
            self.scene_list = AVAILABLE_SCENES

            # Add keyboard callback for scene switching
            def on_key_press(event):
                if event.key == "n":  # Next scene
                    self.current_scene_index = (self.current_scene_index + 1) % len(
                        self.scene_list
                    )
                    scene = self.scene_list[self.current_scene_index]
                    self.topo_controller.set_scene(scene)
                    self._update_topology_view()
                    self.main_fig.canvas.draw()
                    print(f"Switched to scene: {scene}")
                elif event.key == "p":  # Previous scene
                    self.current_scene_index = (self.current_scene_index - 1) % len(
                        self.scene_list
                    )
                    scene = self.scene_list[self.current_scene_index]
                    self.topo_controller.set_scene(scene)
                    self._update_topology_view()
                    self.main_fig.canvas.draw()
                    print(f"Switched to scene: {scene}")

            if hasattr(self, "main_fig"):
                self.main_fig.canvas.mpl_connect("key_press_event", on_key_press)
                print("Scene switching enabled: Press 'n' for next, 'p' for previous")

    def launch(self):
        """Create and display the enhanced dashboard."""
        # Create the layout
        fig, axes = self.layout_manager.create_dashboard_figure()
        self.main_fig = fig  # Store reference for scene switching

        # Set axes for controllers
        self.topo_controller.set_axis(axes["topology"])
        self.morphic_controller.set_axis(axes["morphic"])

        # Add interactive controls
        self._add_interactive_controls()

        # Add scene selector for topology
        self.add_scene_selector()

        # Initial render
        self._render_all_views()

        # Print usage info
        if TOPO_VIZ_AVAILABLE:
            print("\n🎯 ENHANCED TOPOLOGY DASHBOARD LAUNCHED!")
            print(f"📊 {len(AVAILABLE_SCENES)} mathematical scenes available")
            print("⌨️  Keyboard controls:")
            print("   'n' = Next topology scene")
            print("   'p' = Previous topology scene")
            print(f"\n🔬 Current scene: {self.topo_controller.current_scene}")
            print("\n📚 Scene categories:")
            for cat, scenes in self.topo_controller.get_available_scenes().items():
                print(f"   {cat}: {len(scenes)} scenes")
        else:
            print("\n📊 DASHBOARD LAUNCHED (Basic topology mode)")
            print("📝 Install topo_viz dependencies for advanced scenes")

        # Show the dashboard
        plt.show()

        return fig


def main():
    """Launch the enhanced dimensional dashboard with topo_viz integration."""
    print("🚀 Launching Dimensional Emergence Dashboard...")

    # Check dependencies
    if TOPO_VIZ_AVAILABLE:
        print(f"✅ topo_viz.py loaded successfully ({len(AVAILABLE_SCENES)} scenes)")
    else:
        print("⚠️  topo_viz.py not available - using basic topology mode")

    try:
        dashboard = DimensionalDashboard()
        dashboard.launch()
    except Exception as e:
        print(f"❌ Dashboard launch failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
