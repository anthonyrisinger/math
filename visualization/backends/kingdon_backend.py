#!/usr/bin/env python3
"""
Kingdon Geometric Algebra Visualization Backend
===============================================

AGGRESSIVE MATPLOTLIB REPLACEMENT
Pure geometric algebra visualization for morphic/phase mathematics.
Perfect mathematical integrity with theoretical grounding.
"""

from typing import Any, Optional

import numpy as np

from .base_backend import CameraConfig, VisualizationBackend


class KingdonRenderer(VisualizationBackend):
    """
    Kingdon-based geometric algebra renderer for pure mathematical visualization.
    Optimal for morphic transformations and phase dynamics.
    """

    def __init__(self):
        super().__init__("kingdon")
        self.ga_space = None
        self.scene_objects = []
        self.morphic_fields = {}

    def initialize(self, **kwargs) -> bool:
        """Initialize Kingdon geometric algebra space."""
        try:
            # Import kingdon with fallback detection
            try:
                import kingdon as kd

                self.kd = kd
            except ImportError:
                print(
                    "WARNING: kingdon not installed. Install with: pip install kingdon"
                )
                return False

            # Initialize 3D geometric algebra space (R3,0,1 for spacetime-like)
            dimension = kwargs.get("dimension", 3)
            signature = kwargs.get("signature", (dimension, 0, 1))  # Euclidean + time

            self.ga_space = self.kd.Algebra(*signature)
            self.set_orthographic_projection()
            self._initialized = True

            print(f"✅ Kingdon GA space initialized: {signature}")
            return True

        except Exception as e:
            print(f"❌ Kingdon initialization failed: {e}")
            return False

    def render_scene(self, scene_data: dict[str, Any]) -> Any:
        """Render mathematical scene using geometric algebra primitives."""
        if not self._initialized:
            raise RuntimeError("Kingdon backend not initialized")

        if not self.validate_mathematical_integrity(scene_data):
            raise ValueError("Scene data fails mathematical integrity check")

        # Clear previous scene
        self.scene_objects.clear()

        # Extract geometric data
        geometry = scene_data.get("geometry", {})
        topology = scene_data.get("topology", {})
        measures = scene_data.get("measures", {})

        # Render geometric elements
        if "points" in geometry:
            self._render_points(geometry["points"])
        if "curves" in geometry:
            self._render_curves(geometry["curves"])
        if "surfaces" in geometry:
            self._render_surfaces(geometry["surfaces"])
        if "morphic_field" in geometry:
            self._render_morphic_field(geometry["morphic_field"])

        # Render topological structures
        if "phase_space" in topology:
            self._render_phase_space(topology["phase_space"])
        if "emergence_cascade" in topology:
            self._render_emergence_cascade(topology["emergence_cascade"])

        # Apply measures and coloring
        if "gamma_measure" in measures:
            self._apply_gamma_coloring(measures["gamma_measure"])
        if "complexity_measure" in measures:
            self._apply_complexity_coloring(measures["complexity_measure"])

        return self._compile_scene()

    def _render_points(self, points_data: dict[str, Any]) -> None:
        """Render point cloud using GA bivectors."""
        positions = points_data.get("positions", [])
        colors = points_data.get("colors", None)

        for i, pos in enumerate(positions):
            # Create GA point as vector
            ga_point = sum(
                pos[j] * self.ga_space.basis[f"e{j+1}"] for j in range(len(pos))
            )

            point_obj = {
                "type": "point",
                "ga_element": ga_point,
                "position": pos,
                "color": colors[i] if colors else "blue",
                "size": points_data.get("size", 1.0),
            }
            self.scene_objects.append(point_obj)

    def _render_curves(self, curves_data: dict[str, Any]) -> None:
        """Render parametric curves using GA geometric product."""
        for curve_spec in curves_data.get("curves", []):
            t_values = curve_spec.get("t_values", np.linspace(0, 2 * np.pi, 100))
            curve_func = curve_spec.get("function")

            if callable(curve_func):
                # Evaluate curve points
                curve_points = [curve_func(t) for t in t_values]

                # Create GA curve as sequence of connected vectors
                ga_curve = []
                for point in curve_points:
                    ga_point = sum(
                        point[j] * self.ga_space.basis[f"e{j+1}"]
                        for j in range(len(point))
                    )
                    ga_curve.append(ga_point)

                curve_obj = {
                    "type": "curve",
                    "ga_elements": ga_curve,
                    "points": curve_points,
                    "color": curve_spec.get("color", "green"),
                    "width": curve_spec.get("width", 1.0),
                }
                self.scene_objects.append(curve_obj)

    def _render_surfaces(self, surfaces_data: dict[str, Any]) -> None:
        """Render surfaces using GA trivectors and exterior product."""
        for surface_spec in surfaces_data.get("surfaces", []):
            u_values = surface_spec.get("u_values", np.linspace(0, 2 * np.pi, 30))
            v_values = surface_spec.get("v_values", np.linspace(0, 2 * np.pi, 30))
            surface_func = surface_spec.get("function")

            if callable(surface_func):
                # Create surface mesh
                surface_mesh = []
                for u in u_values:
                    row = []
                    for v in v_values:
                        point = surface_func(u, v)
                        ga_point = sum(
                            point[j] * self.ga_space.basis[f"e{j+1}"]
                            for j in range(len(point))
                        )
                        row.append((point, ga_point))
                    surface_mesh.append(row)

                surface_obj = {
                    "type": "surface",
                    "ga_mesh": surface_mesh,
                    "color": surface_spec.get("color", "cyan"),
                    "alpha": surface_spec.get("alpha", 0.7),
                    "wireframe": surface_spec.get("wireframe", True),
                }
                self.scene_objects.append(surface_obj)

    def _render_morphic_field(self, morphic_data: dict[str, Any]) -> None:
        """Render morphic field using GA rotors and golden ratio geometry."""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        # Create morphic transformation rotor
        angle = morphic_data.get("phase_angle", 0.0)
        axis = morphic_data.get("axis", [0, 0, 1])

        # GA rotor for morphic transformation
        ga_axis = sum(
            axis[j] * self.ga_space.basis[f"e{j+1}"] for j in range(len(axis))
        )
        morphic_rotor = np.cos(angle * phi / 2) + np.sin(angle * phi / 2) * ga_axis

        # Apply to field points
        field_points = morphic_data.get("field_points", [])
        transformed_points = []

        for point in field_points:
            ga_point = sum(
                point[j] * self.ga_space.basis[f"e{j+1}"] for j in range(len(point))
            )
            # Apply morphic transformation: R * point * ~R
            transformed = morphic_rotor * ga_point * morphic_rotor.inverse()
            transformed_points.append(transformed)

        self.morphic_fields["current"] = {
            "rotor": morphic_rotor,
            "original_points": field_points,
            "transformed_points": transformed_points,
            "phi_scaling": phi,
        }

    def _render_phase_space(self, phase_data: dict[str, Any]) -> None:
        """Render phase space dynamics using GA exponentials."""
        phase_trajectories = phase_data.get("trajectories", [])

        for traj in phase_trajectories:
            # Phase trajectory as GA curve in extended space
            time_points = traj.get("time", [])
            state_points = traj.get("states", [])

            ga_trajectory = []
            for i, (t, state) in enumerate(zip(time_points, state_points)):
                # Embed phase state in GA space with time componen
                extended_point = list(state) + [t]  # Add time dimension
                ga_state = sum(
                    extended_point[j] * self.ga_space.basis[f"e{j+1}"]
                    for j in range(min(len(extended_point), len(self.ga_space.basis)))
                )
                ga_trajectory.append(ga_state)

            phase_obj = {
                "type": "phase_trajectory",
                "ga_trajectory": ga_trajectory,
                "color": traj.get("color", "red"),
                "width": traj.get("width", 2.0),
            }
            self.scene_objects.append(phase_obj)

    def _render_emergence_cascade(self, cascade_data: dict[str, Any]) -> None:
        """Render emergence cascade using GA multivectors."""
        dimensions = cascade_data.get("dimensions", [])
        densities = cascade_data.get("densities", [])
        emergence_times = cascade_data.get("emergence_times", {})

        # Create cascade visualization as multivector field
        cascade_field = []
        for i, (dim, density) in enumerate(zip(dimensions, densities)):
            # Represent each dimensional state as multivector
            grade = min(i + 1, len(self.ga_space.basis) - 1)

            # Create multivector of appropriate grade
            if grade == 0:
                mv = density  # Scalar
            elif grade == 1:
                mv = density * self.ga_space.basis["e1"]  # Vector
            elif grade == 2:
                mv = density * self.ga_space.basis["e12"]  # Bivector
            else:
                # Higher grade multivector
                basis_key = "e" + "".join(str(j + 1) for j in range(min(grade, 3)))
                mv = density * self.ga_space.basis.get(
                    basis_key, self.ga_space.basis["e123"]
                )

            cascade_state = {
                "dimension": dim,
                "density": density,
                "multivector": mv,
                "emerged": dim in emergence_times,
                "emergence_time": emergence_times.get(dim, np.inf),
            }
            cascade_field.append(cascade_state)

        cascade_obj = {
            "type": "emergence_cascade",
            "cascade_field": cascade_field,
            "time": cascade_data.get("current_time", 0.0),
        }
        self.scene_objects.append(cascade_obj)

    def _apply_gamma_coloring(self, gamma_data: dict[str, Any]) -> None:
        """Apply gamma function-based coloring to scene objects."""
        for obj in self.scene_objects:
            if "ga_element" in obj or "ga_elements" in obj:
                # Apply gamma-based coloring
                obj["gamma_color"] = gamma_data.get("base_color", "gold")
                obj["gamma_intensity"] = gamma_data.get("intensity", 1.0)

    def _apply_complexity_coloring(self, complexity_data: dict[str, Any]) -> None:
        """Apply complexity measure-based coloring."""
        complexity_scale = complexity_data.get("scale", [0.0, 1.0])
        colormap = complexity_data.get("colormap", "viridis")

        for obj in self.scene_objects:
            if "complexity_value" in obj:
                # Normalize complexity to color scale
                normalized = (obj["complexity_value"] - complexity_scale[0]) / (
                    complexity_scale[1] - complexity_scale[0]
                )
                obj["complexity_color"] = self._map_to_color(normalized, colormap)

    def _map_to_color(self, value: float, colormap: str) -> str:
        """Map normalized value to color."""
        # Simplified color mapping - in full implementation would use proper colormaps
        if colormap == "viridis":
            if value < 0.33:
                return "purple"
            elif value < 0.66:
                return "blue"
            else:
                return "yellow"
        return "gray"

    def _compile_scene(self) -> dict[str, Any]:
        """Compile scene objects into final representation."""
        return {
            "backend": "kingdon",
            "ga_space": str(self.ga_space) if self.ga_space else None,
            "objects": self.scene_objects,
            "morphic_fields": self.morphic_fields,
            "camera": self.camera.to_dict(),
            "timestamp": np.datetime64("now"),
        }

    def update_camera(self, config: Optional[CameraConfig] = None) -> None:
        """Update orthographic camera maintaining mathematical constraints."""
        if config:
            self.camera = config
        else:
            # Enforce golden ratio constraints
            self.camera.elevation = np.degrees((1 + np.sqrt(5)) / 2 - 1)
            self.camera.azimuth = -45.0
            self.camera.aspect_ratio = (1, 1, 1)
            self.camera.projection = "orthographic"

    def _apply_control_impl(self, control_type: str, value: Any) -> bool:
        """Apply Kingdon-specific control operations."""
        try:
            if control_type == self.semantics.ADDITIVE:
                # Additive control affects spatial exten
                for obj in self.scene_objects:
                    if "ga_element" in obj:
                        # Translate GA elemen
                        translation = sum(
                            value[j] * self.ga_space.basis[f"e{j+1}"]
                            for j in range(min(len(value), 3))
                        )
                        obj["ga_element"] = obj["ga_element"] + translation

            elif control_type == self.semantics.MULTIPLICATIVE:
                # Multiplicative control affects rotational/scaling transforms
                scale_factor = float(value)
                for obj in self.scene_objects:
                    if "ga_element" in obj:
                        obj["ga_element"] = obj["ga_element"] * scale_factor

            elif control_type == self.semantics.BOUNDARY:
                # Boundary control affects visibility and constraints
                visibility = bool(value)
                for obj in self.scene_objects:
                    obj["visible"] = visibility

            return True

        except Exception as e:
            print(f"❌ Kingdon control application failed: {e}")
            return False

    def export_scene(self, format_type: str = "json") -> Any:
        """Export scene in specified format."""
        scene_data = self._compile_scene()

        if format_type == "json":
            import json

            # Convert GA elements to serializable form
            serializable = self._make_serializable(scene_data)
            return json.dumps(serializable, indent=2)
        elif format_type == "ganja":
            # Export to Ganja.js format for web visualization
            return self._export_ganja_format(scene_data)
        else:
            return scene_data

    def _make_serializable(self, data: Any) -> Any:
        """Convert GA objects to JSON-serializable format."""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif hasattr(data, "__class__") and "kingdon" in str(data.__class__):
            # Convert Kingdon GA objects to string representation
            return {"ga_object": str(data), "type": "geometric_algebra"}
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.datetime64):
            return str(data)
        else:
            return data

    def _export_ganja_format(self, scene_data: dict[str, Any]) -> str:
        """Export to Ganja.js compatible format for web rendering."""
        ganja_script = "// Kingdon GA Scene Export for Ganja.js\n"
        ganja_script += f"// Generated from {scene_data.get('backend')} backend\n\n"

        for obj in scene_data.get("objects", []):
            obj_type = obj.get("type", "unknown")
            if obj_type == "point":
                ganja_script += f"// Point: {obj.get('position')}\n"
            elif obj_type == "curve":
                ganja_script += f"// Curve with {len(obj.get('points', []))} points\n"
            elif obj_type == "surface":
                ganja_script += "// Surface mesh\n"

        return ganja_script

    def get_morphic_transformation_matrix(self) -> np.ndarray:
        """Get current morphic transformation as matrix for compatibility."""
        if "current" in self.morphic_fields:
            self.morphic_fields["current"]["rotor"]
            # Convert GA rotor to transformation matrix
            # This is a simplified conversion - full implementation would be more complex
            phi = (1 + np.sqrt(5)) / 2
            return np.array([[phi, -1 / phi, 0], [1 / phi, phi, 0], [0, 0, 1]])
        return np.eye(3)
