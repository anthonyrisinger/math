#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Visualization Backends
===========================================================

Tests Sprint 3 visualization enhancements including:
- Golden ratio view angle enforcement
- Box aspect ratio compliance
- Control semantics validation
- Real-time parameter sweep optimization
- Robust save path creation
- Mathematical integrity validation

GOAL: Expand test coverage from 60% → 75%
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from visualization.backends.base_backend import (
    CameraConfig,
    ControlSemantics,
    VisualizationBackend,
)

# Import specific backends with graceful fallback
try:
    from visualization.backends.plotly_backend import PlotlyDashboard
    PLOTLY_AVAILABLE = True
except ImportError:
    PlotlyDashboard = None
    PLOTLY_AVAILABLE = False

try:
    from visualization.backends.kingdon_backend import KingdonRenderer
    KINGDON_AVAILABLE = True
except ImportError:
    KingdonRenderer = None
    KINGDON_AVAILABLE = False


class TestCameraConfig:
    """Test camera configuration with golden ratio constraints."""

    def test_camera_initialization(self):
        """Test camera initializes with correct golden ratio parameters."""
        camera = CameraConfig()

        # Verify golden ratio calculation
        expected_phi = (1 + np.sqrt(5)) / 2
        assert abs(camera.phi - expected_phi) < 1e-10

        # Verify viewing angles
        expected_elevation = np.degrees(expected_phi - 1)
        assert abs(camera.elevation - expected_elevation) < 1e-6
        assert camera.azimuth == -45.0

        # Verify aspect ratio
        assert camera.aspect_ratio == (1, 1, 1)
        assert camera.projection == "orthographic"

    def test_camera_to_dict(self):
        """Test camera dictionary conversion."""
        camera = CameraConfig()
        camera_dict = camera.to_dict()

        required_keys = ["aspect_ratio", "elevation", "azimuth", "projection"]
        assert all(key in camera_dict for key in required_keys)

        # Verify mathematical values are preserved
        assert camera_dict["aspect_ratio"] == (1, 1, 1)
        assert abs(camera_dict["elevation"] - np.degrees((1 + np.sqrt(5)) / 2 - 1)) < 1e-6
        assert camera_dict["azimuth"] == -45.0
        assert camera_dict["projection"] == "orthographic"


class TestControlSemantics:
    """Test enhanced control semantics per STYLE.md requirements."""

    def test_control_types_defined(self):
        """Test all required control types are defined."""
        assert hasattr(ControlSemantics, 'ADDITIVE')
        assert hasattr(ControlSemantics, 'MULTIPLICATIVE')
        assert hasattr(ControlSemantics, 'BOUNDARY')
        assert hasattr(ControlSemantics, 'HEURISTIC')

        # Verify string values
        assert ControlSemantics.ADDITIVE == "additive"
        assert ControlSemantics.MULTIPLICATIVE == "multiplicative"
        assert ControlSemantics.BOUNDARY == "boundary"
        assert ControlSemantics.HEURISTIC == "heuristic"

    def test_additive_control_validation(self):
        """Test additive control validation (extent/WHERE operations)."""
        # Valid additive values (must be positive)
        assert ControlSemantics.validate_control("additive", 1.0)
        assert ControlSemantics.validate_control("additive", 100)
        assert ControlSemantics.validate_control("additive", np.array([1, 2, 3]))

        # Invalid additive values (negative or zero)
        assert not ControlSemantics.validate_control("additive", 0.0)
        assert not ControlSemantics.validate_control("additive", -1.0)
        assert not ControlSemantics.validate_control("additive", np.array([1, -2, 3]))
        assert not ControlSemantics.validate_control("additive", "invalid")

    def test_multiplicative_control_validation(self):
        """Test multiplicative control validation (twist/WHAT operations)."""
        # Valid multiplicative values (cannot be zero)
        assert ControlSemantics.validate_control("multiplicative", 1.0)
        assert ControlSemantics.validate_control("multiplicative", -1.0)
        assert ControlSemantics.validate_control("multiplicative", 1+2j)
        assert ControlSemantics.validate_control("multiplicative", np.array([1, 2, 3]))

        # Invalid multiplicative values (zero eliminates dynamics)
        assert not ControlSemantics.validate_control("multiplicative", 0.0)
        assert not ControlSemantics.validate_control("multiplicative", 0+0j)
        assert not ControlSemantics.validate_control("multiplicative", np.array([1, 0, 3]))
        assert not ControlSemantics.validate_control("multiplicative", "invalid")

    def test_boundary_control_validation(self):
        """Test boundary control validation (edge/APS operations)."""
        # Valid boundary values (boolean, numeric parameters)
        assert ControlSemantics.validate_control("boundary", True)
        assert ControlSemantics.validate_control("boundary", False)
        assert ControlSemantics.validate_control("boundary", 1.0)
        assert ControlSemantics.validate_control("boundary", -1)
        assert ControlSemantics.validate_control("boundary", 1+2j)

        # Invalid boundary values
        assert not ControlSemantics.validate_control("boundary", "invalid")
        assert not ControlSemantics.validate_control("boundary", [1, 2, 3])

    def test_heuristic_control_validation(self):
        """Test heuristic control validation (ε-floor/threshold operations)."""
        # Valid heuristic values (normalized weights/thresholds)
        assert ControlSemantics.validate_control("heuristic", 0.0)
        assert ControlSemantics.validate_control("heuristic", 0.5)
        assert ControlSemantics.validate_control("heuristic", 1.0)
        assert ControlSemantics.validate_control("heuristic", True)

        # Invalid heuristic values (outside normalized range)
        assert not ControlSemantics.validate_control("heuristic", -0.1)
        assert not ControlSemantics.validate_control("heuristic", 1.1)
        assert not ControlSemantics.validate_control("heuristic", "invalid")

    def test_semantic_descriptions(self):
        """Test semantic descriptions are provided for all control types."""
        descriptions = [
            ControlSemantics.get_semantic_description("additive"),
            ControlSemantics.get_semantic_description("multiplicative"),
            ControlSemantics.get_semantic_description("boundary"),
            ControlSemantics.get_semantic_description("heuristic")
        ]

        # All descriptions should be non-empty strings
        for desc in descriptions:
            assert isinstance(desc, str)
            assert len(desc) > 0
            assert "WHERE" in desc or "WHAT" in desc or "boundary" in desc or "floor" in desc

    def test_visual_cues(self):
        """Test visual cues are defined for all control types."""
        visual_cues = [
            ControlSemantics.get_visual_cue("additive"),
            ControlSemantics.get_visual_cue("multiplicative"),
            ControlSemantics.get_visual_cue("boundary"),
            ControlSemantics.get_visual_cue("heuristic")
        ]

        # All visual cues should be non-empty strings
        for cue in visual_cues:
            assert isinstance(cue, str)
            assert len(cue) > 0


class MockVisualizationBackend(VisualizationBackend):
    """Mock backend for testing base class functionality."""

    def initialize(self, **kwargs) -> bool:
        self._initialized = True
        return True

    def render_scene(self, scene_data):
        return {"mock_render": scene_data}

    def update_camera(self, config=None):
        if config:
            self.camera = config

    def _apply_control_impl(self, control_type: str, value) -> bool:
        return True

    def export_scene(self, format_type: str = "json"):
        return {"format": format_type, "backend": self.backend_name}


class TestVisualizationBackendBase:
    """Test base visualization backend functionality."""

    def setup_method(self):
        """Set up test backend."""
        self.backend = MockVisualizationBackend("test_backend")

    def test_initialization(self):
        """Test backend initialization."""
        assert self.backend.backend_name == "test_backend"
        assert isinstance(self.backend.camera, CameraConfig)
        assert isinstance(self.backend.semantics, ControlSemantics)
        assert hasattr(self.backend, 'save_system')
        assert hasattr(self.backend, 'parameter_cache')
        assert hasattr(self.backend, 'performance_metrics')

    def test_save_system_initialization(self):
        """Test save system is properly initialized."""
        save_system = self.backend.save_system

        assert "base_path" in save_system
        assert "auto_create_dirs" in save_system
        assert "key_bindings" in save_system

        # Verify key bindings per STYLE.md
        bindings = save_system["key_bindings"]
        assert bindings["save"] == "s"
        assert bindings["reset"] == "r"
        assert bindings["spin"] == "p"
        assert bindings["help"] == "h"

    def test_orthographic_projection_enforcement(self):
        """Test orthographic projection is properly enforced."""
        self.backend.set_orthographic_projection()

        # Verify all STYLE.md requirements are enforced
        assert self.backend.camera.projection == "orthographic"
        assert self.backend.camera.aspect_ratio == (1, 1, 1)

        # Verify golden ratio angles
        from dimensional.mathematics.constants import PHI
        expected_elev = np.degrees(PHI - 1)
        assert abs(self.backend.camera.elevation - expected_elev) < 1e-6
        assert self.backend.camera.azimuth == -45.0
        assert abs(self.backend.camera.phi - PHI) < 1e-10

    def test_golden_ratio_view_retrieval(self):
        """Test golden ratio viewing angles can be retrieved."""
        elev, azim = self.backend.get_golden_ratio_view()

        from dimensional.mathematics.constants import PHI
        expected_elev = np.degrees(PHI - 1)
        assert abs(elev - expected_elev) < 1e-6
        assert azim == -45.0

    def test_mathematical_integrity_validation(self):
        """Test mathematical integrity validation."""
        # Valid scene data
        valid_scene = {
            "geometry": {"points": []},
            "topology": {"invariants": {}},
            "measures": {"gamma": {}}
        }
        assert self.backend.validate_mathematical_integrity(valid_scene)

        # Invalid scene data (missing required fields)
        invalid_scene = {"geometry": {"points": []}}
        assert not self.backend.validate_mathematical_integrity(invalid_scene)

    def test_parameter_sweep_optimization(self):
        """Test real-time parameter sweep optimization."""
        # First access should be cache miss
        result1 = self.backend.optimize_parameter_sweep("test_param", 1.0)
        assert result1
        assert self.backend.performance_metrics["cache_misses"] == 1
        assert self.backend.performance_metrics["cache_hits"] == 0

        # Second access should be cache hit
        result2 = self.backend.optimize_parameter_sweep("test_param", 1.0)
        assert result2
        assert self.backend.performance_metrics["cache_hits"] == 1

    def test_performance_statistics(self):
        """Test performance statistics tracking."""
        # Generate some cache activity
        self.backend.optimize_parameter_sweep("param1", 1.0)
        self.backend.optimize_parameter_sweep("param2", 2.0)
        self.backend.optimize_parameter_sweep("param1", 1.0)  # Should hit cache

        stats = self.backend.get_performance_stats()

        assert "cache_hit_rate" in stats
        assert "total_cached" in stats
        assert "avg_render_time" in stats
        assert "optimization_enabled" in stats
        assert stats["optimization_enabled"] is True

    def test_robust_save_operation(self):
        """Test robust save operation that never fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override save path for testing
            self.backend.save_system["base_path"] = Path(temp_dir)

            # Test successful save
            result = self.backend.robust_save("test_scene.json", "json")
            assert temp_dir in result  # Should return file path

            # Verify file was created
            saved_file = Path(temp_dir) / "test_scene.json"
            assert saved_file.exists()

            # Verify content includes metadata
            with open(saved_file) as f:
                data = json.load(f)
                assert "scene" in data
                assert "metadata" in data
                assert data["metadata"]["backend"] == "test_backend"
                assert data["metadata"]["orthographic"] is True

    @patch('builtins.open', side_effect=PermissionError("Access denied"))
    def test_robust_save_failure_handling(self, mock_open):
        """Test save operation handles failures gracefully."""
        result = self.backend.robust_save("test_scene.json", "json")

        # Should return error message, not raise exception
        assert "Save attempted but encountered" in result
        assert "Access denied" in result

    def test_control_semantic_validation_integration(self):
        """Test control operations integrate with semantic validation."""
        # Test valid additive control
        result = self.backend.apply_control("additive", 5.0)
        assert result is True

        # Test invalid additive control (negative value)
        with pytest.raises(ValueError) as exc_info:
            self.backend.apply_control("additive", -1.0)
        assert "Invalid additive control value" in str(exc_info.value)

        # Test valid multiplicative control
        result = self.backend.apply_control("multiplicative", 2.0)
        assert result is True

        # Test invalid multiplicative control (zero value)
        with pytest.raises(ValueError) as exc_info:
            self.backend.apply_control("multiplicative", 0.0)
        assert "Invalid multiplicative control value" in str(exc_info.value)


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestPlotlyDashboardIntegration:
    """Test Plotly dashboard backend integration."""

    def setup_method(self):
        """Set up Plotly dashboard for testing."""
        self.dashboard = PlotlyDashboard()

    def test_plotly_initialization(self):
        """Test Plotly dashboard initializes correctly."""
        result = self.dashboard.initialize(layout="mathematical_grid")

        if result:  # Only test if initialization succeeded
            assert self.dashboard.is_initialized
            assert self.dashboard.backend_name == "plotly"
            assert self.dashboard.fig is not None

    def test_mathematical_grid_layout(self):
        """Test mathematical grid layout creation."""
        if self.dashboard.initialize(layout="mathematical_grid"):
            assert self.dashboard.subplot_specs is not None
            assert len(self.dashboard.subplot_specs) == 2  # 2 rows
            assert len(self.dashboard.subplot_specs[0]) == 3  # 3 columns

    def test_orthographic_camera_application(self):
        """Test orthographic camera is applied to 3D scenes."""
        if self.dashboard.initialize():
            self.dashboard.set_orthographic_projection()

            # Verify golden ratio angles are set
            elev, azim = self.dashboard.get_golden_ratio_view()
            from dimensional.mathematics.constants import PHI
            expected_elev = np.degrees(PHI - 1)
            assert abs(elev - expected_elev) < 1e-6
            assert azim == -45.0

    def test_enhanced_scene_rendering(self):
        """Test enhanced scene rendering with mathematical integrity."""
        if not self.dashboard.initialize():
            pytest.skip("Plotly initialization failed")

        # Create valid scene data
        scene_data = {
            "geometry": {"landscape": {"d_range": np.linspace(0.1, 12, 10)}},
            "topology": {"cascade": {"dimensions": [1, 2, 3, 4]}},
            "measures": {"volumes": [1, 2, 3, 4]},
        }

        result = self.dashboard.render_scene(scene_data)
        assert result is not None


@pytest.mark.skipif(not KINGDON_AVAILABLE, reason="Kingdon not installed")
class TestKingdonRendererIntegration:
    """Test Kingdon geometric algebra backend integration."""

    def setup_method(self):
        """Set up Kingdon renderer for testing."""
        self.renderer = KingdonRenderer()

    def test_kingdon_initialization(self):
        """Test Kingdon renderer initializes correctly."""
        result = self.renderer.initialize(dimension=3, signature=(3, 0, 1))

        if result:  # Only test if initialization succeeded
            assert self.renderer.is_initialized
            assert self.renderer.backend_name == "kingdon"

    def test_geometric_algebra_scene_handling(self):
        """Test geometric algebra scene data handling."""
        if not self.renderer.initialize():
            pytest.skip("Kingdon initialization failed")

        # Create GA scene data
        scene_data = {
            "geometry": {
                "points": {"positions": [[1, 0, 0], [0, 1, 0]]},
                "morphic_field": {"phase_angle": np.pi/4, "axis": [0, 0, 1]}
            },
            "topology": {
                "phase_space": {"trajectories": [{"time": [0, 1], "states": [[1, 0], [0, 1]]}]}
            },
            "measures": {"gamma_measure": {"base_color": "gold"}}
        }

        result = self.renderer.render_scene(scene_data)
        assert result is not None
        assert "backend" in result
        assert result["backend"] == "kingdon"

    def test_morphic_field_transformation(self):
        """Test morphic field transformations with golden ratio."""
        if not self.renderer.initialize():
            pytest.skip("Kingdon initialization failed")

        # Apply morphic transformation
        morphic_data = {
            "phase_angle": np.pi / 4,
            "axis": [0, 0, 1],
            "field_points": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        }

        self.renderer._render_morphic_field(morphic_data)

        # Verify morphic field was created
        assert "current" in self.renderer.morphic_fields
        field = self.renderer.morphic_fields["current"]
        assert "rotor" in field
        assert "phi_scaling" in field
        assert abs(field["phi_scaling"] - (1 + np.sqrt(5)) / 2) < 1e-10


class TestVisualizationSystemIntegration:
    """Test complete visualization system integration."""

    def test_backend_interoperability(self):
        """Test different backends can work together."""
        backends = []

        # Add available backends
        backends.append(MockVisualizationBackend("mock"))

        if PLOTLY_AVAILABLE:
            backends.append(PlotlyDashboard())

        if KINGDON_AVAILABLE:
            backends.append(KingdonRenderer())

        # Test all backends follow same interface
        for backend in backends:
            assert hasattr(backend, 'initialize')
            assert hasattr(backend, 'render_scene')
            assert hasattr(backend, 'export_scene')
            assert hasattr(backend, 'set_orthographic_projection')
            assert hasattr(backend, 'get_golden_ratio_view')

    def test_mathematical_consistency_across_backends(self):
        """Test mathematical consistency across all backends."""
        backends = [MockVisualizationBackend("mock")]

        if PLOTLY_AVAILABLE:
            backends.append(PlotlyDashboard())

        # Test golden ratio consistency
        for backend in backends:
            backend.set_orthographic_projection()
            elev, azim = backend.get_golden_ratio_view()

            from dimensional.mathematics.constants import PHI
            expected_elev = np.degrees(PHI - 1)
            assert abs(elev - expected_elev) < 1e-6
            assert azim == -45.0

    def test_performance_optimization_effectiveness(self):
        """Test performance optimization is effective across backends."""
        backend = MockVisualizationBackend("performance_test")

        # Simulate parameter sweep
        parameters = [("dimension", d) for d in np.linspace(1, 10, 20)]

        # First pass - all cache misses
        for param_name, param_value in parameters:
            backend.optimize_parameter_sweep(param_name, param_value)

        # Second pass - should have cache hits
        # initial_misses = backend.performance_metrics["cache_misses"]  # Unused variable
        for param_name, param_value in parameters:
            backend.optimize_parameter_sweep(param_name, param_value)

        # Verify cache effectiveness
        stats = backend.get_performance_stats()
        cache_hit_rate = float(stats["cache_hit_rate"].rstrip('%'))
        assert cache_hit_rate > 40.0  # At least 40% cache hit rate


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])
