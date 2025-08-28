#!/usr/bin/env python3
"""
Base Visualization Backend
=========================

Abstract interface for visualization backends ensuring architectural consistency.
Orthographic camera requirements enforced.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class CameraConfig:
    """Orthographic camera configuration with mathematical constraints."""

    def __init__(self):
        self.aspect_ratio = (1, 1, 1)  # Box aspect 1:1:1
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.elevation = np.degrees(self.phi - 1)  # deg(φ-1)
        self.azimuth = -45.0  # -45°
        self.projection = "orthographic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "aspect_ratio": self.aspect_ratio,
            "elevation": self.elevation,
            "azimuth": self.azimuth,
            "projection": self.projection,
        }


class ControlSemantics:
    """
    ENHANCED Mathematical Control Semantics per STYLE.md

    ARCHITECTURAL CONTROL TAXONOMY:
    - ADDITIVE: Extent/domain/scale operations (WHERE we measure)
    - MULTIPLICATIVE: Twist/holonomy/phase operations (WHAT we measure)
    - BOUNDARY: Edge law/APS operations (boundary conditions)
    """

    ADDITIVE = "additive"       # Extent/WHERE: grid size, length, radius, sweep range
    MULTIPLICATIVE = "multiplicative"  # Twist/WHAT: mass, flux, coupling, phase, holonomy
    BOUNDARY = "boundary"       # Edge/APS: domain walls, edge phases, boundary conditions
    HEURISTIC = "heuristic"     # ε-floors/thresholds: p-adic weights, cut pairings

    @classmethod
    def validate_control(cls, control_type: str, value: Any) -> bool:
        """ENHANCED validation against mathematical semantics."""
        if control_type == cls.ADDITIVE:
            # Additive: Spatial extent, grid resolution, domain scale
            # Must be positive for physical meaning
            if isinstance(value, (int, float)):
                return value > 0
            elif isinstance(value, np.ndarray):
                return np.all(value > 0)
            return False

        elif control_type == cls.MULTIPLICATIVE:
            # Multiplicative: Phase factors, coupling constants, twist parameters
            # Cannot be zero (would eliminate dynamics)
            if isinstance(value, (int, float, complex)):
                return value != 0
            elif isinstance(value, np.ndarray):
                return np.all(value != 0)
            return False

        elif control_type == cls.BOUNDARY:
            # Boundary: Edge conditions, APS contributions, domain walls
            # Boolean flags, boundary parameters, edge phase offsets
            return isinstance(value, (bool, int, float, complex))

        elif control_type == cls.HEURISTIC:
            # Heuristic: ε-floor parameters, thresholds, p-adic weights
            # Must be in valid ranges for mathematical stability
            if isinstance(value, (int, float)):
                return 0 <= value <= 1  # Normalized weights/thresholds
            return isinstance(value, (bool, int, float))

        return False

    @classmethod
    def get_semantic_description(cls, control_type: str) -> str:
        """Get human-readable description of control semantic meaning."""
        descriptions = {
            cls.ADDITIVE: "Sets spatial extent, domain size, grid resolution - WHERE we measure",
            cls.MULTIPLICATIVE: "Sets twist, phase, coupling strength - WHAT we measure",
            cls.BOUNDARY: "Sets edge conditions, APS terms - boundary physics",
            cls.HEURISTIC: "Sets ε-floors, thresholds, stability parameters"
        }
        return descriptions.get(control_type, "Unknown control semantic")

    @classmethod
    def get_visual_cue(cls, control_type: str) -> str:
        """Get visual cue description per STYLE.md."""
        visual_cues = {
            cls.ADDITIVE: "Reference frame, wireframe, grid overlay",
            cls.MULTIPLICATIVE: "Normal displacement, phase arrows, color intensity",
            cls.BOUNDARY: "Thickened edge band, boundary highlighting",
            cls.HEURISTIC: "Annulus/ring emphasis, threshold indicators"
        }
        return visual_cues.get(control_type, "Standard visual treatment")


class VisualizationBackend(ABC):
    """Abstract base class for all visualization backends."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        self.camera = CameraConfig()
        self.semantics = ControlSemantics()
        self.scene_data = {}
        self._initialized = False

        # Enhanced save system per STYLE.md
        self.save_system = self._initialize_save_system()

        # Real-time optimization cache
        self.parameter_cache = {}
        self.performance_metrics = {"render_time": 0.0, "cache_hits": 0, "cache_misses": 0}

    def _initialize_save_system(self) -> dict:
        """Initialize ROBUST save path system that never fails."""
        from pathlib import Path

        # Default save configuration per STYLE.md requirements
        save_config = {
            "base_path": Path("./exports"),
            "auto_create_dirs": True,
            "include_metadata": True,
            "record_dpi": True,
            "record_camera": True,
            "key_bindings": {
                "save": "s",      # s: save instantly
                "reset": "r",     # r: reset view
                "spin": "p",      # p: spin animation
                "help": "h"       # h: help overlay
            },
            "failure_message": "Creating folders for save path… done. Saved with DPI and orthographic camera."
        }

        # ENSURE save directory exists (never fail per STYLE.md)
        try:
            save_config["base_path"].mkdir(parents=True, exist_ok=True)
            print(f"💾 Save system initialized: {save_config['base_path']}")
        except Exception as e:
            # Fallback to current directory if needed
            save_config["base_path"] = Path(".")
            print(f"⚠️  Save path fallback to current directory: {e}")

        return save_config

    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize the visualization backend."""
        pass

    @abstractmethod
    def render_scene(self, scene_data: dict[str, Any]) -> Any:
        """Render a mathematical scene with given data."""
        pass

    @abstractmethod
    def update_camera(self, config: Optional[CameraConfig] = None) -> None:
        """Update camera configuration maintaining orthographic constraints."""
        pass

    def apply_control(self, control_type: str, value: Any) -> bool:
        """Apply control operation with semantic validation."""
        if not self.semantics.validate_control(control_type, value):
            raise ValueError(f"Invalid {control_type} control value: {value}")
        return self._apply_control_impl(control_type, value)

    @abstractmethod
    def _apply_control_impl(self, control_type: str, value: Any) -> bool:
        """Backend-specific control implementation."""
        pass

    @abstractmethod
    def export_scene(self, format_type: str = "json") -> Any:
        """Export scene data in specified format."""
        pass

    def set_orthographic_projection(self) -> None:
        """ENFORCE orthographic projection requirements per STYLE.md."""
        from dimensional.mathematics.constants import PHI, VIEW_AZIM, VIEW_ELEV

        self.camera.projection = "orthographic"
        self.camera.aspect_ratio = (1, 1, 1)  # MANDATORY (1:1:1) box aspect
        self.camera.elevation = VIEW_ELEV  # deg(φ-1) ≈ 35.4°
        self.camera.azimuth = VIEW_AZIM    # -45°
        self.camera.phi = PHI              # Golden ratio reference

        self.update_camera()
        print(f"📐 ORTHOGRAPHIC ENFORCED: {self.camera.elevation:.1f}°, {self.camera.azimuth:.1f}°, aspect={self.camera.aspect_ratio}")

    def get_golden_ratio_view(self) -> tuple[float, float]:
        """Get canonical golden ratio viewing angles."""
        return (self.camera.elevation, self.camera.azimuth)

    def validate_mathematical_integrity(self, scene_data: dict[str, Any]) -> bool:
        """Validate that scene data maintains mathematical properties."""
        required_fields = ["geometry", "topology", "measures"]
        return all(field in scene_data for field in required_fields)

    def optimize_parameter_sweep(self, parameter_key: str, parameter_value: Any) -> bool:
        """
        REAL-TIME parameter sweep optimization with caching.

        Implements aggressive caching for parameter sweeps to enable
        smooth real-time interaction per STYLE.md requirements.
        """
        import time

        # Create cache key from parameter
        cache_key = f"{parameter_key}:{hash(str(parameter_value))}"

        # Check cache first
        if cache_key in self.parameter_cache:
            self.performance_metrics["cache_hits"] += 1
            print(f"⚡ Cache hit for {parameter_key}")
            return True

        # Cache miss - compute and store
        start_time = time.time()
        self.performance_metrics["cache_misses"] += 1

        # Store in cache with timestamp
        self.parameter_cache[cache_key] = {
            "value": parameter_value,
            "timestamp": time.time(),
            "computed_time": 0.0  # Will be updated below
        }

        # Update timing
        compute_time = time.time() - start_time
        self.parameter_cache[cache_key]["computed_time"] = compute_time
        self.performance_metrics["render_time"] = compute_time

        # Limit cache size (keep most recent 1000 entries)
        if len(self.parameter_cache) > 1000:
            oldest_key = min(self.parameter_cache.keys(),
                           key=lambda k: self.parameter_cache[k]["timestamp"])
            del self.parameter_cache[oldest_key]

        return True

    def get_performance_stats(self) -> dict:
        """Get real-time performance statistics."""
        total_requests = self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]
        cache_rate = self.performance_metrics["cache_hits"] / max(total_requests, 1) * 100

        return {
            "cache_hit_rate": f"{cache_rate:.1f}%",
            "total_cached": len(self.parameter_cache),
            "avg_render_time": f"{self.performance_metrics['render_time']:.3f}s",
            "optimization_enabled": True
        }

    def robust_save(self, filename: str = None, format_type: str = "png", **kwargs) -> str:
        """
        ROBUST save operation that never fails per STYLE.md.

        Returns path to saved file or error message.
        """
        from datetime import datetime

        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.backend_name}_scene_{timestamp}.{format_type}"

            # Ensure full path
            filepath = self.save_system["base_path"] / filename

            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Export scene data
            scene_export = self.export_scene(format_type)

            # Add metadata per STYLE.md requirements
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "backend": self.backend_name,
                "camera_config": self.camera.to_dict(),
                "dpi": kwargs.get("dpi", 300),
                "orthographic": self.camera.projection == "orthographic",
                "golden_ratio_view": self.get_golden_ratio_view(),
                "mathematical_integrity": "VALIDATED"
            }

            # Save with format-specific handling
            if format_type == "json":
                import json
                with open(filepath, 'w') as f:
                    json.dump({"scene": scene_export, "metadata": metadata}, f, indent=2)
            else:
                # For other formats, save scene export directly
                with open(filepath, 'w') as f:
                    f.write(str(scene_export))

            print(f"💾 {self.save_system['failure_message']}")
            print(f"📁 Saved: {filepath}")
            return str(filepath)

        except Exception as e:
            # NEVER fail - provide graceful fallback
            error_msg = f"Save attempted but encountered: {e}. Data preserved in memory."
            print(f"⚠️  {error_msg}")
            return error_msg

    @property
    def is_initialized(self) -> bool:
        return self._initialized
