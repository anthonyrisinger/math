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
    """Mathematical control semantics enforcement."""

    ADDITIVE = "additive"  # Extent/WHERE operations
    MULTIPLICATIVE = "multiplicative"  # Twist/WHAT operations
    BOUNDARY = "boundary"  # Edge/APS operations

    @classmethod
    def validate_control(cls, control_type: str, value: Any) -> bool:
        """Validate control operations against mathematical semantics."""
        if control_type == cls.ADDITIVE:
            # Additive controls affect spatial exten
            return isinstance(value, (int, float, np.ndarray))
        elif control_type == cls.MULTIPLICATIVE:
            # Multiplicative controls affect rotational/scaling transforms
            return isinstance(value, (int, float, np.ndarray)) and value != 0
        elif control_type == cls.BOUNDARY:
            # Boundary controls affect edge conditions and constraints
            return isinstance(value, (bool, int, float))
        return False


class VisualizationBackend(ABC):
    """Abstract base class for all visualization backends."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        self.camera = CameraConfig()
        self.semantics = ControlSemantics()
        self.scene_data = {}
        self._initialized = False

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

    @abstractmethod
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
        """Enforce orthographic projection requirements."""
        self.camera.projection = "orthographic"
        self.camera.aspect_ratio = (1, 1, 1)
        self.update_camera()

    def get_golden_ratio_view(self) -> tuple[float, float]:
        """Get canonical golden ratio viewing angles."""
        return (self.camera.elevation, self.camera.azimuth)

    def validate_mathematical_integrity(self, scene_data: dict[str, Any]) -> bool:
        """Validate that scene data maintains mathematical properties."""
        required_fields = ["geometry", "topology", "measures"]
        return all(field in scene_data for field in required_fields)

    @property
    def is_initialized(self) -> bool:
        return self._initialized
