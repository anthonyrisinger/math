#!/usr/bin/env python3
"""
Simplified Core Mathematical Types (without pydantic dependency)
=================================================================

Type-safe mathematical modeling using built-in Python types and numpy.
Provides runtime validation and mathematical semantic types without external dependencies.
"""

from typing import Union, Literal, Protocol, TypeVar, Generic, Any, Optional
import numpy as np
from abc import ABC, abstractmethod

# Core numeric types
Real = Union[float, np.floating]
Complex = Union[complex, np.complexfloating]
Numeric = Union[Real, Complex]
ArrayLike = Union[Numeric, np.ndarray]

# Type aliases for documentation
Dimension = float  # d ≥ 0
PositiveReal = float  # > 0
UnitInterval = float  # [0,1]
Volume = float  # V_d(r) ≥ 0
SurfaceArea = float  # S_d(r) ≥ 0
Complexity = float  # C(d)

T = TypeVar('T', bound=Numeric)


class DimensionalParameter:
    """Type-safe dimensional parameter with validation."""
    
    def __init__(self, value: float, tolerance: float = 1e-12):
        self.tolerance = tolerance
        if not np.isfinite(value):
            raise ValueError(f"Dimension must be finite, got {value}")
        if value < 0:
            raise ValueError(f"Dimension must be non-negative, got {value}")
        self.value = float(value)
    
    @property
    def is_integer(self) -> bool:
        """Check if dimension is an integer."""
        return abs(self.value - round(self.value)) < self.tolerance
    
    @property
    def is_critical(self) -> bool:
        """Check if dimension is near critical boundary."""
        from .constants import CRITICAL_DIMENSIONS
        return any(
            abs(self.value - critical) < self.tolerance
            for critical in CRITICAL_DIMENSIONS.values()
        )
    
    def __float__(self) -> float:
        return self.value
    
    def __repr__(self) -> str:
        return f"DimensionalParameter(value={self.value})"


class GammaArgument:
    """Type-safe gamma function argument with pole detection."""
    
    def __init__(self, value: Complex, tolerance: float = 1e-12):
        self.tolerance = tolerance
        self.value = complex(value)
        
        # Check for poles
        if np.isreal(self.value):
            real_part = float(np.real(self.value))
            if real_part <= 0 and abs(real_part - round(real_part)) < tolerance:
                # This is a pole, but we allow construction and flag it
                self._is_pole = True
            else:
                self._is_pole = False
        else:
            self._is_pole = False
    
    @property
    def is_pole(self) -> bool:
        """Check if argument is at a gamma function pole."""
        return self._is_pole
    
    def __complex__(self) -> complex:
        return self.value
    
    def __repr__(self) -> str:
        return f"GammaArgument(value={self.value}, is_pole={self.is_pole})"


class MeasureValue:
    """Type-safe measure value (volume, surface, complexity)."""
    
    def __init__(self, value: float, dimension: DimensionalParameter, measure_type: str):
        self.value = float(value)
        self.dimension = dimension
        self.measure_type = measure_type
    
    @property
    def is_peak(self) -> bool:
        """Check if this represents a peak value for the measure."""
        try:
            from .measures import find_peak
            peak_d, _ = find_peak(self.measure_type)
            return abs(self.dimension.value - peak_d) < self.dimension.tolerance
        except:
            return False
    
    def __float__(self) -> float:
        return self.value
    
    def __repr__(self) -> str:
        return f"MeasureValue(value={self.value}, dimension={self.dimension.value}, type={self.measure_type})"


class PhaseState:
    """Type-safe phase space state."""
    
    def __init__(self, dimension: DimensionalParameter, phase: float, 
                 sap_rate: float, energy: float, coherence: float):
        self.dimension = dimension
        self.phase = float(phase)
        self.sap_rate = float(sap_rate)
        self.energy = float(energy)
        
        # Validate coherence is in [0,1]
        if not 0 <= coherence <= 1:
            raise ValueError(f"Coherence must be in [0,1], got {coherence}")
        self.coherence = float(coherence)
    
    @property
    def is_emergent(self) -> bool:
        """Check if phase state indicates emergence."""
        return self.coherence > 0.5 and abs(self.sap_rate) < self.dimension.tolerance


class MorphicPolynomial:
    """Type-safe morphic polynomial configuration."""
    
    def __init__(self, tau: float, k: float):
        if tau <= 0:
            raise ValueError(f"Morphic parameter τ must be positive, got {tau}")
        self.tau = float(tau)
        self.k = float(k)
    
    @property
    def discriminant(self) -> float:
        """Compute polynomial discriminant."""
        from .morphic import discriminant
        return discriminant(self.tau)
    
    @property
    def is_stable(self) -> bool:
        """Check if polynomial parameters are in stable region."""
        return self.discriminant >= 0


# Function wrappers for type safety
class VolumeFunction:
    """Type-safe N-ball volume function."""
    
    def __call__(self, d: Union[float, DimensionalParameter]) -> MeasureValue:
        """Compute N-ball volume V_d."""
        if isinstance(d, DimensionalParameter):
            dim = d
        else:
            dim = DimensionalParameter(value=d)
        
        from .measures import ball_volume
        volume = ball_volume(dim.value)
        
        return MeasureValue(
            value=volume,
            dimension=dim,
            measure_type="volume"
        )


class SurfaceFunction:
    """Type-safe N-sphere surface function."""
    
    def __call__(self, d: Union[float, DimensionalParameter]) -> MeasureValue:
        """Compute N-sphere surface S_d."""
        if isinstance(d, DimensionalParameter):
            dim = d
        else:
            dim = DimensionalParameter(value=d)
        
        from .measures import sphere_surface
        surface = sphere_surface(dim.value)
        
        return MeasureValue(
            value=surface,
            dimension=dim,
            measure_type="surface"
        )


class ComplexityFunction:
    """Type-safe complexity measure function."""
    
    def __call__(self, d: Union[float, DimensionalParameter]) -> MeasureValue:
        """Compute complexity measure C(d)."""
        if isinstance(d, DimensionalParameter):
            dim = d
        else:
            dim = DimensionalParameter(value=d)
        
        from .measures import complexity_measure
        complexity = complexity_measure(dim.value)
        
        return MeasureValue(
            value=complexity,
            dimension=dim,
            measure_type="complexity"
        )


class GammaFunction:
    """Type-safe gamma function implementation."""
    
    def __call__(self, z: Union[Complex, GammaArgument]) -> Complex:
        """Evaluate gamma function with type safety."""
        if isinstance(z, GammaArgument):
            if z.is_pole:
                return complex(np.inf)
            arg = z.value
        else:
            # Validate argument
            arg_obj = GammaArgument(value=z)
            if arg_obj.is_pole:
                return complex(np.inf)
            arg = arg_obj.value
        
        from .gamma import gamma_safe
        return gamma_safe(arg)


# Singleton instances for convenient access
gamma_func = GammaFunction()
volume_func = VolumeFunction()
surface_func = SurfaceFunction()
complexity_func = ComplexityFunction()


# Export types for public API
__all__ = [
    # Core types
    "Real", "Complex", "Numeric", "ArrayLike",
    # Domain types
    "Dimension", "PositiveReal", "UnitInterval", 
    # Physical quantities
    "Volume", "SurfaceArea", "Complexity",
    # Value classes
    "DimensionalParameter", "GammaArgument", "MeasureValue", 
    "PhaseState", "MorphicPolynomial",
    # Function classes
    "VolumeFunction", "SurfaceFunction", "ComplexityFunction", "GammaFunction",
    # Singleton instances
    "gamma_func", "volume_func", "surface_func", "complexity_func",
]