#!/usr/bin/env python3
"""
Core Mathematical Types
========================

Type-safe mathematical modeling for the dimensional emergence framework.
Provides runtime validation, IDE support, and mathematical semantic types.

Key Features:
- Dimensional value validation (d ≥ 0, complex support)
- Gamma function domain checking (poles at negative integers)
- Measure types (volume, surface, complexity) with physical units
- Phase space coordinates with stability regions
- Morphic polynomial parameters with convergence bounds
"""

from typing import Union, Literal, Protocol, TypeVar, Generic, Annotated
import numpy as np
from pydantic import BaseModel, Field, field_validator
from abc import ABC, abstractmethod

# Core numeric types
Real = Union[float, np.floating]
Complex = Union[complex, np.complexfloating]
Numeric = Union[Real, Complex]
ArrayLike = Union[Numeric, np.ndarray]

# Mathematical domain constraints
Dimension = Annotated[float, Field(ge=0, description="Dimensional parameter d ≥ 0")]
PositiveReal = Annotated[float, Field(gt=0, description="Positive real number")]
UnitInterval = Annotated[float, Field(ge=0, le=1, description="Value in [0,1]")]
GoldenRatio = Annotated[float, Field(description="Golden ratio φ ≈ 1.618")]

# Physical quantities with units
Volume = Annotated[float, Field(ge=0, description="N-ball volume V_d(r)")]
SurfaceArea = Annotated[float, Field(ge=0, description="N-sphere surface S_d(r)")]
Complexity = Annotated[float, Field(description="Complexity measure C(d)")]

# Phase space types
PhaseCoordinate = Annotated[float, Field(description="Phase space coordinate")]
SappingRate = Annotated[float, Field(description="Phase sapping rate dφ/dt")]
EmergenceThreshold = Annotated[float, Field(ge=0, description="Emergence threshold")]

T = TypeVar('T', bound=Numeric)


class MathematicalValue(BaseModel, Generic[T]):
    """Base class for type-safe mathematical values with validation."""
    
    value: T
    precision: int = Field(default=15, ge=6, le=20)
    tolerance: float = Field(default=1e-12, ge=1e-16, le=1e-6)
    
    def __float__(self) -> float:
        """Convert to Python float."""
        return float(self.value)
    
    def __complex__(self) -> complex:
        """Convert to Python complex."""
        return complex(self.value)
    
    def __array__(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.asarray(self.value)


class DimensionalParameter(MathematicalValue[float]):
    """Type-safe dimensional parameter with validation."""
    
    value: Dimension
    
    @field_validator('value')
    @classmethod
    def validate_dimension(cls, v: float) -> float:
        """Ensure dimension is non-negative and finite."""
        if not np.isfinite(v):
            raise ValueError(f"Dimension must be finite, got {v}")
        if v < 0:
            raise ValueError(f"Dimension must be non-negative, got {v}")
        return v
    
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


class GammaArgument(MathematicalValue[Complex]):
    """Type-safe gamma function argument with pole detection."""
    
    value: Complex
    
    @field_validator('value')
    @classmethod
    def validate_gamma_domain(cls, v: Complex) -> Complex:
        """Check for gamma function poles (negative integers)."""
        if np.isreal(v):
            real_part = float(np.real(v))
            if real_part <= 0 and abs(real_part - round(real_part)) < 1e-12:
                raise ValueError(f"Gamma function pole at {v} (negative integer)")
        return v
    
    @property
    def is_pole(self) -> bool:
        """Check if argument is at a gamma function pole."""
        if not np.isreal(self.value):
            return False
        real_part = float(np.real(self.value))
        return real_part <= 0 and abs(real_part - round(real_part)) < self.tolerance


class MeasureValue(MathematicalValue[float]):
    """Type-safe measure value (volume, surface, complexity)."""
    
    value: float
    dimension: DimensionalParameter
    measure_type: Literal["volume", "surface", "complexity", "ratio", "phase_capacity"]
    
    @property
    def is_peak(self) -> bool:
        """Check if this represents a peak value for the measure."""
        # Implementation would check against known peak locations
        from .measures import find_peak
        try:
            peak_d, _ = find_peak(self.measure_type)
            return abs(self.dimension.value - peak_d) < self.tolerance
        except:
            return False


class PhaseState(BaseModel):
    """Type-safe phase space state."""
    
    dimension: DimensionalParameter
    phase: PhaseCoordinate
    sap_rate: SappingRate
    energy: float = Field(description="Total phase energy")
    coherence: UnitInterval = Field(description="Phase coherence [0,1]")
    
    @field_validator('coherence')
    @classmethod
    def validate_coherence(cls, v: float) -> float:
        """Ensure coherence is in unit interval."""
        if not 0 <= v <= 1:
            raise ValueError(f"Coherence must be in [0,1], got {v}")
        return v
    
    @property
    def is_emergent(self) -> bool:
        """Check if phase state indicates emergence."""
        return self.coherence > 0.5 and abs(self.sap_rate) < self.dimension.tolerance


class MorphicPolynomial(BaseModel):
    """Type-safe morphic polynomial configuration."""
    
    tau: PositiveReal = Field(description="Morphic parameter τ > 0")
    k: float = Field(description="Scaling parameter k")
    
    @field_validator('tau')
    @classmethod
    def validate_tau(cls, v: float) -> float:
        """Ensure tau is positive."""
        if v <= 0:
            raise ValueError(f"Morphic parameter τ must be positive, got {v}")
        return v
    
    @property
    def discriminant(self) -> float:
        """Compute polynomial discriminant."""
        from .morphic import discriminant
        return discriminant(self.tau)
    
    @property
    def is_stable(self) -> bool:
        """Check if polynomial parameters are in stable region."""
        return self.discriminant >= 0


# Protocol for mathematical functions
class MathematicalFunction(Protocol[T]):
    """Protocol for type-safe mathematical functions."""
    
    def __call__(self, x: T) -> T:
        """Function evaluation."""
        ...
    
    @property
    def domain(self) -> str:
        """Function domain description."""
        ...
    
    @property
    def codomain(self) -> str:
        """Function codomain description."""
        ...


class GammaFunction(MathematicalFunction[Complex]):
    """Type-safe gamma function implementation."""
    
    @property
    def domain(self) -> str:
        return "ℂ \\ {0, -1, -2, -3, ...}"
    
    @property
    def codomain(self) -> str:
        return "ℂ"
    
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


class MeasureFunction(MathematicalFunction[float]):
    """Base class for dimensional measure functions."""
    
    measure_type: str
    
    @property
    def domain(self) -> str:
        return "ℝ≥0"
    
    @property
    def codomain(self) -> str:
        return "ℝ≥0"
    
    @abstractmethod
    def __call__(self, d: Union[float, DimensionalParameter]) -> MeasureValue:
        """Evaluate measure function."""
        pass


class VolumeFunction(MeasureFunction):
    """Type-safe N-ball volume function."""
    
    measure_type = "volume"
    
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


class SurfaceFunction(MeasureFunction):
    """Type-safe N-sphere surface function."""
    
    measure_type = "surface"
    
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


class ComplexityFunction(MeasureFunction):
    """Type-safe complexity measure function."""
    
    measure_type = "complexity"
    
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
    "Dimension", "PositiveReal", "UnitInterval", "GoldenRatio",
    # Physical quantities
    "Volume", "SurfaceArea", "Complexity",
    # Phase space types
    "PhaseCoordinate", "SappingRate", "EmergenceThreshold",
    # Value classes
    "MathematicalValue", "DimensionalParameter", "GammaArgument",
    "MeasureValue", "PhaseState", "MorphicPolynomial",
    # Function protocols
    "MathematicalFunction", "GammaFunction", "MeasureFunction",
    # Concrete functions
    "VolumeFunction", "SurfaceFunction", "ComplexityFunction",
    # Singleton instances
    "gamma_func", "volume_func", "surface_func", "complexity_func",
]