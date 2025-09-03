#!/usr/bin/env python3
"""Input validation and sanitization for security."""

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .errors import ArraySizeError, InvalidDimensionError, MathematicalError


class InputValidator:
    """Centralized input validation for security and correctness."""

    # Maximum array size to prevent memory exhaustion attacks
    MAX_ARRAY_SIZE = 1_000_000

    # Valid dimension ranges for physical computations
    MIN_DIMENSION = -100.0
    MAX_DIMENSION = 1000.0

    @classmethod
    def sanitize_dimension(cls, d: Any, param_name: str = "dimension") -> NDArray[np.float64]:
        """
        Sanitize and validate dimensional input.

        Args:
            d: Input dimension(s)
            param_name: Parameter name for error messages

        Returns:
            Sanitized numpy array of dimensions

        Raises:
            ValueError: If input is invalid or dangerous
        """
        # Convert to numpy array
        try:
            d_array = np.asarray(d, dtype=np.float64)
        except (TypeError, ValueError):
            raise InvalidDimensionError(d, "Cannot convert to numeric array")

        # Check for size limits (prevent memory exhaustion)
        if d_array.size > cls.MAX_ARRAY_SIZE:
            raise ArraySizeError(d_array.size, cls.MAX_ARRAY_SIZE)

        # Check for NaN or Inf
        if np.any(~np.isfinite(d_array)):
            if np.any(np.isnan(d_array)):
                raise InvalidDimensionError(float('nan'))
            else:
                raise InvalidDimensionError(float('inf'))

        # Check reasonable bounds
        if np.any(d_array < cls.MIN_DIMENSION):
            min_val = float(np.min(d_array))
            raise InvalidDimensionError(min_val)
        if np.any(d_array > cls.MAX_DIMENSION):
            max_val = float(np.max(d_array))
            raise InvalidDimensionError(max_val)

        return d_array

    @classmethod
    def sanitize_array(cls, arr: Any, param_name: str = "array") -> NDArray[np.float64]:
        """
        Sanitize general array input.

        Args:
            arr: Input array
            param_name: Parameter name for error messages

        Returns:
            Sanitized numpy array

        Raises:
            ValueError: If input is invalid or dangerous
        """
        # Convert to numpy array
        try:
            array = np.asarray(arr, dtype=np.float64)
        except (TypeError, ValueError):
            raise InvalidDimensionError(arr, "Cannot convert to numeric array")

        # Check size limits
        if array.size > cls.MAX_ARRAY_SIZE:
            raise ValueError(
                f"Array too large: {array.size} elements exceeds maximum {cls.MAX_ARRAY_SIZE}"
            )

        # Check for NaN or Inf
        if np.any(~np.isfinite(array)):
            raise ValueError(f"Invalid {param_name}: contains NaN or Inf")

        return array

    @classmethod
    def sanitize_positive(cls, value: Any, param_name: str = "value") -> float:
        """
        Sanitize positive scalar input.

        Args:
            value: Input value
            param_name: Parameter name for error messages

        Returns:
            Sanitized positive float

        Raises:
            ValueError: If input is not positive
        """
        try:
            val = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid {param_name}: must be numeric") from e

        if not np.isfinite(val):
            raise ValueError(f"Invalid {param_name}: must be finite")

        if val <= 0:
            raise MathematicalError(param_name, val, "positive values only")

        return val

    @classmethod
    def sanitize_integer(cls, value: Any, param_name: str = "value", min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
        """
        Sanitize integer input with bounds checking.

        Args:
            value: Input value
            param_name: Parameter name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Sanitized integer

        Raises:
            ValueError: If input is invalid
        """
        try:
            val = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid {param_name}: must be integer") from e

        if min_val is not None and val < min_val:
            raise ValueError(f"Invalid {param_name}: must be >= {min_val}")

        if max_val is not None and val > max_val:
            raise ValueError(f"Invalid {param_name}: must be <= {max_val}")

        return val
