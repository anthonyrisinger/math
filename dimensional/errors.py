#!/usr/bin/env python3
"""User-friendly error handling with helpful messages and examples."""

import sys
from difflib import get_close_matches
from typing import Any, Optional

import numpy as np


# Terminal colors for better visibility
class Colors:
    RED = '\033[91m' if sys.stdout.isatty() else ''
    YELLOW = '\033[93m' if sys.stdout.isatty() else ''
    GREEN = '\033[92m' if sys.stdout.isatty() else ''
    BLUE = '\033[94m' if sys.stdout.isatty() else ''
    RESET = '\033[0m' if sys.stdout.isatty() else ''
    BOLD = '\033[1m' if sys.stdout.isatty() else ''


class DimensionalError(Exception):
    """Base class for dimensional framework errors with helpful messages."""

    def __init__(self, message: str, suggestion: Optional[str] = None, example: Optional[str] = None):
        self.message = message
        self.suggestion = suggestion
        self.example = example
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with colors and helpful information."""
        parts = [f"{Colors.RED}{Colors.BOLD}Error:{Colors.RESET} {self.message}"]

        if self.suggestion:
            parts.append(f"\n{Colors.YELLOW}💡 Suggestion:{Colors.RESET} {self.suggestion}")

        if self.example:
            parts.append(f"\n{Colors.GREEN}✓ Example:{Colors.RESET} {self.example}")

        return "\n".join(parts)


class InvalidDimensionError(DimensionalError):
    """Raised when dimension value is invalid."""

    def __init__(self, value: Any, reason: str = ""):
        message = f"Invalid dimension value: {value}"

        # Provide specific guidance based on the value
        if isinstance(value, str):
            suggestion = "Dimensions must be numeric. Convert strings to numbers first."
            example = "Use: dimensional.ball_volume(4.5) or float('4.5')"
        elif value < 0:
            suggestion = f"Dimension {value} is negative. Physical dimensions are typically positive."
            example = f"Try: dimensional.ball_volume(abs({value})) for dimension {abs(value)}"
        elif value > 1000:
            suggestion = f"Dimension {value} is very large and may cause numerical issues."
            example = "Consider using dimensions < 100 for stable results"
        elif np.isnan(value):
            suggestion = "NaN (Not a Number) detected. Check your calculations."
            example = "Ensure all inputs are valid numbers"
        elif np.isinf(value):
            suggestion = "Infinite value detected. Use finite dimensions."
            example = "Try dimensions between 0.1 and 100"
        else:
            suggestion = "Ensure dimension is a positive real number"
            example = "Common values: 2 (2D), 3 (3D), 4.5 (fractional)"

        super().__init__(message, suggestion, example)


class ArraySizeError(DimensionalError):
    """Raised when array is too large for safe computation."""

    def __init__(self, size: int, max_size: int):
        message = f"Array size {size:,} exceeds maximum {max_size:,}"
        suggestion = (
            f"Process data in chunks of {max_size:,} or less.\n"
            f"  Or increase MAX_ARRAY_SIZE in validation.py if you have sufficient memory."
        )
        example = (
            f"# Process in chunks:\n"
            f"  chunk_size = {max_size}\n"
            f"  for i in range(0, {size}, chunk_size):\n"
            f"      chunk = data[i:i+chunk_size]\n"
            f"      result = ball_volume(chunk)"
        )
        super().__init__(message, suggestion, example)


class MathematicalError(DimensionalError):
    """Raised for mathematical domain errors."""

    def __init__(self, operation: str, value: Any, requirement: str):
        message = f"Mathematical error in {operation}"

        if "gamma" in operation.lower() and value <= 0:
            suggestion = f"Gamma function requires positive values, got {value}"
            example = "gamma_safe(5.5) or gamma_safe(abs(x)) for safety"
        elif "factorial" in operation.lower() and value < 0:
            suggestion = f"Factorial of {value} is undefined for negative integers"
            example = "factorial_extension(5) returns 120.0"
        elif "log" in operation.lower() and value <= 0:
            suggestion = f"Logarithm requires positive values, got {value}"
            example = "Use: np.log(abs(x) + 1e-10) to avoid log(0)"
        else:
            suggestion = f"Value {value} doesn't meet requirement: {requirement}"
            example = "Check mathematical constraints for this operation"

        super().__init__(message, suggestion, example)


class CLIUsageError(DimensionalError):
    """Raised for CLI usage errors with helpful command examples."""

    def __init__(self, command: str, available_commands: Optional[list[str]] = None):
        message = f"Unknown command or incorrect usage: '{command}'"

        # Try to find similar commands
        if available_commands:
            matches = get_close_matches(command, available_commands, n=3, cutoff=0.6)
            if matches:
                suggestion = f"Did you mean: {', '.join(matches)}?"
            else:
                suggestion = f"Available commands: {', '.join(available_commands[:5])}..."
        else:
            suggestion = "Use 'dimensional --help' to see all commands"

        example = (
            "Common commands:\n"
            "  dimensional measure --dim 4.5    # Calculate measures\n"
            "  dimensional peaks                # Find critical peaks\n"
            "  dimensional lab                  # Interactive exploration"
        )

        super().__init__(message, suggestion, example)


def helpful_error(error: Exception) -> DimensionalError:
    """Convert generic exceptions to helpful dimensional errors."""

    error_str = str(error)

    # Parse common error patterns
    if "cannot convert" in error_str.lower():
        return InvalidDimensionError(
            error_str.split("'")[1] if "'" in error_str else "input",
            "Type conversion failed"
        )

    elif "must be positive" in error_str.lower():
        return MathematicalError(
            "operation",
            "negative value",
            "positive values only"
        )

    elif "too large" in error_str.lower():
        return ArraySizeError(1000000, 100000)

    elif "nan" in error_str.lower() or "inf" in error_str.lower():
        return InvalidDimensionError(
            float('nan') if 'nan' in error_str.lower() else float('inf')
        )

    else:
        # Generic helpful error
        return DimensionalError(
            error_str,
            "Check your input values and try again",
            "Use 'dimensional --help' for usage examples"
        )


# Example usage demonstration
if __name__ == "__main__":
    print("Error Message Examples:\n")

    # Example 1: Invalid dimension
    try:
        raise InvalidDimensionError(-5)
    except DimensionalError as e:
        print(e)
        print()

    # Example 2: Array too large
    try:
        raise ArraySizeError(2_000_000, 1_000_000)
    except DimensionalError as e:
        print(e)
        print()

    # Example 3: Mathematical error
    try:
        raise MathematicalError("gamma_safe", -2, "positive values")
    except DimensionalError as e:
        print(e)
        print()

    # Example 4: CLI usage error
    try:
        commands = ["measure", "peaks", "lab", "demo", "plot"]
        raise CLIUsageError("measur", commands)
    except DimensionalError as e:
        print(e)
