#!/usr/bin/env python3
"""CLI formatting utilities - minimal and clean."""

from typing import Any

# Color scheme
COLORS = {
    "primary": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "info": "blue",
    "accent": "magenta",
}

# Icons for different states
ICONS = {
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "compute": "🔬",
    "analyze": "📊",
    "optimize": "⚡",
    "result": "📈",
}


def format_success(message: str) -> str:
    """Format success message."""
    return f"{ICONS['success']} {message}"


def format_batch_summary(results: list[Any]) -> str:
    """Format batch processing summary."""
    return f"Processed {len(results)} items successfully"
