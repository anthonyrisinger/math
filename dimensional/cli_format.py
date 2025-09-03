#!/usr/bin/env python3
"""Consistent CLI formatting and beautiful output displays."""

from typing import Any, Optional

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table

console = Console()

# Consistent color scheme
COLORS = {
    "primary": "bold blue",
    "secondary": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "accent": "magenta",
    "value": "bright_yellow",
    "label": "bright_cyan",
    "dim": "dim white",
}

# Icons for different states
ICONS = {
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "rocket": "🚀",
    "sparkles": "✨",
    "lightning": "⚡",
    "chart": "📊",
    "science": "🔬",
    "math": "📐",
    "checkmark": "✓",
    "crossmark": "✗",
    "arrow_right": "→",
    "fire": "🔥",
}


def format_header(text: str, subtitle: str = "", icon: str = "rocket") -> Panel:
    """Create a beautiful header panel."""
    header_icon = ICONS.get(icon, "")
    content = f"[{COLORS['primary']}]{header_icon} {text}[/{COLORS['primary']}]"
    if subtitle:
        content += f"\n[{COLORS['dim']}]{subtitle}[/{COLORS['dim']}]"

    return Panel.fit(
        content,
        border_style=COLORS['primary'],
        box=box.DOUBLE,
    )


def format_result(label: str, value: Any, precision: int = 6, show_timing: Optional[float] = None) -> str:
    """Format a single result with consistent styling."""
    # Format the value based on type
    if isinstance(value, (int, float, np.number)):
        if isinstance(value, (int, np.integer)):
            value_str = f"{value}"
        else:
            value_str = f"{value:.{precision}f}"
    else:
        value_str = str(value)

    # Build the formatted string
    result = f"[{COLORS['label']}]{label}:[/{COLORS['label']}] [{COLORS['value']}]{value_str}[/{COLORS['value']}]"

    # Add timing if provided (in microseconds)
    if show_timing is not None:
        from .progress import MicrosecondTimer
        timer = MicrosecondTimer()
        timing_str = timer.format_time(show_timing)
        result += f" [{COLORS['dim']}]({timing_str})[/{COLORS['dim']}]"

    return result


def create_results_table(title: str, data: list[dict[str, Any]], show_timing: bool = False) -> Table:
    """Create a consistently formatted results table."""
    table = Table(
        title=f"[{COLORS['primary']}]{title}[/{COLORS['primary']}]",
        box=box.ROUNDED,
        title_style=Style(bold=True),
        header_style=Style(bold=True, color="bright_white"),
    )

    # Add columns based on data structure
    if data:
        for key in data[0].keys():
            if key == "timing" and not show_timing:
                continue

            # Determine column style and alignment
            if key in ["dimension", "name", "function"]:
                style = COLORS['secondary']
                justify = "left"
            elif key in ["value", "result"]:
                style = COLORS['value']
                justify = "right"
            elif key == "timing":
                style = COLORS['dim']
                justify = "right"
            else:
                style = None
                justify = "left"

            table.add_column(
                key.replace("_", " ").title(),
                style=style,
                justify=justify,
            )

        # Add rows
        for row_data in data:
            row = []
            for key in row_data.keys():
                if key == "timing" and not show_timing:
                    continue

                value = row_data[key]

                # Format based on type
                if isinstance(value, (float, np.number)) and not isinstance(value, (bool, np.bool_)):
                    if key == "timing":
                        from .progress import MicrosecondTimer
                        timer = MicrosecondTimer()
                        row.append(timer.format_time(value))
                    else:
                        row.append(f"{value:.6f}")
                else:
                    row.append(str(value))

            table.add_row(*row)

    return table


def format_success(message: str) -> str:
    """Format a success message."""
    return f"[{COLORS['success']}]{ICONS['checkmark']} {message}[/{COLORS['success']}]"


def format_error(message: str) -> str:
    """Format an error message."""
    return f"[{COLORS['error']}]{ICONS['crossmark']} {message}[/{COLORS['error']}]"


def format_warning(message: str) -> str:
    """Format a warning message."""
    return f"[{COLORS['warning']}]{ICONS['warning']} {message}[/{COLORS['warning']}]"


def format_info(message: str) -> str:
    """Format an info message."""
    return f"[{COLORS['secondary']}]{ICONS['info']} {message}[/{COLORS['secondary']}]"


def create_summary_panel(
    title: str,
    stats: dict[str, Any],
    style: str = "success",
    show_icon: bool = True
) -> Panel:
    """Create a summary panel with statistics."""
    icon = ICONS.get(style, "") if show_icon else ""

    content = []
    for key, value in stats.items():
        if isinstance(value, (float, np.number)):
            content.append(f"[{COLORS['label']}]{key}:[/{COLORS['label']}] [{COLORS['value']}]{value:.6f}[/{COLORS['value']}]")
        else:
            content.append(f"[{COLORS['label']}]{key}:[/{COLORS['label']}] {value}")

    panel_title = f"{icon} {title}" if icon else title

    return Panel(
        "\n".join(content),
        title=f"[{COLORS[style]}]{panel_title}[/{COLORS[style]}]",
        border_style=COLORS[style],
        box=box.ROUNDED,
    )


def format_dimension_range(start: float, end: float, steps: int) -> str:
    """Format a dimension range display."""
    return f"[{COLORS['value']}]{start:.3f}[/{COLORS['value']}] {ICONS['arrow_right']} [{COLORS['value']}]{end:.3f}[/{COLORS['value']}] ([{COLORS['dim']}]{steps} steps[/{COLORS['dim']}])"


def create_performance_badge(speedup: float) -> str:
    """Create a performance badge based on speedup factor."""
    if speedup >= 1000:
        icon = ICONS['fire']
        color = "bright_red"
        label = f"{speedup:.0f}x BLAZING"
    elif speedup >= 100:
        icon = ICONS['lightning']
        color = "bright_yellow"
        label = f"{speedup:.0f}x faster"
    elif speedup >= 10:
        icon = ICONS['rocket']
        color = "green"
        label = f"{speedup:.1f}x faster"
    else:
        icon = ICONS['checkmark']
        color = "cyan"
        label = f"{speedup:.1f}x"

    return f"[{color}]{icon} {label}[/{color}]"


def format_code_block(code: str, language: str = "python") -> Syntax:
    """Format a code block with syntax highlighting."""
    return Syntax(
        code,
        language,
        theme="monokai",
        line_numbers=True,
    )


def create_comparison_table(
    title: str,
    before: dict[str, float],
    after: dict[str, float]
) -> Table:
    """Create a before/after comparison table."""
    table = Table(
        title=f"[{COLORS['primary']}]{title}[/{COLORS['primary']}]",
        box=box.DOUBLE_EDGE,
    )

    table.add_column("Metric", style=COLORS['secondary'])
    table.add_column("Before", style=COLORS['dim'], justify="right")
    table.add_column("After", style=COLORS['value'], justify="right")
    table.add_column("Speedup", style=COLORS['success'], justify="center")

    for key in before.keys():
        if key in after:
            speedup = before[key] / after[key] if after[key] > 0 else float('inf')
            speedup_badge = create_performance_badge(speedup)

            table.add_row(
                key,
                f"{before[key]:.3f}",
                f"{after[key]:.3f}",
                speedup_badge
            )

    return table


def format_batch_summary(
    total_items: int,
    successful: int,
    failed: int,
    elapsed_time: float
) -> Panel:
    """Format a batch processing summary."""
    from .progress import MicrosecondTimer
    timer = MicrosecondTimer()

    success_rate = (successful / total_items * 100) if total_items > 0 else 0
    rate = (total_items / elapsed_time) * 1_000_000 if elapsed_time > 0 else 0

    if rate < 1000:
        rate_str = f"{rate:.0f} items/sec"
    elif rate < 1_000_000:
        rate_str = f"{rate/1000:.1f}K items/sec"
    else:
        rate_str = f"{rate/1_000_000:.1f}M items/sec"

    # Choose icon and color based on success rate
    if failed == 0:
        _ = ICONS['success']  # Mark as intentionally accessed
        style = "success"
        status = "All Successful"
    elif successful == 0:
        _ = ICONS['error']  # Mark as intentionally accessed
        style = "error"
        status = "All Failed"
    else:
        style = "warning"
        status = f"{success_rate:.1f}% Success"

    stats = {
        "Total Items": total_items,
        "Successful": f"{successful} ({success_rate:.1f}%)",
        "Failed": failed,
        "Processing Time": timer.format_time(elapsed_time),
        "Processing Rate": rate_str,
        "Status": status,
    }

    return create_summary_panel(
        "Batch Processing Complete",
        stats,
        style=style,
        show_icon=True
    )


def create_section_divider(text: str = "") -> str:
    """Create a section divider."""
    if text:
        return f"\n[{COLORS['dim']}]{'─' * 10} {text} {'─' * 10}[/{COLORS['dim']}]\n"
    else:
        return f"\n[{COLORS['dim']}]{'─' * 30}[/{COLORS['dim']}]\n"


def format_measurement(
    dimension: float,
    measures: dict[str, float],
    show_formulas: bool = False
) -> Panel:
    """Format dimensional measurements."""
    lines = [f"[{COLORS['primary']}]Dimension: {dimension:.3f}[/{COLORS['primary']}]"]
    lines.append("")

    for name, value in measures.items():
        lines.append(f"  [{COLORS['label']}]{name.title()}:[/{COLORS['label']}] [{COLORS['value']}]{value:.8f}[/{COLORS['value']}]")

        if show_formulas:
            if name == "volume":
                lines.append(f"    [{COLORS['dim']}]V(d) = π^(d/2) / Γ(d/2 + 1)[/{COLORS['dim']}]")
            elif name == "surface":
                lines.append(f"    [{COLORS['dim']}]S(d) = 2π^(d/2) / Γ(d/2)[/{COLORS['dim']}]")
            elif name == "complexity":
                lines.append(f"    [{COLORS['dim']}]C(d) = V(d) × S(d)[/{COLORS['dim']}]")

    return Panel(
        "\n".join(lines),
        title=f"{ICONS['math']} Measurements",
        border_style=COLORS['secondary'],
        box=box.ROUNDED,
    )


def show_welcome_banner() -> None:
    """Display a welcome banner."""
    console.print(
        Panel.fit(
            f"[{COLORS['primary']}]{ICONS['rocket']} Dimensional Mathematics Framework[/{COLORS['primary']}]\n"
            f"[{COLORS['secondary']}]Production-ready with blazing performance[/{COLORS['secondary']}]\n\n"
            f"[{COLORS['dim']}]Type 'dimensional --help' for available commands[/{COLORS['dim']}]",
            border_style=COLORS['primary'],
            box=box.DOUBLE,
        )
    )


# Export common formatting functions
__all__ = [
    "format_header",
    "format_result",
    "create_results_table",
    "format_success",
    "format_error",
    "format_warning",
    "format_info",
    "create_summary_panel",
    "format_dimension_range",
    "create_performance_badge",
    "format_code_block",
    "create_comparison_table",
    "format_batch_summary",
    "create_section_divider",
    "format_measurement",
    "show_welcome_banner",
    "COLORS",
    "ICONS",
    "console",
]
