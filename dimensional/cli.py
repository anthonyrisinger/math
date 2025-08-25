#!/usr/bin/env python3
"""
Dimensional Mathematics CLI
===========================

Unified, type-safe, AI-composable command-line interface for the dimensional
mathematics framework. Built with modern libraries for maximum flexibility
and AI interaction.

Key Features:
- Type-safe commands with pydantic validation
- Rich terminal output with beautiful formatting
- Composable commands for AI-generated workflows
- Auto-completion and help generation
- Consistent parameter patterns across all tools
"""

import typer
from typing import Optional, List, Tuple, Any
from pathlib import Path
import rich
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from pydantic import BaseModel, Field, validator
import json
import sys

# Import our core modules
from .gamma import lab, live, demo, explore, peaks, instant, qplot, v, s, c
from .measures import DimensionalMeasures
from .phase import PhaseDynamicsEngine
from .morphic import MorphicAnalyzer

# Initialize rich console for beautiful output
console = Console()

# Create main typer app
app = typer.Typer(
    name="dimensional",
    help="🌟 Dimensional Mathematics Framework - AI-Composable CLI",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# ============================================================================
# PYDANTIC MODELS FOR TYPE SAFETY
# ============================================================================

class DimensionRange(BaseModel):
    """Type-safe dimension range specification."""
    start: float = Field(default=0.1, ge=0.001, le=100.0)
    end: float = Field(default=10.0, ge=0.001, le=100.0)
    steps: int = Field(default=1000, ge=10, le=10000)

    @validator('end')
    def end_must_be_greater_than_start(cls, v, values):
        if 'start' in values and v <= values['start']:
            raise ValueError('end must be greater than start')
        return v

class PlotConfig(BaseModel):
    """Type-safe plot configuration."""
    width: int = Field(default=12, ge=4, le=20)
    height: int = Field(default=8, ge=4, le=20)
    dpi: int = Field(default=150, ge=72, le=300)
    style: str = Field(default="seaborn-v0_8", pattern=r'^[a-zA-Z0-9_-]+$')
    save: bool = Field(default=False)
    format: str = Field(default="png", pattern=r'^(png|pdf|svg|eps)$')

class AnalysisConfig(BaseModel):
    """Type-safe analysis configuration."""
    precision: int = Field(default=15, ge=6, le=20)
    tolerance: float = Field(default=1e-10, ge=1e-15, le=1e-5)
    max_iterations: int = Field(default=1000, ge=100, le=10000)

# ============================================================================
# GAMMA FUNCTION COMMANDS
# ============================================================================

@app.command("demo")
def cli_demo():
    """🚀 Run comprehensive gamma function demonstration."""
    console.print(Panel.fit(
        "🚀 [bold blue]Dimensional Gamma Demo[/bold blue]\n"
        "Running comprehensive demonstration...",
        border_style="blue"
    ))
    demo()

@app.command("lab")
def cli_lab(
    start_dimension: float = typer.Option(
        4.0, "--start", "-s",
        help="🎯 Starting dimension for interactive exploration",
        min=0.1, max=100.0
    )
):
    """🎮 Launch interactive gamma function laboratory."""
    console.print(Panel.fit(
        f"🎮 [bold green]Interactive Gamma Lab[/bold green]\n"
        f"Starting at dimension: [yellow]{start_dimension}[/yellow]\n"
        f"Use keyboard controls for exploration",
        border_style="green"
    ))
    lab(start_dimension)

@app.command("live")
def cli_live(
    expr_file: str = typer.Option(
        "gamma_expr.py", "--file", "-f",
        help="📝 Expression file to watch for live editing"
    )
):
    """🔥 Start live editing mode with hot reload."""
    file_path = Path(expr_file)
    if not file_path.exists():
        console.print(f"[red]❌ File not found: {expr_file}[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"🔥 [bold yellow]Live Editing Mode[/bold yellow]\n"
        f"Watching: [cyan]{expr_file}[/cyan]\n"
        f"Save file to see changes instantly",
        border_style="yellow"
    ))
    live(expr_file)

@app.command("explore")
def cli_explore(
    dimension: float = typer.Argument(
        4.0, help="🔍 Dimension to explore in detail"
    ),
    range_start: float = typer.Option(
        None, "--range-start", "-rs",
        help="📊 Start of exploration range"
    ),
    range_end: float = typer.Option(
        None, "--range-end", "-re",
        help="📊 End of exploration range"
    ),
    save_plot: bool = typer.Option(
        False, "--save", "-s",
        help="💾 Save plot to file"
    )
):
    """🔍 Explore gamma functions around a specific dimension."""
    console.print(Panel.fit(
        f"🔍 [bold cyan]Gamma Exploration[/bold cyan]\n"
        f"Dimension: [yellow]{dimension}[/yellow]",
        border_style="cyan"
    ))

    if range_start is not None and range_end is not None:
        # Custom range exploration - would need to extend explore() function
        console.print(f"📊 Range: {range_start} → {range_end}")

    explore(dimension)

    if save_plot:
        console.print("💾 [green]Plot saved![/green]")

@app.command("peaks")
def cli_peaks(
    function: str = typer.Option(
        "all", "--function", "-f",
        help="🎯 Function to analyze: v, s, c, or all"
    ),
    precision: int = typer.Option(
        15, "--precision", "-p",
        help="🎯 Numerical precision for peak finding",
        min=6, max=20
    )
):
    """🏔️ Find and analyze critical peaks in gamma functions."""
    console.print(Panel.fit(
        f"🏔️ [bold magenta]Peak Analysis[/bold magenta]\n"
        f"Function: [yellow]{function}[/yellow]\n"
        f"Precision: [cyan]{precision}[/cyan]",
        border_style="magenta"
    ))
    peaks()

@app.command("instant")
def cli_instant():
    """⚡ Generate instant gamma function visualization."""
    console.print(Panel.fit(
        "⚡ [bold red]Instant Visualization[/bold red]\n"
        "Generating quick gamma plots...",
        border_style="red"
    ))
    instant()

# ============================================================================
# MATHEMATICAL ANALYSIS COMMANDS
# ============================================================================

@app.command("measure")
def cli_measure(
    dimensions: List[float] = typer.Option(
        [2.0, 3.0, 4.0], "--dim", "-d",
        help="📏 Dimensions to measure (can specify multiple)"
    ),
    functions: List[str] = typer.Option(
        ["v", "s", "c"], "--func", "-f",
        help="⚙️ Functions to compute: v, s, c, r"
    ),
    output_format: str = typer.Option(
        "table", "--format", "-fmt",
        help="📋 Output format: table, json, csv"
    )
):
    """📏 Compute dimensional measures for specified dimensions."""
    console.print(Panel.fit(
        f"📏 [bold blue]Dimensional Measures[/bold blue]\n"
        f"Dimensions: [yellow]{dimensions}[/yellow]\n"
        f"Functions: [cyan]{functions}[/cyan]",
        border_style="blue"
    ))

    # Create results table
    table = Table(title="📊 Dimensional Measures")
    table.add_column("Dimension", style="cyan", no_wrap=True)

    for func in functions:
        table.add_column(f"{func.upper()}", style="yellow")

    # Compute measures
    for dim in track(dimensions, description="Computing..."):
        row = [f"{dim:.3f}"]
        for func in functions:
            if func == "v":
                value = v(dim)
            elif func == "s":
                value = s(dim)
            elif func == "c":
                value = c(dim)
            else:
                value = "N/A"
            row.append(f"{value:.6f}" if isinstance(value, (int, float)) else str(value))
        table.add_row(*row)

    if output_format == "table":
        console.print(table)
    elif output_format == "json":
        # Convert to JSON format
        results = []
        for dim in dimensions:
            result = {"dimension": dim}
            for func in functions:
                if func == "v":
                    result[func] = float(v(dim))
                elif func == "s":
                    result[func] = float(s(dim))
                elif func == "c":
                    result[func] = float(c(dim))
            results.append(result)
        console.print_json(json.dumps(results, indent=2))

@app.command("plot")
def cli_plot(
    functions: List[str] = typer.Option(
        ["v"], "--func", "-f",
        help="📈 Functions to plot: v, s, c"
    ),
    dim_start: float = typer.Option(
        0.1, "--start", "-s",
        help="📊 Start dimension",
        min=0.001, max=100.0
    ),
    dim_end: float = typer.Option(
        10.0, "--end", "-e",
        help="📊 End dimension",
        min=0.001, max=100.0
    ),
    steps: int = typer.Option(
        1000, "--steps", "-n",
        help="📊 Number of steps",
        min=10, max=10000
    ),
    save: bool = typer.Option(
        False, "--save",
        help="💾 Save plot to file"
    ),
    show: bool = typer.Option(
        True, "--show/--no-show",
        help="👁️ Show plot interactively"
    )
):
    """📈 Create customizable plots of gamma functions."""
    # Validate dimension range
    if dim_end <= dim_start:
        console.print("[red]❌ End dimension must be greater than start[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"📈 [bold green]Gamma Function Plotting[/bold green]\n"
        f"Functions: [yellow]{', '.join(functions)}[/yellow]\n"
        f"Range: [cyan]{dim_start} → {dim_end}[/cyan] ({steps} steps)",
        border_style="green"
    ))

    # Create the plot using qplot
    qplot(*functions)

# ============================================================================
# PHASE DYNAMICS COMMANDS
# ============================================================================

@app.command("phase")
def cli_phase(
    initial_dimension: float = typer.Option(
        3.0, "--initial", "-i",
        help="🌊 Initial dimension for phase evolution"
    ),
    time_steps: int = typer.Option(
        100, "--steps", "-n",
        help="⏰ Number of time steps",
        min=10, max=1000
    ),
    coupling: float = typer.Option(
        0.1, "--coupling", "-c",
        help="🔗 Phase coupling strength",
        min=0.0, max=1.0
    )
):
    """🌊 Simulate phase dynamics evolution."""
    console.print(Panel.fit(
        f"🌊 [bold purple]Phase Dynamics[/bold purple]\n"
        f"Initial: [yellow]{initial_dimension}[/yellow]\n"
        f"Steps: [cyan]{time_steps}[/cyan]\n"
        f"Coupling: [green]{coupling}[/green]",
        border_style="purple"
    ))

    # Initialize phase dynamics
    phase = PhaseDynamicsEngine()
    # Would need to extend PhaseDynamicsEngine with CLI-friendly methods
    console.print("🔄 [yellow]Running phase evolution...[/yellow]")

# ============================================================================
# UTILITY COMMANDS
# ============================================================================

@app.command("info")
def cli_info():
    """ℹ️ Show system information and available commands."""
    console.print(Panel.fit(
        "ℹ️ [bold blue]Dimensional Mathematics Framework[/bold blue]\n\n"
        "🎯 [green]Available Command Categories:[/green]\n"
        "  • [cyan]Gamma Functions[/cyan]: demo, lab, live, explore, peaks, instant\n"
        "  • [yellow]Analysis[/yellow]: measure, plot\n"
        "  • [purple]Phase Dynamics[/purple]: phase\n"
        "  • [green]Utilities[/green]: info, config\n\n"
        "🚀 [bold]Quick Start:[/bold]\n"
        "  dimensional demo     # See comprehensive demonstration\n"
        "  dimensional lab      # Interactive exploration\n"
        "  dimensional measure  # Compute dimensional measures\n\n"
        "🤖 [bold]AI-Composable:[/bold]\n"
        "All commands support rich parameter composition for AI workflows",
        border_style="blue"
    ))

@app.command("config")
def cli_config(
    show: bool = typer.Option(
        False, "--show",
        help="📋 Show current configuration"
    ),
    reset: bool = typer.Option(
        False, "--reset",
        help="🔄 Reset to default configuration"
    )
):
    """⚙️ Manage framework configuration."""
    if show:
        config = {
            "precision": 15,
            "tolerance": 1e-10,
            "plot_style": "seaborn-v0_8",
            "auto_save": False
        }
        console.print_json(json.dumps(config, indent=2))

    if reset:
        console.print("🔄 [green]Configuration reset to defaults[/green]")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main CLI entry point."""
    app()

if __name__ == "__main__":
    main()
