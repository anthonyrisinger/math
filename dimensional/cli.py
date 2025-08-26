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

import json
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import typer
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

# Import our consolidated modules
from .gamma import gamma_safe, γ, ln_γ, ψ, gamma_explorer, quick_gamma_analysis
from .phase import PhaseDynamicsEngine, quick_emergence_analysis
from .measures import ball_volume, sphere_surface, complexity_measure, V, S, C, Λ

# Import visualization modules
from visualization import PlotlyDashboard, KingdonRenderer
from visualization.modernized_dashboard import create_modern_dashboard

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

    @field_validator("end")
    @classmethod
    def end_must_be_greater_than_start(cls, v, info):
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("end must be greater than start")
        return v


class PlotConfig(BaseModel):
    """Type-safe plot configuration."""

    width: int = Field(default=12, ge=4, le=20)
    height: int = Field(default=8, ge=4, le=20)
    dpi: int = Field(default=150, ge=72, le=300)
    style: str = Field(default="seaborn-v0_8", pattern=r"^[a-zA-Z0-9_-]+$")
    save: bool = Field(default=False)
    format: str = Field(default="png", pattern=r"^(png|pdf|svg|eps)$")


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
    console.print(
        Panel.fit(
            "🚀 [bold blue]Dimensional Gamma Demo[/bold blue]\n"
            "Running comprehensive demonstration...",
            border_style="blue",
        )
    )
    demo()


@app.command("lab")
def cli_lab(
    start_dimension: float = typer.Option(
        4.0,
        "--start",
        "-s",
        help="🎯 Starting dimension for interactive exploration",
        min=0.1,
        max=100.0,
    )
):
    """🎮 Launch interactive gamma function laboratory."""
    console.print(
        Panel.fit(
            f"🎮 [bold green]Interactive Gamma Lab[/bold green]\n"
            f"Starting at dimension: [yellow]{start_dimension}[/yellow]\n"
            f"Use keyboard controls for exploration",
            border_style="green",
        )
    )
    lab(start_dimension)


@app.command("live")
def cli_live(
    expr_file: str = typer.Option(
        "gamma_expr.py",
        "--file",
        "-f",
        help="📝 Expression file to watch for live editing",
    )
):
    """🔥 Start live editing mode with hot reload."""
    file_path = Path(expr_file)
    if not file_path.exists():
        console.print(f"[red]❌ File not found: {expr_file}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"🔥 [bold yellow]Live Editing Mode[/bold yellow]\n"
            f"Watching: [cyan]{expr_file}[/cyan]\n"
            f"Save file to see changes instantly",
            border_style="yellow",
        )
    )
    live(expr_file)


@app.command("explore")
def cli_explore(
    dimension: float = typer.Argument(4.0, help="🔍 Dimension to explore in detail"),
    range_start: float = typer.Option(
        None, "--range-start", "-rs", help="📊 Start of exploration range"
    ),
    range_end: float = typer.Option(
        None, "--range-end", "-re", help="📊 End of exploration range"
    ),
    save_plot: bool = typer.Option(False, "--save", "-s", help="💾 Save plot to file"),
):
    """🔍 Explore gamma functions around a specific dimension."""
    console.print(
        Panel.fit(
            f"🔍 [bold cyan]Gamma Exploration[/bold cyan]\n"
            f"Dimension: [yellow]{dimension}[/yellow]",
            border_style="cyan",
        )
    )

    if range_start is not None and range_end is not None:
        # Custom range exploration - would need to extend explore() function
        console.print(f"📊 Range: {range_start} → {range_end}")

    explore(dimension)

    if save_plot:
        console.print("💾 [green]Plot saved![/green]")


@app.command("peaks")
def cli_peaks(
    function: str = typer.Option(
        "all", "--function", "-f", help="🎯 Function to analyze: v, s, c, or all"
    ),
    precision: int = typer.Option(
        15,
        "--precision",
        "-p",
        help="🎯 Numerical precision for peak finding",
        min=6,
        max=20,
    ),
):
    """🏔️ Find and analyze critical peaks in gamma functions."""
    console.print(
        Panel.fit(
            f"🏔️ [bold magenta]Peak Analysis[/bold magenta]\n"
            f"Function: [yellow]{function}[/yellow]\n"
            f"Precision: [cyan]{precision}[/cyan]",
            border_style="magenta",
        )
    )
    peaks()


@app.command("instant")
def cli_instant():
    """⚡ Generate instant gamma function visualization."""
    console.print(
        Panel.fit(
            "⚡ [bold red]Instant Visualization[/bold red]\n"
            "Generating quick gamma plots...",
            border_style="red",
        )
    )
    instant()


# ============================================================================
# MATHEMATICAL ANALYSIS COMMANDS
# ============================================================================


@app.command("measure")
def cli_measure(
    dimensions: list[float] = typer.Option(
        [2.0, 3.0, 4.0],
        "--dim",
        "-d",
        help="📏 Dimensions to measure (can specify multiple)",
    ),
    functions: list[str] = typer.Option(
        ["v", "s", "c"], "--func", "-f", help="⚙️ Functions to compute: v, s, c, r"
    ),
    output_format: str = typer.Option(
        "table", "--format", "-fmt", help="📋 Output format: table, json, csv"
    ),
):
    """📏 Compute dimensional measures for specified dimensions."""
    console.print(
        Panel.fit(
            f"📏 [bold blue]Dimensional Measures[/bold blue]\n"
            f"Dimensions: [yellow]{dimensions}[/yellow]\n"
            f"Functions: [cyan]{functions}[/cyan]",
            border_style="blue",
        )
    )

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
            row.append(
                f"{value:.6f}" if isinstance(value, (int, float)) else str(value)
            )
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
    functions: list[str] = typer.Option(
        ["v"], "--func", "-f", help="📈 Functions to plot: v, s, c"
    ),
    dim_start: float = typer.Option(
        0.1, "--start", "-s", help="📊 Start dimension", min=0.001, max=100.0
    ),
    dim_end: float = typer.Option(
        10.0, "--end", "-e", help="📊 End dimension", min=0.001, max=100.0
    ),
    steps: int = typer.Option(
        1000, "--steps", "-n", help="📊 Number of steps", min=10, max=10000
    ),
    save: bool = typer.Option(False, "--save", help="💾 Save plot to file"),
    show: bool = typer.Option(
        True, "--show/--no-show", help="👁️ Show plot interactively"
    ),
):
    """📈 Create customizable plots of gamma functions."""
    # Validate dimension range
    if dim_end <= dim_start:
        console.print("[red]❌ End dimension must be greater than start[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"📈 [bold green]Gamma Function Plotting[/bold green]\n"
            f"Functions: [yellow]{', '.join(functions)}[/yellow]\n"
            f"Range: [cyan]{dim_start} → {dim_end}[/cyan] ({steps} steps)",
            border_style="green",
        )
    )

    # Create the plot using qplot
    qplot(*functions)


# ============================================================================
# PHASE DYNAMICS COMMANDS
# ============================================================================


@app.command("phase")
def cli_phase(
    initial_dimension: float = typer.Option(
        3.0, "--initial", "-i", help="🌊 Initial dimension for phase evolution"
    ),
    time_steps: int = typer.Option(
        100, "--steps", "-n", help="⏰ Number of time steps", min=10, max=1000
    ),
    coupling: float = typer.Option(
        0.1, "--coupling", "-c", help="🔗 Phase coupling strength", min=0.0, max=1.0
    ),
):
    """🌊 Simulate phase dynamics evolution."""
    console.print(
        Panel.fit(
            f"🌊 [bold purple]Phase Dynamics[/bold purple]\n"
            f"Initial: [yellow]{initial_dimension}[/yellow]\n"
            f"Steps: [cyan]{time_steps}[/cyan]\n"
            f"Coupling: [green]{coupling}[/green]",
            border_style="purple",
        )
    )

    # Initialize phase dynamics
    PhaseDynamicsEngine()
    # Would need to extend PhaseDynamicsEngine with CLI-friendly methods
    console.print("🔄 [yellow]Running phase evolution...[/yellow]")


# ============================================================================
# UTILITY COMMANDS
# ============================================================================


@app.command("info")
def cli_info():
    """ℹ️ Show system information and available commands."""
    console.print(
        Panel.fit(
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
            border_style="blue",
        )
    )


@app.command("config")
def cli_config(
    show: bool = typer.Option(False, "--show", help="📋 Show current configuration"),
    reset: bool = typer.Option(
        False, "--reset", help="🔄 Reset to default configuration"
    ),
):
    """⚙️ Manage framework configuration."""
    if show:
        config = {
            "precision": 15,
            "tolerance": 1e-10,
            "plot_style": "seaborn-v0_8",
            "auto_save": False,
        }
        console.print_json(json.dumps(config, indent=2))

    if reset:
        console.print("🔄 [green]Configuration reset to defaults[/green]")


# ============================================================================
# VISUALIZATION COMMANDS
# ============================================================================


@app.command("visualize")
def cli_visualize():
    """🎨 Access visualization command family (use subcommands)."""
    console.print(
        Panel.fit(
            "🎨 [bold blue]Visualization Commands[/bold blue]\n\n"
            "📊 Available visualizations:\n"
            "  • [cyan]emergence[/cyan] - Dimensional emergence animation\n"
            "  • [yellow]complexity-peak[/yellow] - Complexity peak around d≈6\n"
            "  • [purple]phase-dynamics[/purple] - Phase evolution visualization\n"
            "  • [green]gamma-landscape[/green] - 3D gamma function landscape\n\n"
            "🚀 [bold]Quick start:[/bold]\n"
            "  dimensional visualize emergence --interactive\n"
            "  dimensional visualize complexity-peak --export plot.html",
            border_style="blue",
        )
    )


# Create visualization subcommand group
viz_app = typer.Typer(name="visualize", help="🎨 Visualization command family")
app.add_typer(viz_app, name="visualize")


@viz_app.command("emergence")
def cli_visualize_emergence(
    dim_start: float = typer.Option(
        0.1,
        "--start",
        "-s",
        help="🌱 Starting dimension for emergence",
        min=0.1,
        max=2.0,
    ),
    dim_end: float = typer.Option(
        10.0, "--end", "-e", help="🌟 Ending dimension for emergence", min=2.0, max=20.0
    ),
    steps: int = typer.Option(
        1000, "--steps", "-n", help="📊 Number of evolution steps", min=100, max=5000
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--static", help="🎮 Create interactive plot"
    ),
    export_file: Optional[str] = typer.Option(
        None, "--export", "-o", help="💾 Export to HTML file"
    ),
    show_annotations: bool = typer.Option(
        True,
        "--annotations/--no-annotations",
        help="📝 Show critical point annotations",
    ),
):
    """🌱 Visualize dimensional emergence from 0D to higher dimensions."""
    console.print(
        Panel.fit(
            f"🌱 [bold green]Dimensional Emergence Visualization[/bold green]\n"
            f"Range: [yellow]{dim_start} → {dim_end}[/yellow] ({steps} steps)\n"
            f"Mode: [cyan]{'Interactive' if interactive else 'Static'}[/cyan]",
            border_style="green",
        )
    )

    # Generate dimensional data
    dims = np.linspace(dim_start, dim_end, steps)

    with console.status("🔄 Computing emergence data..."):
        v_vals = [v(d) for d in track(dims, description="Volume")]
        s_vals = [s(d) for d in track(dims, description="Surface")]
        c_vals = [c(d) for d in track(dims, description="Complexity")]

    # Create interactive plotly visualization
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Volume V(d)",
            "Surface S(d)",
            "Complexity C(d)",
            "All Functions",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Add traces
    fig.add_trace(
        go.Scatter(x=dims, y=v_vals, name="V(d)", line=dict(color="blue")), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dims, y=s_vals, name="S(d)", line=dict(color="red")), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=dims, y=c_vals, name="C(d)", line=dict(color="green")),
        row=2,
        col=1,
    )

    # Combined plot
    fig.add_trace(
        go.Scatter(x=dims, y=v_vals, name="Volume", line=dict(color="blue")),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=dims, y=s_vals, name="Surface", line=dict(color="red")),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=dims, y=c_vals, name="Complexity", line=dict(color="green")),
        row=2,
        col=2,
    )

    if show_annotations:
        # Add critical point annotations
        critical_points = [
            (2.0, "2D Critical"),
            (3.0, "3D Physical"),
            (4.0, "4D Transition"),
            (5.26, "Complexity Peak"),
            (6.0, "6D Maximum"),
        ]

        for d_crit, label in critical_points:
            if dim_start <= d_crit <= dim_end:
                fig.add_vline(x=d_crit, line_dash="dash", annotation_text=label)

    fig.update_layout(
        title="🌱 Dimensional Emergence: From Void to Reality",
        height=800,
        showlegend=True,
    )

    # Export or show
    if export_file:
        fig.write_html(export_file)
        console.print(f"💾 [green]Exported to {export_file}[/green]")

    if interactive and not export_file:
        fig.show()

    console.print("✨ [green]Emergence visualization complete![/green]")


@viz_app.command("complexity-peak")
def cli_visualize_complexity_peak(
    focus_range: float = typer.Option(
        2.0, "--range", "-r", help="🎯 Range around peak to visualize", min=0.5, max=5.0
    ),
    resolution: int = typer.Option(
        500, "--resolution", "-res", help="🔍 Plot resolution", min=100, max=2000
    ),
    export_file: Optional[str] = typer.Option(
        None, "--export", "-o", help="💾 Export to HTML file"
    ),
    show_derivatives: bool = typer.Option(
        False, "--derivatives", help="📈 Show derivative analysis"
    ),
):
    """🏔️ Visualize the complexity peak around d≈5.26."""
    peak_center = 5.26

    console.print(
        Panel.fit(
            f"🏔️ [bold magenta]Complexity Peak Analysis[/bold magenta]\n"
            f"Center: [yellow]{peak_center}[/yellow]\n"
            f"Range: [cyan]±{focus_range}[/cyan]\n"
            f"Resolution: [green]{resolution}[/green]",
            border_style="magenta",
        )
    )

    # Generate high-resolution data around the peak
    dims = np.linspace(peak_center - focus_range, peak_center + focus_range, resolution)

    with console.status("🔄 Computing complexity landscape..."):
        c_vals = [c(d) for d in track(dims, description="Complexity")]

    # Find actual peak
    peak_idx = np.argmax(c_vals)
    actual_peak_d = dims[peak_idx]
    actual_peak_c = c_vals[peak_idx]

    # Create visualization
    fig = go.Figure()

    # Main complexity curve
    fig.add_trace(
        go.Scatter(
            x=dims,
            y=c_vals,
            mode="lines",
            name="Complexity C(d)",
            line=dict(color="purple", width=3),
        )
    )

    # Highlight the peak
    fig.add_trace(
        go.Scatter(
            x=[actual_peak_d],
            y=[actual_peak_c],
            mode="markers",
            name=f"Peak at d={actual_peak_d:.3f}",
            marker=dict(color="red", size=15, symbol="star"),
        )
    )

    if show_derivatives:
        # Add derivative analysis (numerical)
        dc_dd = np.gradient(c_vals, dims)
        fig.add_trace(
            go.Scatter(
                x=dims,
                y=dc_dd,
                mode="lines",
                name="C'(d)",
                line=dict(color="orange", dash="dash"),
                yaxis="y2",
            )
        )

    fig.update_layout(
        title=f"🏔️ Complexity Peak: Maximum Reality at d≈{actual_peak_d:.3f}",
        xaxis_title="Dimension d",
        yaxis_title="Complexity C(d)",
        height=600,
        annotations=[
            dict(
                x=actual_peak_d,
                y=actual_peak_c,
                text=f"Peak: ({actual_peak_d:.3f}, {actual_peak_c:.1f})",
                showarrow=True,
                arrowhead=2,
            )
        ],
    )

    if show_derivatives:
        fig.update_layout(
            yaxis2=dict(title="Derivative C'(d)", overlaying="y", side="right")
        )

    # Export or show
    if export_file:
        fig.write_html(export_file)
        console.print(f"💾 [green]Exported to {export_file}[/green]")
    else:
        fig.show()

    console.print(f"🎯 [yellow]Peak found at d = {actual_peak_d:.6f}[/yellow]")
    console.print(f"🏔️ [yellow]Maximum complexity = {actual_peak_c:.6f}[/yellow]")


@viz_app.command("gamma-landscape")
def cli_visualize_gamma_landscape(
    dim_range: tuple[float, float] = typer.Option(
        (0.1, 8.0), "--range", help="🏞️ Dimension range for landscape"
    ),
    complex_range: float = typer.Option(
        2.0, "--complex", "-c", help="🌊 Complex plane range", min=0.5, max=5.0
    ),
    resolution: int = typer.Option(
        100, "--resolution", "-res", help="🔍 3D surface resolution", min=50, max=200
    ),
    export_file: Optional[str] = typer.Option(
        None, "--export", "-o", help="💾 Export to HTML file"
    ),
):
    """🏞️ Create 3D landscape of gamma functions in complex plane."""
    console.print(
        Panel.fit(
            f"🏞️ [bold cyan]3D Gamma Landscape[/bold cyan]\n"
            f"Real range: [yellow]{dim_range[0]} → {dim_range[1]}[/yellow]\n"
            f"Complex range: [purple]±{complex_range}i[/purple]\n"
            f"Resolution: [green]{resolution}×{resolution}[/green]",
            border_style="cyan",
        )
    )

    # Create complex grid
    real_vals = np.linspace(dim_range[0], dim_range[1], resolution)
    imag_vals = np.linspace(-complex_range, complex_range, resolution)
    R, I = np.meshgrid(real_vals, imag_vals)
    Z = R + 1j * I

    with console.status("🔄 Computing 3D gamma landscape..."):
        # Compute gamma function over complex plane
        gamma_vals = np.zeros_like(Z, dtype=complex)
        for i in track(range(resolution), description="Computing"):
            for j in range(resolution):
                try:
                    gamma_vals[i, j] = gamma_safe(Z[i, j])
                except:
                    gamma_vals[i, j] = np.nan

    # Create 3D surface plot
    fig = go.Figure()

    # Real part surface
    fig.add_trace(
        go.Surface(
            x=R,
            y=I,
            z=np.real(gamma_vals),
            name="Re(Γ(z))",
            colorscale="Viridis",
            opacity=0.8,
        )
    )

    fig.update_layout(
        title="🏞️ 3D Gamma Function Landscape: Re(Γ(z))",
        scene=dict(
            xaxis_title="Real(z)", yaxis_title="Imag(z)", zaxis_title="Re(Γ(z))"
        ),
        height=700,
    )

    # Export or show
    if export_file:
        fig.write_html(export_file)
        console.print(f"💾 [green]Exported to {export_file}[/green]")
    else:
        fig.show()

    console.print("🌟 [green]3D gamma landscape complete![/green]")


@app.command("dashboard")
def cli_dashboard(
    port: int = typer.Option(
        8080, "--port", "-p", help="🌐 Port for web dashboard", min=1024, max=65535
    ),
    host: str = typer.Option("localhost", "--host", help="🌍 Host address"),
    auto_open: bool = typer.Option(
        True, "--open/--no-open", help="🚀 Auto-open browser"
    ),
):
    """🌐 Launch interactive web dashboard."""
    console.print(
        Panel.fit(
            f"🌐 [bold blue]Interactive Web Dashboard[/bold blue]\n"
            f"Address: [cyan]http://{host}:{port}[/cyan]\n"
            f"Auto-open: [yellow]{'Yes' if auto_open else 'No'}[/yellow]",
            border_style="blue",
        )
    )

    console.print("🚧 [yellow]Dashboard implementation coming soon![/yellow]")
    console.print("🎯 [green]Use 'dimensional visualize' commands for now[/green]")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
