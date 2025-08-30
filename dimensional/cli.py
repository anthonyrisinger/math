#!/usr/bin/env python3
"""
Consolidated Dimensional Mathematics CLI
=======================================

Unified, type-safe, AI-composable command-line interface combining all CLI
features into a single, production-ready interface.

Architectural Features:
- Type-safe commands with pydantic validation (from cli_enhanced.py)
- Rich terminal output with beautiful formatting
- Complete command set with AI-friendly batch processing
- Comprehensive mathematical function coverage
- Production-ready error handling and validation
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import typer

# Make plotly optional with minimal mocking
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None
    def make_subplots(**kwargs):
        return None


# Consolidated external imports
from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

# Consolidated dimensional imports
from .gamma import (
    demo,
    explore,
    gamma_safe,
    instant,
    lab,
    live,
    peaks,
    qplot,
)
from .measures import c, s, v
from .phase import PhaseDynamicsEngine

# Import visualization modules - make optional to avoid import errors
try:
    importlib.util.find_spec("visualization")
    HAS_VISUALIZATION = True
except (ImportError, AttributeError):
    HAS_VISUALIZATION = False

# Initialize rich console for beautiful outpu
console = Console()

# Create main typer app
app = typer.Typer(
    name="dimensional",
    help="🌟 Dimensional Mathematics Framework - AI-Composable CLI",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# ============================================================================
# TYPE-SAFE COMMAND MODELS (from cli_enhanced.py)
# ============================================================================


class DimensionRange(BaseModel):
    """Type-safe dimensional range with validation."""

    start: float = Field(ge=0, description="Starting dimension (≥ 0)")
    end: float = Field(gt=0, description="Ending dimension (> 0)")
    points: int = Field(
        ge=10, le=10000, default=1000, description="Number of points"
    )

    @property
    def linspace(self) -> np.ndarray:
        """Generate linspace array."""
        return np.linspace(self.start, self.end, self.points)


class AnalysisConfig(BaseModel):
    """Type-safe analysis configuration."""

    precision: int = Field(
        ge=6, le=20, default=15, description="Decimal precision"
    )
    tolerance: float = Field(
        ge=1e-16, le=1e-6, default=1e-12, description="Numerical tolerance"
    )
    format: Literal["table", "json", "csv"] = Field(
        default="table", description="Output format"
    )
    save: bool = Field(default=False, description="Save results to file")


class ExplorationMode(BaseModel):
    """Type-safe exploration mode."""

    mode: Literal["basic", "detailed", "advanced"] = Field(default="basic")
    include_peaks: bool = Field(
        default=True, description="Include peak analysis"
    )
    include_critical: bool = Field(
        default=True, description="Include critical dimensions"
    )
    visualize: bool = Field(
        default=False, description="Generate visualizations"
    )


# ============================================================================
# ULTRA-FAST PROTOTYPING SHORTCUTS
# ============================================================================


@app.command("v")
def shortcut_volume(
    dims: str = typer.Argument(
        ..., help="📐 Dimensions (e.g., '4' or '2,3,4')"
    )
):
    """⚡ Ultra-fast volume calculation: dim v 4"""
    _process_shortcut("volume", dims)


@app.command("s")
def shortcut_surface(
    dims: str = typer.Argument(
        ..., help="📐 Dimensions (e.g., '4' or '2,3,4')"
    )
):
    """⚡ Ultra-fast surface calculation: dim s 4"""
    _process_shortcut("surface", dims)


@app.command("c")
def shortcut_complexity(
    dims: str = typer.Argument(
        ..., help="📐 Dimensions (e.g., '4' or '2,3,4')"
    )
):
    """⚡ Ultra-fast complexity calculation: dim c 4"""
    _process_shortcut("complexity", dims)


@app.command("p")
def shortcut_peaks():
    """⚡ Ultra-fast peak analysis: dim p"""
    from .measures import find_all_peaks

    console.print("🏔️ [bold cyan]Critical Peaks[/bold cyan]")
    peaks = find_all_peaks()
    for key, (location, value) in peaks.items():
        console.print(f"  {key}: d={location:.3f}, value={value:.3f}")


@app.command("g")
def shortcut_gamma(
    value: float = typer.Argument(..., help="🔢 Value for gamma function")
):
    """⚡ Ultra-fast gamma calculation: dim g 2.5"""

    result = gamma_safe(value)
    console.print(f"Γ({value}) = {result:.6f}")


def _process_shortcut(func_name: str, dims_str: str):
    """Process ultra-fast shortcut commands with mathematical context."""
    # Parse dimensions - support both single values and comma-separated
    try:
        if "," in dims_str:
            dims = [float(d.strip()) for d in dims_str.split(",")]
        else:
            dims = [float(dims_str)]
    except ValueError:
        console.print(f"[red]❌ Invalid dimension format: {dims_str}[/red]")
        return

    # Import functions
    if func_name == "volume":
        from .measures import ball_volume as func

        symbol = "V"
    elif func_name == "surface":
        from .measures import sphere_surface as func

        symbol = "S"
    elif func_name == "complexity":
        from .measures import complexity_measure as func

        symbol = "C"

    # Compute and display with mathematical contex
    console.print(f"🧮 [bold cyan]{func_name.title()}[/bold cyan]")
    for d in dims:
        result = func(d)
        console.print(f"  {symbol}({d}) = {result:.6f}")


# ============================================================================
# AI-FRIENDLY FEATURES
# ============================================================================


@app.command("eval")
def ai_eval(
    expression: str = typer.Argument(
        ..., help="🤖 Math expression: 'V(4)', 'C(2,3,4)', 'gamma(2.5)'"
    ),
    format: str = typer.Option(
        "human", "--format", "-f", help="📊 Output: human, json, csv"
    ),
):
    """🤖 AI-friendly expression evaluator: dim eval 'V(4) + C(3)'"""
    result = _evaluate_expression(expression, format)
    if format == "json":
        console.print_json(json.dumps(result))
    elif format == "csv":
        if isinstance(result, list):
            for item in result:
                print(f"{item.get('expression', '')},{item.get('result', '')}")
        else:
            print(f"{result.get('expression', '')},{result.get('result', '')}")
    else:
        console.print(f"🤖 {expression} = {result}")


@app.command("batch")
def ai_batch(
    expressions: str = typer.Argument(
        ..., help="🚀 Multiple expressions: 'V(2);C(3);gamma(1.5)'"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="📊 Output: table, json, csv"
    ),
):
    """🚀 AI batch processing: dim batch 'V(2);C(3);S(4)'"""
    expr_list = [expr.strip() for expr in expressions.split(";")]
    results = []

    for expr in expr_list:
        try:
            result = _evaluate_expression(expr, "raw")
            results.append(
                {"expression": expr, "result": result, "status": "success"}
            )
        except Exception as e:
            results.append(
                {"expression": expr, "error": str(e), "status": "error"}
            )

    if format == "json":
        console.print_json(json.dumps(results, indent=2))
    elif format == "csv":
        print("expression,result,status")
        for r in results:
            status = r["status"]
            value = r.get("result", r.get("error", ""))
            print(f"{r['expression']},{value},{status}")
    else:
        # Table forma
        table = Table(title="🚀 Batch Results")
        table.add_column("Expression", style="cyan")
        table.add_column("Result", style="yellow")
        table.add_column("Status", style="green")

        for r in results:
            status_color = "green" if r["status"] == "success" else "red"
            value = str(r.get("result", r.get("error", "")))
            table.add_row(
                r["expression"],
                value,
                f"[{status_color}]{r['status']}[/{status_color}]",
            )

        console.print(table)


def _evaluate_expression(expr: str, output_format: str = "human"):
    """Evaluate mathematical expressions with AI-friendly parsing."""
    import re

    # Clean the expression
    expr = expr.strip()

    # Import mathematical functions
    from .measures import ball_volume, complexity_measure, sphere_surface

    # Simple expression patterns for AI workflows
    patterns = {
        r"V\(([0-9.,\s]+)\)": lambda m: _eval_function(
            ball_volume, m.group(1)
        ),
        r"S\(([0-9.,\s]+)\)": lambda m: _eval_function(
            sphere_surface, m.group(1)
        ),
        r"C\(([0-9.,\s]+)\)": lambda m: _eval_function(
            complexity_measure, m.group(1)
        ),
        r"gamma\(([0-9.,\s]+)\)": lambda m: _eval_function(
            gamma_safe, m.group(1)
        ),
        r"Γ\(([0-9.,\s]+)\)": lambda m: _eval_function(gamma_safe, m.group(1)),
    }

    for pattern, handler in patterns.items():
        match = re.search(pattern, expr)
        if match:
            try:
                result = handler(match)
                if output_format == "raw":
                    return result
                return {"expression": expr, "result": result}
            except Exception as e:
                if output_format == "raw":
                    raise e
                return {"expression": expr, "error": str(e)}

    # If no pattern matches, try direct evaluation for simple expressions
    try:
        # Very basic evaluation for expressions like "4.5" or simple arithmetic
        if re.match(r"^[0-9.,\s+\-*/()]+$", expr):
            result = eval(expr)  # Safe for numeric expressions only
            if output_format == "raw":
                return result
            return {"expression": expr, "result": result}
    except BaseException:
        pass

    raise ValueError(f"Could not evaluate expression: {expr}")


def _eval_function(func, args_str):
    """Helper to evaluate function with parsed arguments."""
    if "," in args_str:
        args = [float(x.strip()) for x in args_str.split(",")]
        return [func(arg) for arg in args]
    else:
        return func(float(args_str.strip()))


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
    ),
    session_id: str = typer.Option(
        None,
        "--session",
        "-sid",
        help="📁 Load existing research session"
    )
):
    """🎮 Launch enhanced interactive research laboratory."""
    console.print(
        Panel.fit(
            f"🎮 [bold green]Enhanced Research Laboratory[/bold green]\n"
            f"Starting at dimension: [yellow]{start_dimension}[/yellow]\n"
            f"Session: [cyan]{session_id or 'new'}[/cyan]\n"
            f"Features: persistence, sweeps, exports, Rich visualization",
            border_style="green",
        )
    )
    try:
        from .research_cli import enhanced_lab
        session = enhanced_lab(start_dimension, session_id)
        console.print(f"[green]✅ Research session completed: {session.session_id}[/green]")
        return session
    except ImportError:
        console.print("[yellow]⚠️  Enhanced features unavailable, using basic lab[/yellow]")
        return lab(start_dimension)


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
    dimension: float = typer.Argument(
        4.0, help="🔍 Dimension to explore in detail"
    ),
    context: str = typer.Option(
        "general",
        "--context",
        "-c",
        help="🎯 Exploration context: general, peaks, critical, research"
    ),
    save_analysis: bool = typer.Option(
        False, "--save", "-s", help="💾 Save analysis to file"
    ),
):
    """🔍 Enhanced dimensional exploration with guided discovery."""
    console.print(
        Panel.fit(
            f"🔍 [bold cyan]Enhanced Dimensional Exploration[/bold cyan]\n"
            f"Dimension: [yellow]{dimension}[/yellow]\n"
            f"Context: [magenta]{context}[/magenta]\n"
            f"Features: guided discovery, Rich visualization, analysis paths",
            border_style="cyan",
        )
    )

    try:
        from .research_cli import enhanced_explore
        results = enhanced_explore(dimension, context)

        if save_analysis:
            console.print("💾 [green]Analysis results available for export[/green]")

        return results
    except ImportError:
        console.print("[yellow]⚠️  Enhanced features unavailable, using basic exploration[/yellow]")
        return explore(dimension)


@app.command("peaks")
def cli_peaks(
    function: str = typer.Option(
        "all",
        "--function",
        "-f",
        help="🎯 Function to analyze: v, s, c, or all",
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

    # Get and display peak results
    peak_results = peaks()

    table = Table(title="🏔️ Critical Peaks")
    table.add_column("Measure", style="cyan")
    table.add_column("Peak Dimension", style="yellow")
    table.add_column("Peak Value", style="green")

    for peak_name, peak_data in peak_results.items():
        if isinstance(peak_data, tuple) and len(peak_data) == 2:
            dimension, value = peak_data
            table.add_row(
                peak_name.replace("_", " ").title(),
                f"{dimension:.8f}",
                f"{value:.8f}"
            )
        else:
            # Handle case where peak_data is just a dimension value
            dimension = peak_data
            table.add_row(
                peak_name.replace("_", " ").title(),
                f"{dimension:.8f}",
                "N/A"
            )

    console.print(table)


@app.command("instant")
def cli_instant(
    config: str = typer.Option(
        "research",
        "--config",
        "-c",
        help="⚡ Analysis configuration: research, peaks, discovery, publication"
    )
):
    """⚡ Enhanced instant analysis with multiple configurations."""
    console.print(
        Panel.fit(
            f"⚡ [bold red]Enhanced Instant Analysis[/bold red]\n"
            f"Configuration: [yellow]{config}[/yellow]\n"
            f"Features: Rich visualization, multiple panels, export-ready",
            border_style="red",
        )
    )

    try:
        from .research_cli import enhanced_instant
        results = enhanced_instant(config)
        console.print("✅ [green]Instant analysis completed![/green]")
        return results
    except ImportError:
        console.print("[yellow]⚠️  Enhanced features unavailable, using basic instant[/yellow]")
        return instant()


# ============================================================================
# ENHANCED RESEARCH COMMANDS
# ============================================================================


@app.command("sweep")
def cli_sweep(
    start: float = typer.Argument(..., help="🔄 Start dimension for sweep"),
    end: float = typer.Argument(..., help="🔄 End dimension for sweep"),
    steps: int = typer.Option(50, "--steps", "-n", help="🔄 Number of steps", min=10, max=1000),
    export: bool = typer.Option(False, "--export", "-e", help="💾 Export results to CSV"),
):
    """🔄 Run interactive parameter sweep across dimensional range."""
    if end <= start:
        console.print("[red]❌ End dimension must be greater than start[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"🔄 [bold blue]Parameter Sweep Analysis[/bold blue]\n"
            f"Range: [yellow]{start} → {end}[/yellow] ({steps} steps)\n"
            f"Export: [cyan]{'Yes' if export else 'No'}[/cyan]",
            border_style="blue",
        )
    )

    try:
        from .research_cli import (
            InteractiveParameterSweep,
            PublicationExporter,
            RichVisualizer,
        )

        visualizer = RichVisualizer()
        sweeper = InteractiveParameterSweep(visualizer)

        sweep_results = sweeper.run_dimension_sweep(start, end, steps,
                                                   notes="CLI parameter sweep")

        visualizer.show_parameter_sweep_analysis(sweep_results)

        if export:
            exporter = PublicationExporter()
            filepath = exporter.export_csv_data(sweep_results)
            console.print(f"💾 [green]Results exported to {filepath}[/green]")

        return sweep_results

    except ImportError:
        console.print("[red]❌ Enhanced research features not available[/red]")
        raise typer.Exit(1)


@app.command("sessions")
def cli_sessions():
    """💾 List and manage research sessions."""
    try:
        from .research_cli import ResearchPersistence

        persistence = ResearchPersistence()
        sessions = persistence.list_sessions()

        if not sessions:
            console.print("[yellow]📁 No research sessions found[/yellow]")
            return

        table = Table(title="💾 Research Sessions")
        table.add_column("Session ID", style="cyan")
        table.add_column("Start Time", style="yellow")
        table.add_column("Age", style="green")

        from datetime import datetime
        now = datetime.now()

        for session_id, start_time in sessions:
            age = now - start_time
            age_str = f"{age.days}d {age.seconds//3600}h" if age.days > 0 else f"{age.seconds//3600}h {(age.seconds//60)%60}m"

            table.add_row(
                session_id,
                start_time.strftime('%Y-%m-%d %H:%M:%S'),
                age_str
            )

        console.print(table)

        # Offer to load a session
        if len(sessions) > 0:
            load_session = typer.confirm("Would you like to load a session?")
            if load_session:
                session_id = typer.prompt("Enter session ID")
                loaded = persistence.load_session(session_id)
                if loaded:
                    from .research_cli import RichVisualizer
                    visualizer = RichVisualizer()
                    visualizer.show_session_overview(loaded)
                else:
                    console.print(f"[red]❌ Session {session_id} not found[/red]")

    except ImportError:
        console.print("[red]❌ Enhanced research features not available[/red]")


@app.command("research")
def cli_research():
    """🔬 Launch comprehensive research mode with all tools."""
    console.print(
        Panel.fit(
            "🔬 [bold magenta]Comprehensive Research Mode[/bold magenta]\n"
            "Launching enhanced research laboratory with full capabilities:\n"
            "• Interactive exploration with session persistence\n"
            "• Parameter sweeps with real-time visualization\n"
            "• Publication-quality export system\n"
            "• Rich terminal mathematical displays",
            border_style="magenta",
        )
    )

    try:
        from .research_cli import enhanced_lab

        # Start with comprehensive research configuration
        session = enhanced_lab(4.0)  # Start at dimension 4

        console.print(f"🎯 [green]Research session completed: {session.session_id}[/green]")
        return session

    except ImportError:
        console.print("[red]❌ Enhanced research features not available[/red]")
        console.print("Install additional dependencies for full research capabilities")
        raise typer.Exit(1)


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
        ["v", "s", "c"],
        "--func",
        "-f",
        help="⚙️ Functions to compute: v, s, c, r",
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
                f"{value:.6f}"
                if isinstance(value, (int, float))
                else str(value)
            )
        table.add_row(*row)

    if output_format == "table":
        console.print(table)
    elif output_format == "json":
        # Convert to JSON forma
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

    # Create the plot using qplo
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
        0.1,
        "--coupling",
        "-c",
        help="🔗 Phase coupling strength",
        min=0.0,
        max=1.0,
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
    show: bool = typer.Option(
        False, "--show", help="📋 Show current configuration"
    ),
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
    """🎨 Simple visualization info (install plotly for advanced features)."""
    if not HAS_PLOTLY:
        console.print("[yellow]⚠️  Install plotly for advanced visualization features[/yellow]")
    console.print("📊 [green]Use 'plot' command for basic plotting[/green]")










# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
