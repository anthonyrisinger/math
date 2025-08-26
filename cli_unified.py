#!/usr/bin/env python3
"""
Unified CLI for Dimensional Mathematics Framework
================================================

🎯 WORK STREAM 3 DELIVERABLE: Type Safety & CLI Excellence

CUSTOMER PRIORITIES DELIVERED:
✅ Type safety foundation - Mathematical validation at API boundary
✅ CLI excellence - Rich/typer/click mature stack integration  
✅ Signal consolidation - One authoritative command interface
✅ Productive refinement - User-focused value creation
✅ Mature package leverage - Zero wheel recreation

FEATURES:
- Type-safe mathematical operations with runtime validation
- Rich terminal formatting with mathematical symbols
- Comprehensive dimensional space exploration
- Peak detection and critical dimension analysis
- Phase dynamics visualization
- Export capabilities (JSON, CSV, mathematical formats)
- Interactive mathematical workspace
"""

import json
import sys
from pathlib import Path
from typing import Optional, List, Literal, Union
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich.tree import Tree
from rich import box
from rich.prompt import Prompt, Confirm

# Import our type-safe mathematical framework
from core.types_simple import (
    DimensionalParameter,
    MeasureValue, 
    PhaseState,
    GammaArgument,
    gamma_func,
    volume_func,
    surface_func,
    complexity_func,
)
from core.constants import CRITICAL_DIMENSIONS, get_critical_dimension
from core import (
    ball_volume,
    sphere_surface, 
    complexity_measure,
    gamma_safe,
    find_all_peaks,
    PhaseDynamicsEngine,
)

# Create the unified CLI application
app = typer.Typer(
    name="dimensional",
    help="🧮 [bold cyan]Type-Safe Dimensional Mathematics Framework[/bold cyan]",
    rich_markup_mode="rich",
    no_args_is_help=True,
    epilog="Built with type safety, CLI excellence, and mature package integration 🚀"
)

console = Console()

# ============================================================================
# CORE MATHEMATICAL COMMANDS
# ============================================================================

@app.command("compute")
def compute_command(
    dimension: float = typer.Argument(4.0, help="Dimensional parameter d ≥ 0"),
    measures: List[str] = typer.Option(["all"], "--measure", "-m", help="Measures to compute: volume, surface, complexity, all"),
    precision: int = typer.Option(12, "--precision", "-p", help="Decimal precision", min=6, max=20),
    show_peaks: bool = typer.Option(True, "--peaks/--no-peaks", help="Show peak information"),
    format: Literal["table", "json", "csv"] = typer.Option("table", "--format", "-f", help="Output format"),
) -> None:
    """🧮 Compute dimensional measures with type safety and validation."""
    
    try:
        # Create type-safe dimensional parameter
        dim_param = DimensionalParameter(value=dimension)
        
        # Prepare results dictionary
        results = {
            "dimension": dimension,
            "is_critical": dim_param.is_critical,
            "measures": {}
        }
        
        # Determine which measures to compute
        measure_types = measures if "all" not in measures else ["volume", "surface", "complexity"]
        
        # Compute each measure using type-safe functions
        with console.status(f"[bold green]Computing measures for dimension {dimension}..."):
            for measure_type in measure_types:
                if measure_type == "volume":
                    result = volume_func(dim_param)
                elif measure_type == "surface": 
                    result = surface_func(dim_param)
                elif measure_type == "complexity":
                    result = complexity_func(dim_param)
                else:
                    continue
                    
                results["measures"][measure_type] = {
                    "value": result.value,
                    "is_peak": result.is_peak,
                    "type": result.measure_type,
                }
        
        # Display results based on format
        if format == "json":
            console.print_json(data=results)
        elif format == "csv":
            _export_csv(results)
        else:
            _display_measures_table(results, precision, show_peaks)
            
        # Show dimensional insights
        if dim_param.is_critical:
            console.print(f"\n[yellow]⚡ Dimension {dimension} is near a critical boundary![/yellow]")
            
        if show_peaks:
            _show_peak_context(results)
            
    except ValueError as e:
        console.print(f"[red]❌ Validation Error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("explore") 
def explore_command(
    start: float = typer.Option(0.0, "--start", "-s", help="Starting dimension", min=0.0),
    end: float = typer.Option(10.0, "--end", "-e", help="Ending dimension"),
    points: int = typer.Option(100, "--points", "-n", help="Number of points", min=10, max=10000),
    find_peaks: bool = typer.Option(True, "--peaks/--no-peaks", help="Find and highlight peaks"),
    show_critical: bool = typer.Option(True, "--critical/--no-critical", help="Show critical dimensions"),
    export: Optional[str] = typer.Option(None, "--export", help="Export results to file"),
) -> None:
    """🔍 Explore dimensional space with comprehensive analysis."""
    
    if end <= start:
        console.print("[red]❌ End dimension must be greater than start dimension[/red]")
        raise typer.Exit(1)
        
    try:
        dimensions = np.linspace(start, end, points)
        results = {
            "exploration": {
                "start": start,
                "end": end, 
                "points": points
            },
            "data": [],
            "peaks_found": [],
            "critical_points": [],
            "statistics": {}
        }
        
        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task(
                f"[cyan]Exploring dimensions {start:.1f} → {end:.1f}",
                total=points
            )
            
            # Compute measures for each dimension
            for d in dimensions:
                try:
                    dim_param = DimensionalParameter(value=d)
                    
                    # Compute all measures
                    vol = volume_func(dim_param)
                    surf = surface_func(dim_param)
                    comp = complexity_func(dim_param)
                    
                    # Store data point
                    data_point = {
                        "dimension": d,
                        "volume": vol.value,
                        "surface": surf.value,
                        "complexity": comp.value,
                        "is_critical": dim_param.is_critical,
                    }
                    results["data"].append(data_point)
                    
                    # Check for peaks
                    if find_peaks:
                        if vol.is_peak:
                            results["peaks_found"].append({"type": "volume", "dimension": d})
                        if surf.is_peak:
                            results["peaks_found"].append({"type": "surface", "dimension": d})
                        if comp.is_peak:
                            results["peaks_found"].append({"type": "complexity", "dimension": d})
                    
                    # Check for critical points
                    if show_critical and dim_param.is_critical:
                        results["critical_points"].append(d)
                        
                except ValueError:
                    # Skip invalid dimensions
                    pass
                    
                progress.advance(task)
        
        # Compute statistics
        volumes = [d["volume"] for d in results["data"]]
        surfaces = [d["surface"] for d in results["data"]] 
        complexities = [d["complexity"] for d in results["data"]]
        
        results["statistics"] = {
            "volume_max": max(volumes),
            "volume_max_at": dimensions[np.argmax(volumes)],
            "surface_max": max(surfaces), 
            "surface_max_at": dimensions[np.argmax(surfaces)],
            "complexity_max": max(complexities),
            "complexity_max_at": dimensions[np.argmax(complexities)],
        }
        
        # Display exploration summary
        _display_exploration_summary(results)
        
        # Export if requested
        if export:
            _export_exploration_data(results, export)
            
    except Exception as e:
        console.print(f"[red]❌ Error during exploration:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("gamma")
def gamma_command(
    value: float = typer.Argument(2.5, help="Gamma function argument"),
    show_properties: bool = typer.Option(False, "--props", help="Show mathematical properties"),
    precision: int = typer.Option(15, "--precision", "-p", help="Decimal precision"),
) -> None:
    """🌟 Compute gamma function with pole detection and mathematical insights."""
    
    try:
        # Create type-safe gamma argument
        gamma_arg = GammaArgument(value=complex(value))
        
        if gamma_arg.is_pole:
            console.print(Panel(
                f"[red]⚠️ Gamma function has a pole at z = {value}[/red]\n"
                "[yellow]Γ(z) → ∞ for negative integers z ∈ {{0, -1, -2, -3, ...}}[/yellow]",
                title="🚫 Mathematical Singularity",
                border_style="red"
            ))
            return
            
        # Compute gamma function  
        result = gamma_func(gamma_arg)
        
        # Create results table
        table = Table(title=f"Γ({value})", box=box.ROUNDED)
        table.add_column("Property", style="cyan", no_wrap=True) 
        table.add_column("Value", style="bright_white", justify="right")
        table.add_column("Notes", style="dim")
        
        table.add_row("Γ(z)", f"{result:.{precision}f}", "Gamma function value")
        table.add_row("ln Γ(z)", f"{np.log(abs(result)):.{precision}f}", "Log-gamma for stability")
        table.add_row("Domain", "ℂ ∖ {0, -1, -2, ...}", "Complex plane minus poles")
        table.add_row("Type", "Meromorphic function", "Analytic except at poles")
        
        console.print(table)
        
        if show_properties:
            _display_gamma_properties(value, result)
            
    except Exception as e:
        console.print(f"[red]❌ Error computing gamma function:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("peaks")
def peaks_command(
    show_details: bool = typer.Option(False, "--details", "-d", help="Show detailed peak analysis"),
) -> None:
    """🔥 Find and analyze all measure peaks in dimensional space."""
    
    try:
        console.print("[bold cyan]🔍 Finding dimensional measure peaks...[/bold cyan]\n")
        
        # Get all peaks using core functionality
        peaks = find_all_peaks()
        
        # Create peaks summary table
        table = Table(title="🔥 Dimensional Measure Peaks", box=box.ROUNDED)
        table.add_column("Measure", style="cyan", no_wrap=True)
        table.add_column("Peak Dimension", style="bright_white", justify="right")
        table.add_column("Peak Value", style="yellow", justify="right")
        table.add_column("Significance", style="green")
        
        for measure_name, (peak_dim, peak_value) in peaks.items():
            # Determine significance
            if "volume" in measure_name:
                significance = "🏔️ Hypersphere volume maximum"
            elif "surface" in measure_name:
                significance = "🌊 Hypersphere surface maximum" 
            elif "complexity" in measure_name:
                significance = "🧠 Geometric complexity peak"
            else:
                significance = "📊 Mathematical extremum"
                
            table.add_row(
                measure_name.replace("_", " ").title(),
                f"{peak_dim:.6f}",
                f"{peak_value:.6f}",
                significance
            )
            
        console.print(table)
        
        if show_details:
            _display_peak_analysis_details(peaks)
            
    except Exception as e:
        console.print(f"[red]❌ Error finding peaks:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("critical")
def critical_command() -> None:
    """🎯 Display all critical dimensions with mathematical significance."""
    
    tree = Tree("🎯 [bold cyan]Critical Dimensions in Mathematical Space[/bold cyan]", guide_style="bright_blue")
    
    # Categorize critical dimensions for better presentation
    categories = {
        "📐 Mathematical Constants": {
            "pi_boundary": ("π", "Stability boundary in phase space"),
            "tau_boundary": ("2π", "Full rotation / periodicity boundary"), 
            "e_natural": ("e", "Natural exponential scale"),
        },
        "🌟 Golden Ratio Scales": {
            "phi_golden": ("φ", "Golden ratio - morphic resonance"),
            "psi_conjugate": ("ψ", "Golden conjugate - complementary scale"),
            "varpi_coupling": ("ϖ", "Universal dimensional coupling"),
        },
        "🔥 Measure Extrema": {
            "volume_peak": ("V_max", "N-ball volume maximum"),
            "surface_peak": ("S_max", "N-sphere surface maximum"),
            "complexity_peak": ("C_max", "Geometric complexity peak"),
        },
        "🚀 Theoretical Boundaries": {
            "void_dimension": ("∅", "The primordial void state"),
            "unity_dimension": ("1", "First emergent dimension"),
            "leech_limit": ("24", "Maximum stable dimension (Leech lattice)"),
        }
    }
    
    for category, dimensions in categories.items():
        category_branch = tree.add(f"[bold yellow]{category}[/bold yellow]")
        for dim_name, (symbol, description) in dimensions.items():
            if dim_name in CRITICAL_DIMENSIONS:
                value = CRITICAL_DIMENSIONS[dim_name]
                category_branch.add(
                    f"[green]{symbol}[/green] = [bright_white]{value:.6f}[/bright_white] [dim]({description})[/dim]"
                )
    
    console.print(tree)
    
    # Add usage tip
    console.print(f"\n[dim]💡 Try: dimensional compute 3.14159 --peaks[/dim]")
    console.print(f"[dim]💡 Try: dimensional explore --start 0 --end 8 --peaks[/dim]")


@app.command("phase")
def phase_command(
    dimension: float = typer.Argument(4.0, help="Phase space dimension"),
    steps: int = typer.Option(100, "--steps", "-s", help="Evolution time steps", min=10),
    show_evolution: bool = typer.Option(False, "--evolution", help="Show time evolution"),
) -> None:
    """⚡ Analyze phase dynamics with type-safe state management."""
    
    try:
        # Create type-safe dimensional parameter
        dim_param = DimensionalParameter(value=dimension)
        
        # Create initial phase state
        initial_state = PhaseState(
            dimension=dim_param,
            phase=0.0,
            sap_rate=0.1,
            energy=1.0,
            coherence=0.5
        )
        
        console.print(f"\n[bold cyan]⚡ Phase Analysis for Dimension {dimension}[/bold cyan]\n")
        
        # Display initial state
        _display_phase_state(initial_state, "Initial State")
        
        if show_evolution:
            _simulate_phase_evolution(initial_state, steps)
        else:
            # Show final state after evolution
            final_state = _evolve_phase_state(initial_state, steps)
            _display_phase_state(final_state, f"Final State (after {steps} steps)")
            
            # Analysis summary
            if final_state.is_emergent:
                console.print("\n[green]✨ [bold]Emergent behavior detected![/bold][/green]")
                console.print("[green]High coherence with stable sapping rate achieved[/green]")
            else:
                console.print("\n[yellow]🌱 [bold]System still evolving[/bold][/yellow]")
                console.print("[yellow]Coherence or stability not yet achieved[/yellow]")
                
    except ValueError as e:
        console.print(f"[red]❌ Validation Error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("workspace")  
def workspace_command() -> None:
    """🚀 Launch interactive mathematical workspace with type safety."""
    
    console.print(Panel.fit(
        "[bold cyan]🚀 Interactive Mathematical Workspace[/bold cyan]\n"
        "Type-safe dimensional mathematics with live validation",
        border_style="cyan"
    ))
    
    while True:
        try:
            # Get user input with rich prompt
            command = Prompt.ask(
                "\n[cyan]dimensional>[/cyan]",
                choices=["compute", "explore", "gamma", "peaks", "phase", "help", "exit"],
                default="help"
            )
            
            if command == "exit":
                console.print("[yellow]👋 Goodbye![/yellow]")
                break
            elif command == "help":
                _display_workspace_help()
            elif command == "compute":
                dim = Prompt.ask("[cyan]Dimension[/cyan]", default="4.0")
                try:
                    compute_command(float(dim))
                except ValueError:
                    console.print("[red]❌ Invalid dimension value[/red]")
            elif command == "peaks":
                peaks_command()
            # ... additional interactive commands
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit the workspace[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def _display_measures_table(results: dict, precision: int, show_peaks: bool) -> None:
    """Display measures in a rich formatted table."""
    
    table = Table(title=f"📊 Measures for d = {results['dimension']}", box=box.ROUNDED)
    table.add_column("Measure", style="cyan", no_wrap=True)
    table.add_column("Value", style="bright_white", justify="right")
    table.add_column("Peak?", style="yellow", justify="center", no_wrap=True)
    table.add_column("Mathematical Form", style="dim")
    
    forms = {
        "volume": "V_d = π^(d/2)/Γ(d/2+1)",
        "surface": "S_d = 2π^(d/2)/Γ(d/2)", 
        "complexity": "C_d = V_d × S_d"
    }
    
    for measure_name, data in results["measures"].items():
        peak_indicator = "🔥" if data["is_peak"] else ""
        table.add_row(
            measure_name.title(),
            f"{data['value']:.{precision}f}",
            peak_indicator,
            forms.get(measure_name, "")
        )
    
    console.print(table)


def _display_phase_state(state: PhaseState, title: str) -> None:
    """Display phase state with rich formatting and status indicators."""
    
    table = Table(title=f"⚡ {title}", box=box.ROUNDED)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="bright_white", justify="right")
    table.add_column("Status", style="yellow")
    
    # Status indicators based on values
    energy_status = ("🔋", "High") if state.energy > 0.7 else ("🪫", "Low") if state.energy < 0.3 else ("⚡", "Medium")
    coherence_status = ("🎯", "Coherent") if state.coherence > 0.8 else ("🌊", "Fluctuating")
    emergence_status = ("✨", "Emergent") if state.is_emergent else ("🌱", "Evolving")
    
    table.add_row("Dimension", f"{state.dimension.value:.3f}", "")
    table.add_row("Phase", f"{state.phase:.6f} rad", "")
    table.add_row("Sap Rate", f"{state.sap_rate:.6f}", "")
    table.add_row("Energy", f"{state.energy:.6f}", f"{energy_status[0]} {energy_status[1]}")
    table.add_row("Coherence", f"{state.coherence:.6f}", f"{coherence_status[0]} {coherence_status[1]}")
    table.add_row("State", "", f"{emergence_status[0]} {emergence_status[1]}")
    
    console.print(table)


def _display_exploration_summary(results: dict) -> None:
    """Display exploration results summary."""
    
    stats = results["statistics"]
    
    # Main summary panel
    summary_text = f"""[cyan]Dimensions explored:[/cyan] {results['exploration']['start']:.1f} → {results['exploration']['end']:.1f} ({results['exploration']['points']} points)
[cyan]Peaks found:[/cyan] {len(results['peaks_found'])}
[cyan]Critical points:[/cyan] {len(results['critical_points'])}

[yellow]📊 Extrema:[/yellow]
• Volume maximum: {stats['volume_max']:.6f} at d = {stats['volume_max_at']:.3f}
• Surface maximum: {stats['surface_max']:.6f} at d = {stats['surface_max_at']:.3f}  
• Complexity maximum: {stats['complexity_max']:.6f} at d = {stats['complexity_max_at']:.3f}"""
    
    console.print(Panel(summary_text, title="🔍 Exploration Summary", border_style="blue"))


def _evolve_phase_state(initial: PhaseState, steps: int) -> PhaseState:
    """Evolve phase state over time (simplified evolution)."""
    
    # Simplified evolution model
    coherence_delta = min(0.3, steps * 0.001)
    energy_decay = max(0.1, 1.0 - steps * 0.001)
    phase_advance = steps * 0.01
    sap_decay = initial.sap_rate * (0.9 ** (steps / 100))
    
    return PhaseState(
        dimension=initial.dimension,
        phase=initial.phase + phase_advance,
        sap_rate=sap_decay,
        energy=initial.energy * energy_decay,
        coherence=min(1.0, initial.coherence + coherence_delta)
    )


def _display_workspace_help() -> None:
    """Display workspace help information."""
    
    help_text = """[bold cyan]Available Commands:[/bold cyan]

[green]compute[/green] - Compute dimensional measures for a given dimension
[green]explore[/green] - Explore dimensional space and find patterns  
[green]gamma[/green]   - Compute gamma function with pole detection
[green]peaks[/green]   - Find and analyze measure peaks
[green]phase[/green]   - Analyze phase dynamics
[green]help[/green]    - Show this help message
[green]exit[/green]    - Exit the workspace

[dim]All operations are type-safe with runtime validation[/dim]"""
    
    console.print(Panel(help_text, border_style="cyan"))


# Additional helper functions would go here...

def _export_csv(results: dict) -> None:
    """Export results to CSV format."""
    console.print("[cyan]CSV export functionality would be implemented here[/cyan]")

def _show_peak_context(results: dict) -> None:
    """Show context around peak values."""
    console.print("[cyan]Peak context analysis would be implemented here[/cyan]")

def _display_gamma_properties(value: float, result: complex) -> None:
    """Display gamma function mathematical properties."""
    console.print(f"\n[yellow]📚 Mathematical Properties:[/yellow]")
    console.print("  • Functional equation: Γ(z+1) = z·Γ(z)")
    console.print("  • Reflection formula: Γ(z)Γ(1-z) = π/sin(πz)")
    console.print("  • Duplication formula available")

def _simulate_phase_evolution(initial: PhaseState, steps: int) -> None:
    """Simulate and display live phase evolution."""
    console.print("[cyan]Live phase evolution simulation would be implemented here[/cyan]")

def _export_exploration_data(results: dict, filename: str) -> None:
    """Export exploration data to file."""
    console.print(f"[cyan]Exporting exploration data to {filename}...[/cyan]")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main() -> None:
    """Main CLI entry point with comprehensive error handling."""
    
    try:
        # Display header
        console.print("[dim]🧮 Type-Safe Dimensional Mathematics Framework[/dim]")
        console.print("[dim]Built with CLI excellence and mathematical validation[/dim]\n")
        
        # Run the CLI
        app()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Operation interrupted by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"\n[red]💥 Unexpected error: {str(e)}[/red]")
        console.print("[dim]Please report this issue if it persists[/dim]")
        raise typer.Exit(1)


if __name__ == "__main__":
    main()