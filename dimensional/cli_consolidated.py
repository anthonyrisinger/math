#!/usr/bin/env python3
"""
Consolidated CLI for Dimensional Mathematics
============================================

Clean, idiomatic command-line interface using Typer.
All commands delegate to core business logic.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.table import Table

# Import core functionality
from . import (
    ball_volume,
    complexity_measure,
    digamma,
    explore,
    gamma,
    gammaln,
    peaks,
    sphere_surface,
)

# Initialize
app = typer.Typer(help="Dimensional Mathematics CLI")
console = Console()

# Command Groups
compute_app = typer.Typer(help="Compute dimensional measures")
analyze_app = typer.Typer(help="Analysis and exploration")
visual_app = typer.Typer(help="Visualization and plotting")

app.add_typer(compute_app, name="compute")
app.add_typer(analyze_app, name="analyze")
app.add_typer(visual_app, name="visual")


# ============= Direct Commands (shortcuts) =============

@app.command("v")
def volume_shortcut(d: float = typer.Argument(..., help="Dimension")):
    """Quick volume calculation V(d)"""
    result = ball_volume(d)
    console.print(f"V({d}) = {result:.6f}")


@app.command("s")
def surface_shortcut(d: float = typer.Argument(..., help="Dimension")):
    """Quick surface calculation S(d)"""
    result = sphere_surface(d)
    console.print(f"S({d}) = {result:.6f}")


@app.command("c")
def complexity_shortcut(d: float = typer.Argument(..., help="Dimension")):
    """Quick complexity calculation C(d)"""
    result = complexity_measure(d)
    console.print(f"C({d}) = {result:.6f}")


@app.command("g")
def gamma_shortcut(z: float = typer.Argument(..., help="Input value")):
    """Quick gamma calculation Γ(z)"""
    result = gamma(z)
    console.print(f"Γ({z}) = {result:.6f}")


# ============= Compute Commands =============

@compute_app.command("volume")
def compute_volume(
    dimensions: str = typer.Argument(..., help="Dimensions (comma-separated or range)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file")
):
    """Compute ball volume for given dimensions"""
    dims = _parse_dimensions(dimensions)
    results = ball_volume(dims)

    if output:
        _save_results(output, {"dimensions": dims.tolist(), "volume": results.tolist()})
    else:
        _display_results("Volume", dims, results)


@compute_app.command("surface")
def compute_surface(
    dimensions: str = typer.Argument(..., help="Dimensions"),
    output: Optional[Path] = typer.Option(None, "--output", "-o")
):
    """Compute sphere surface for given dimensions"""
    dims = _parse_dimensions(dimensions)
    results = sphere_surface(dims)

    if output:
        _save_results(output, {"dimensions": dims.tolist(), "surface": results.tolist()})
    else:
        _display_results("Surface", dims, results)


@compute_app.command("all")
def compute_all(
    dimensions: str = typer.Argument(..., help="Dimensions"),
    output: Optional[Path] = typer.Option(None, "--output", "-o")
):
    """Compute all measures (volume, surface, complexity, ratio)"""
    dims = _parse_dimensions(dimensions)

    # Compute all measures
    from .measures import batch_measures
    results = batch_measures(dims, validate=False)

    if output:
        data = {"dimensions": dims.tolist()}
        data.update({k: v.tolist() for k, v in results.items()})
        _save_results(output, data)
    else:
        # Display table
        table = Table(title="Dimensional Measures", box=box.ROUNDED)
        table.add_column("Dimension", style="cyan")
        table.add_column("Volume", style="green")
        table.add_column("Surface", style="yellow")
        table.add_column("Complexity", style="magenta")
        table.add_column("Ratio", style="blue")

        for i, d in enumerate(dims):
            table.add_row(
                f"{d:.2f}",
                f"{results['volume'][i]:.6e}",
                f"{results['surface'][i]:.6e}",
                f"{results['complexity'][i]:.6e}",
                f"{results['ratio'][i]:.6e}"
            )
        console.print(table)


@compute_app.command("gamma")
def compute_gamma(
    values: str = typer.Argument(..., help="Values (comma-separated)"),
    log: bool = typer.Option(False, "--log", help="Compute log-gamma"),
    digamma_opt: bool = typer.Option(False, "--digamma", help="Compute digamma")
):
    """Compute gamma and related functions"""
    vals = _parse_dimensions(values)

    if log:
        results = gammaln(vals)
        name = "log-Γ"
    elif digamma_opt:
        results = digamma(vals)
        name = "ψ"
    else:
        results = gamma(vals)
        name = "Γ"

    _display_results(name, vals, results)


# ============= Analysis Commands =============

@analyze_app.command("peaks")
def find_peaks():
    """Find peaks in all dimensional measures"""
    peak_data = peaks()

    table = Table(title="Dimensional Peaks", box=box.ROUNDED)
    table.add_column("Measure", style="cyan")
    table.add_column("Peak Dimension", style="yellow")
    table.add_column("Peak Value", style="green")

    for measure in ["volume", "surface", "complexity"]:
        if measure in peak_data:
            d, val = peak_data[measure]
            table.add_row(measure.capitalize(), f"{d:.4f}", f"{val:.6e}")

    console.print(table)


@analyze_app.command("explore")
def explore_dimension(
    dimension: float = typer.Argument(4.0, help="Dimension to explore"),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Use caching")
):
    """Explore a specific dimension"""
    result = explore(dimension, use_cache=cache)

    console.print(f"\n[bold cyan]Dimension {dimension}[/bold cyan]")
    for key, value in result.items():
        if value is not None:
            if isinstance(value, float):
                console.print(f"  {key}: {value:.6e}")
            else:
                console.print(f"  {key}: {value}")


@analyze_app.command("convergence")
def analyze_convergence(
    start: float = typer.Option(1.0, "--start", "-s", help="Start dimension"),
    end: float = typer.Option(100.0, "--end", "-e", help="End dimension"),
    measure: str = typer.Option("volume", "--measure", "-m", help="Measure to analyze")
):
    """Analyze convergence behavior"""
    from .gamma import convergence_diagnostics

    result = convergence_diagnostics(
        d_range=np.arange(start, end, 1),
        measure=measure
    )

    console.print(f"\n[bold]Convergence Analysis: {measure}[/bold]")
    console.print(f"  Convergence dimension: {result.get('convergence_dimension', 'N/A')}")
    console.print(f"  Decay rate: {result.get('decay_rate', 'N/A')}")
    console.print(f"  Final value: {result.get('final_value', 'N/A'):.2e}")
    console.print(f"  Max at d={result.get('max_dimension', 'N/A')}: {result.get('max_value', 'N/A'):.2e}")


# ============= Visualization Commands =============

@visual_app.command("plot")
def plot_measures(
    start: float = typer.Option(0.1, "--start", "-s"),
    end: float = typer.Option(20.0, "--end", "-e"),
    points: int = typer.Option(200, "--points", "-n"),
    measures: str = typer.Option("all", "--measures", "-m", help="Measures to plot")
):
    """Plot dimensional measures"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[red]Error: matplotlib not installed[/red]")
        raise typer.Exit(1)

    dims = np.linspace(start, end, points)

    # Parse measures
    if measures == "all":
        plot_list = ["volume", "surface", "complexity", "ratio"]
    else:
        plot_list = measures.split(",")

    fig, axes = plt.subplots(len(plot_list), 1, figsize=(10, 3*len(plot_list)))
    if len(plot_list) == 1:
        axes = [axes]

    for ax, measure in zip(axes, plot_list):
        if measure == "volume":
            values = ball_volume(dims, validate=False)
            color = "blue"
        elif measure == "surface":
            values = sphere_surface(dims, validate=False)
            color = "green"
        elif measure == "complexity":
            values = complexity_measure(dims, validate=False)
            color = "red"
        elif measure == "ratio":
            from .measures import ratio_measure
            values = ratio_measure(dims, validate=False)
            color = "purple"
        else:
            continue

        ax.plot(dims, values, color=color, linewidth=2)
        ax.set_xlabel("Dimension")
        ax.set_ylabel(measure.capitalize())
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


@visual_app.command("dashboard")
def dashboard():
    """Launch interactive dashboard (if available)"""
    try:
        from .visualization.modernized_dashboard import launch_dashboard
        launch_dashboard()
    except ImportError:
        console.print("[yellow]Dashboard not available. Install visualization extras.[/yellow]")
        raise typer.Exit(1)


# ============= Utility Functions =============

def _parse_dimensions(dims_str: str) -> np.ndarray:
    """Parse dimension string (e.g., '1,2,3' or '1:10:0.5')"""
    if ":" in dims_str:
        # Range format: start:end:step
        parts = dims_str.split(":")
        if len(parts) == 2:
            return np.arange(float(parts[0]), float(parts[1]), 1.0)
        elif len(parts) == 3:
            return np.arange(float(parts[0]), float(parts[1]), float(parts[2]))
    elif "," in dims_str:
        # List format
        return np.array([float(d.strip()) for d in dims_str.split(",")])
    else:
        # Single value
        return np.array([float(dims_str)])


def _display_results(name: str, dims: np.ndarray, results: np.ndarray):
    """Display results in a nice format"""
    if len(dims) == 1:
        console.print(f"{name}({dims[0]}) = {results[0]:.6f}")
    else:
        table = Table(title=f"{name} Values", box=box.SIMPLE)
        table.add_column("Input", style="cyan")
        table.add_column("Result", style="green")

        for d, r in zip(dims, results):
            table.add_row(f"{d:.2f}", f"{r:.6e}")
        console.print(table)


def _save_results(path: Path, data: dict):
    """Save results to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    console.print(f"[green]Results saved to {path}[/green]")


# ============= Main Entry Point =============

def main():
    """Main CLI entry point"""
    app()


if __name__ == "__main__":
    main()
