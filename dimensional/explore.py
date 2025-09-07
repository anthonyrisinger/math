"""
Interactive exploration of dimensional mathematics.
Beautiful visualizations and interactive discovery tools.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.optimize import minimize_scalar

from .core import PI, c, gamma, r, s, v

console = Console()


def explore(d, show_plot=True, save_html=None):
    """
    Explore a specific dimension with beautiful visualizations.

    Args:
        d: Dimension to explore (can be fractional!)
        show_plot: Display interactive plot
        save_html: Path to save HTML plot

    Returns:
        Dictionary with all computed values and insights
    """
    console.print(f"\n[bold cyan]🔮 Exploring Dimension {d}[/bold cyan]\n")

    # Compute all measures
    volume = v(d)
    surface = s(d)
    complexity = c(d)
    ratio = r(d)
    gamma_val = gamma(d/2) if d > 0 else None

    # Create beautiful table
    table = Table(title=f"Dimension {d} Properties", box=box.ROUNDED)
    table.add_column("Measure", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_column("Insight", style="yellow")

    # Add rows with insights
    table.add_row(
        "Volume",
        f"{volume:.6e}" if volume < 0.01 else f"{volume:.6f}",
        "🔵 Unit ball volume"
    )
    table.add_row(
        "Surface",
        f"{surface:.6e}" if surface < 0.01 else f"{surface:.6f}",
        "🟢 Sphere surface area"
    )
    table.add_row(
        "Complexity",
        f"{complexity:.6e}" if complexity < 0.01 else f"{complexity:.6f}",
        "🔴 V × S measure"
    )
    table.add_row(
        "Ratio",
        f"{ratio:.6f}",
        "⚡ Surface/Volume"
    )

    console.print(table)

    # Find nearby interesting dimensions
    interesting = _find_interesting_nearby(d)
    if interesting:
        console.print("\n[bold green]🎯 Nearby interesting dimensions:[/bold green]")
        for dim, reason in interesting:
            console.print(f"  • d={dim:.3f}: {reason}")

    # Create interactive plot if requested
    if show_plot:
        fig = _create_exploration_plot(d)
        if save_html:
            fig.write_html(save_html)
            console.print(f"\n[green]✅ Plot saved to {save_html}[/green]")
        fig.show()

    return {
        "dimension": d,
        "volume": volume,
        "surface": surface,
        "complexity": complexity,
        "ratio": ratio,
        "gamma": gamma_val,
        "interesting_nearby": interesting
    }


def instant(d_range=(0.1, 20), points=500):
    """
    Instant visualization of dimensional measures across a range.
    Creates a beautiful multi-panel interactive plot.
    """
    console.print("\n[bold magenta]⚡ Instant Dimensional Visualization[/bold magenta]\n")

    # Generate dimension range
    dims = np.linspace(d_range[0], d_range[1], points)

    # Compute all measures
    with console.status("[cyan]Computing measures...[/cyan]"):
        volumes = v(dims)
        surfaces = s(dims)
        complexities = c(dims)
        ratios = r(dims)

    # Create beautiful subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Volume", "Surface Area", "Complexity", "Ratio"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Volume plot
    fig.add_trace(
        go.Scatter(x=dims, y=volumes, mode='lines', name='Volume',
                   line=dict(color='#3498db', width=3),
                   hovertemplate='d=%{x:.2f}<br>V=%{y:.3e}'),
        row=1, col=1
    )

    # Surface plot
    fig.add_trace(
        go.Scatter(x=dims, y=surfaces, mode='lines', name='Surface',
                   line=dict(color='#2ecc71', width=3),
                   hovertemplate='d=%{x:.2f}<br>S=%{y:.3e}'),
        row=1, col=2
    )

    # Complexity plot
    fig.add_trace(
        go.Scatter(x=dims, y=complexities, mode='lines', name='Complexity',
                   line=dict(color='#e74c3c', width=3),
                   hovertemplate='d=%{x:.2f}<br>C=%{y:.3e}'),
        row=2, col=1
    )

    # Ratio plot
    fig.add_trace(
        go.Scatter(x=dims, y=ratios, mode='lines', name='Ratio',
                   line=dict(color='#f39c12', width=3),
                   hovertemplate='d=%{x:.2f}<br>R=%{y:.3f}'),
        row=2, col=2
    )

    # Update layout for beauty
    fig.update_layout(
        title=dict(
            text="<b>Dimensional Mathematics Explorer</b>",
            font=dict(size=24)
        ),
        showlegend=False,
        height=800,
        hovermode='x unified',
        template='plotly_dark'
    )

    # Log scale for first 3 plots
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=2, col=1)

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#444')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#444')

    console.print("[green]✨ Interactive plot ready![/green]")
    fig.show()

    return fig


def lab(start_d=4.0):
    """
    Interactive laboratory for exploring dimensional mathematics.
    A fun, visual environment for discovery.
    """
    console.print(Panel.fit(
        "[bold cyan]🧪 Dimensional Mathematics Laboratory[/bold cyan]\n\n"
        "Welcome to the interactive exploration environment!\n"
        "Discover the beautiful patterns of high-dimensional geometry.",
        border_style="cyan"
    ))

    current_d = start_d
    history = []

    while True:
        console.print(f"\n[bold]Current dimension: {current_d:.3f}[/bold]")

        # Show mini analysis
        vol = v(current_d)
        surf = s(current_d)
        console.print(f"  Volume: {vol:.3e} | Surface: {surf:.3e}")

        # Interactive menu
        console.print("\n[cyan]Commands:[/cyan]")
        console.print("  [e] Explore current dimension")
        console.print("  [p] Find peaks")
        console.print("  [j] Jump to dimension")
        console.print("  [+/-] Increment/decrement by 0.1")
        console.print("  [r] Random interesting dimension")
        console.print("  [h] Show history")
        console.print("  [q] Quit lab")

        choice = console.input("\n[yellow]Choose action: [/yellow]").lower()

        if choice == 'e':
            explore(current_d)
            history.append(current_d)
        elif choice == 'p':
            peaks()
            console.print("\n[green]Found peaks![/green]")
        elif choice == 'j':
            try:
                new_d = float(console.input("[yellow]Enter dimension: [/yellow]"))
                current_d = new_d
                history.append(current_d)
            except ValueError:
                console.print("[red]Invalid dimension![/red]")
        elif choice == '+':
            current_d += 0.1
        elif choice == '-':
            current_d = max(0.1, current_d - 0.1)
        elif choice == 'r':
            current_d = _get_random_interesting()
            console.print(f"[magenta]Jumped to interesting dimension: {current_d:.3f}[/magenta]")
            history.append(current_d)
        elif choice == 'h':
            if history:
                console.print("\n[cyan]Exploration history:[/cyan]")
                for d in history[-10:]:
                    console.print(f"  • {d:.3f}")
            else:
                console.print("[yellow]No history yet![/yellow]")
        elif choice == 'q':
            console.print("[green]Thanks for exploring! 🚀[/green]")
            break
        else:
            console.print("[red]Unknown command![/red]")

    return history


def peaks():
    """
    Find and visualize peaks in dimensional measures.
    Returns the most interesting dimensions.
    """
    console.print("\n[bold green]🏔️ Finding Dimensional Peaks[/bold green]\n")

    results = {}

    # Find volume peak
    with console.status("[cyan]Finding volume peak...[/cyan]"):
        vol_peak = minimize_scalar(lambda d: -v(d), bounds=(1, 10), method='bounded')
        results['volume'] = (-vol_peak.fun, vol_peak.x)

    # Find surface peak
    with console.status("[cyan]Finding surface peak...[/cyan]"):
        surf_peak = minimize_scalar(lambda d: -s(d), bounds=(1, 12), method='bounded')
        results['surface'] = (-surf_peak.fun, surf_peak.x)

    # Find complexity peak
    with console.status("[cyan]Finding complexity peak...[/cyan]"):
        comp_peak = minimize_scalar(lambda d: -c(d), bounds=(1, 15), method='bounded')
        results['complexity'] = (-comp_peak.fun, comp_peak.x)

    # Create results table
    table = Table(title="Dimensional Peaks", box=box.DOUBLE_EDGE)
    table.add_column("Measure", style="cyan", no_wrap=True)
    table.add_column("Peak Dimension", style="yellow")
    table.add_column("Peak Value", style="green")

    for measure, (value, dimension) in results.items():
        table.add_row(
            measure.capitalize(),
            f"{dimension:.4f}",
            f"{value:.4e}" if value < 0.01 else f"{value:.4f}"
        )

    console.print(table)

    # Create visualization of peaks
    dims = np.linspace(0.1, 20, 1000)
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(
        x=dims, y=v(dims), mode='lines', name='Volume',
        line=dict(color='#3498db', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dims, y=s(dims), mode='lines', name='Surface',
        line=dict(color='#2ecc71', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dims, y=c(dims), mode='lines', name='Complexity',
        line=dict(color='#e74c3c', width=2)
    ))

    # Mark peaks
    for measure, (value, dimension) in results.items():
        color = {'volume': '#3498db', 'surface': '#2ecc71', 'complexity': '#e74c3c'}[measure]
        fig.add_trace(go.Scatter(
            x=[dimension], y=[value],
            mode='markers',
            name=f'{measure.capitalize()} Peak',
            marker=dict(size=15, color=color, symbol='star'),
            showlegend=False
        ))

    fig.update_layout(
        title="<b>Dimensional Peaks</b>",
        xaxis_title="Dimension",
        yaxis_title="Value (log scale)",
        yaxis_type="log",
        template='plotly_dark',
        height=600,
        hovermode='x'
    )

    console.print("\n[green]✨ Interactive peak visualization ready![/green]")
    fig.show()

    return results


def _create_exploration_plot(d):
    """Create beautiful exploration plot for a specific dimension."""
    # Create range around the dimension
    d_min = max(0.1, d - 2)
    d_max = d + 2
    dims = np.linspace(d_min, d_max, 200)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Volume around d={d:.2f}",
            f"Surface around d={d:.2f}",
            f"Complexity around d={d:.2f}",
            f"Ratio around d={d:.2f}"
        )
    )

    # Add traces with current dimension marked
    measures = [
        (v, 'Volume', '#3498db', 1, 1),
        (s, 'Surface', '#2ecc71', 1, 2),
        (c, 'Complexity', '#e74c3c', 2, 1),
        (r, 'Ratio', '#f39c12', 2, 2)
    ]

    for func, name, color, row, col in measures:
        values = func(dims)
        fig.add_trace(
            go.Scatter(x=dims, y=values, mode='lines',
                      line=dict(color=color, width=2),
                      name=name),
            row=row, col=col
        )
        # Mark current dimension
        fig.add_trace(
            go.Scatter(x=[d], y=[func(d)], mode='markers',
                      marker=dict(size=12, color='white',
                                symbol='circle-open', line=dict(width=3)),
                      showlegend=False),
            row=row, col=col
        )

    fig.update_layout(
        title=f"<b>Exploring Dimension {d:.3f}</b>",
        showlegend=False,
        template='plotly_dark',
        height=700
    )

    # Log scale for first 3
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=2, col=1)

    return fig


def _find_interesting_nearby(d, radius=1.0):
    """Find interesting dimensions near a given dimension."""
    interesting = []

    # Check for integer dimensions
    if abs(d - round(d)) < 0.1 and round(d) != d:
        interesting.append((round(d), "Integer dimension"))

    # Check for peaks nearby
    test_points = np.linspace(max(0.1, d-radius), d+radius, 50)
    vol_vals = v(test_points)

    # Local maximum?
    if len(test_points) > 2:
        idx = len(test_points) // 2
        if idx > 0 and idx < len(test_points)-1:
            if vol_vals[idx] > vol_vals[idx-1] and vol_vals[idx] > vol_vals[idx+1]:
                interesting.append((d, "Local volume maximum"))

    # Golden ratio related?
    golden_dims = [1.618, 2.618, 4.236]  # φ, φ+1, φ²
    for gd in golden_dims:
        if abs(d - gd) < 0.2:
            interesting.append((gd, "Golden ratio dimension (φ-related)"))

    # Pi related?
    pi_dims = [PI, 2*PI, PI**2]
    for pd in pi_dims:
        if abs(d - pd) < 0.2:
            interesting.append((pd, "π-related dimension"))

    return interesting[:3]  # Return top 3


def _get_random_interesting():
    """Get a random interesting dimension to explore."""
    interesting_dims = [
        1.0, 2.0, 3.0, 4.0,  # Integer dimensions
        PI, 2*PI,  # Pi-related
        PHI, 2.618,  # Golden ratio
        5.257, 7.257,  # Near peaks
        np.e, np.e**2,  # e-related
    ]
    return np.random.choice(interesting_dims)


# Import PHI for golden ratio
from .core import PHI
