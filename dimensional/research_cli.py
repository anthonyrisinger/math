#!/usr/bin/env python3
"""
Enhanced Interactive Research CLI
================================

Advanced research workflow system for dimensional mathematics discovery.
Provides enhanced lab(), explore(), and instant() functions with:
- Research session persistence
- Rich terminal visualizations
- Interactive parameter sweeps
- Publication-quality exports
- Mathematical discovery tools

Designed for researchers exploring "the true shape of one" through
dimensional mathematics and gamma function analysis.
"""

import json
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.tree import Tree

from .gamma import c_peak, s_peak, v_peak
from .mathematics import (
    CRITICAL_DIMENSIONS,
)
from .measures import c, s, v

console = Console()

# Research Session Data Structures
@dataclass
class ResearchPoint:
    """Single research data point with metadata."""
    dimension: float
    volume: float
    surface: float
    complexity: float
    timestamp: datetime
    notes: str = ""
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ParameterSweep:
    """Parameter sweep configuration and results."""
    parameter: str
    start: float
    end: float
    steps: int
    results: list[ResearchPoint] = None
    sweep_id: str = ""

    def __post_init__(self):
        if self.results is None:
            self.results = []
        if not self.sweep_id:
            self.sweep_id = f"sweep_{int(time.time())}"

@dataclass
class ResearchSession:
    """Complete research session with history and exports."""
    session_id: str
    start_time: datetime
    points: list[ResearchPoint]
    sweeps: list[ParameterSweep]
    bookmarks: dict[str, float]
    notes: str
    exports: list[str]

    def __post_init__(self):
        if not hasattr(self, 'points'):
            self.points = []
        if not hasattr(self, 'sweeps'):
            self.sweeps = []
        if not hasattr(self, 'bookmarks'):
            self.bookmarks = {}
        if not hasattr(self, 'exports'):
            self.exports = []

class ResearchPersistence:
    """Handles saving/loading research sessions."""

    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path.home() / ".dimensional" / "research"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_session(self, session: ResearchSession) -> Path:
        """Save research session to file."""
        filepath = self.base_path / f"session_{session.session_id}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(session, f)

        # Also save JSON version for human readability
        json_path = filepath.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(asdict(session), f, indent=2, default=str)

        return filepath

    def load_session(self, session_id: str) -> Optional[ResearchSession]:
        """Load research session from file."""
        filepath = self.base_path / f"session_{session_id}.pkl"
        if not filepath.exists():
            return None

        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def list_sessions(self) -> list[tuple[str, datetime]]:
        """List available research sessions."""
        sessions = []
        for pkl_file in self.base_path.glob("session_*.pkl"):
            try:
                session = pickle.load(open(pkl_file, 'rb'))
                sessions.append((session.session_id, session.start_time))
            except:
                continue
        return sorted(sessions, key=lambda x: x[1], reverse=True)

class RichVisualizer:
    """Enhanced Rich-based mathematical visualizations."""

    def __init__(self):
        self.console = Console()

    def show_dimensional_analysis(self, point: ResearchPoint) -> None:
        """Rich display of dimensional analysis."""
        table = Table(title=f"🔍 Dimensional Analysis at d = {point.dimension:.6f}")
        table.add_column("Measure", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_column("Log Value", style="green")
        table.add_column("Properties", style="magenta")

        measures = [
            ("Volume V(d)", point.volume, "Ball volume"),
            ("Surface S(d)", point.surface, "Sphere surface"),
            ("Complexity C(d)", point.complexity, "V(d) × S(d)"),
        ]

        for name, value, desc in measures:
            log_val = np.log(value) if value > 0 else "undefined"
            log_str = f"{log_val:.6f}" if isinstance(log_val, float) else log_val

            # Add special properties
            props = []
            if "Volume" in name and abs(point.dimension - v_peak()) < 0.01:
                props.append("PEAK")
            if "Surface" in name and abs(point.dimension - s_peak()) < 0.01:
                props.append("PEAK")
            if "Complexity" in name and abs(point.dimension - c_peak()) < 0.01:
                props.append("PEAK")

            table.add_row(
                name,
                f"{value:.8f}",
                log_str,
                " ".join(props) if props else desc
            )

        self.console.print(table)

    def show_parameter_sweep_analysis(self, sweep: ParameterSweep) -> None:
        """Rich display of parameter sweep results."""
        if not sweep.results:
            return

        table = Table(title=f"📊 Parameter Sweep: {sweep.parameter}")
        table.add_column("Step", style="dim")
        table.add_column("Dimension", style="cyan")
        table.add_column("Volume", style="yellow")
        table.add_column("Surface", style="green")
        table.add_column("Complexity", style="magenta")
        table.add_column("Notes", style="blue")

        for i, point in enumerate(sweep.results):
            table.add_row(
                str(i+1),
                f"{point.dimension:.4f}",
                f"{point.volume:.6f}",
                f"{point.surface:.6f}",
                f"{point.complexity:.6f}",
                point.notes[:20] + "..." if len(point.notes) > 20 else point.notes
            )

        self.console.print(table)

        # Show sweep statistics
        dims = [p.dimension for p in sweep.results]
        vols = [p.volume for p in sweep.results]
        surfs = [p.surface for p in sweep.results]
        comps = [p.complexity for p in sweep.results]

        stats_table = Table(title="📈 Sweep Statistics")
        stats_table.add_column("Measure", style="cyan")
        stats_table.add_column("Min", style="green")
        stats_table.add_column("Max", style="red")
        stats_table.add_column("Mean", style="yellow")
        stats_table.add_column("Peak Dimension", style="magenta")

        def find_peak_dim(values, dimensions):
            max_idx = np.argmax(values)
            return dimensions[max_idx]

        stats_table.add_row(
            "Volume",
            f"{min(vols):.6f}",
            f"{max(vols):.6f}",
            f"{np.mean(vols):.6f}",
            f"{find_peak_dim(vols, dims):.6f}"
        )
        stats_table.add_row(
            "Surface",
            f"{min(surfs):.6f}",
            f"{max(surfs):.6f}",
            f"{np.mean(surfs):.6f}",
            f"{find_peak_dim(surfs, dims):.6f}"
        )
        stats_table.add_row(
            "Complexity",
            f"{min(comps):.6f}",
            f"{max(comps):.6f}",
            f"{np.mean(comps):.6f}",
            f"{find_peak_dim(comps, dims):.6f}"
        )

        self.console.print(stats_table)

    def show_critical_dimensions_tree(self) -> None:
        """Show critical dimensions in a tree structure."""
        tree = Tree("🌟 Critical Dimensions")

        # Known mathematical constants
        constants_branch = tree.add("📐 Mathematical Constants")
        constants_branch.add(f"π = {np.pi:.8f}")
        constants_branch.add(f"e = {np.e:.8f}")
        constants_branch.add(f"φ = {(1 + np.sqrt(5))/2:.8f} (Golden ratio)")

        # Peak dimensions
        peaks_branch = tree.add("🏔️ Peak Dimensions")
        v_peak_dim, v_peak_val = v_peak()
        s_peak_dim, s_peak_val = s_peak()
        c_peak_dim, c_peak_val = c_peak()

        peaks_branch.add(f"Volume peak: d = {v_peak_dim:.8f}, V = {v_peak_val:.8f}")
        peaks_branch.add(f"Surface peak: d = {s_peak_dim:.8f}, S = {s_peak_val:.8f}")
        peaks_branch.add(f"Complexity peak: d = {c_peak_dim:.8f}, C = {c_peak_val:.8f}")

        # Known critical dimensions
        critical_branch = tree.add("🔥 Critical Dimensions")
        for name, value in CRITICAL_DIMENSIONS.items():
            critical_branch.add(f"{name}: d = {value:.8f}")

        self.console.print(tree)

    def show_session_overview(self, session: ResearchSession) -> None:
        """Show research session overview."""
        panel_content = f"""
🔬 [bold]Research Session: {session.session_id}[/bold]
📅 Started: {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}
📊 Data Points: {len(session.points)}
🔄 Parameter Sweeps: {len(session.sweeps)}
🔖 Bookmarks: {len(session.bookmarks)}
📁 Exports: {len(session.exports)}

📝 Notes: {session.notes[:100]}{'...' if len(session.notes) > 100 else ''}
        """

        self.console.print(Panel(panel_content, title="Research Session", border_style="blue"))


class InteractiveParameterSweep:
    """Interactive parameter sweep with real-time visualization."""

    def __init__(self, visualizer: RichVisualizer):
        self.visualizer = visualizer

    def run_dimension_sweep(self, start: float, end: float, steps: int,
                          notes: str = "") -> ParameterSweep:
        """Run an interactive dimensional parameter sweep."""
        sweep = ParameterSweep(
            parameter="dimension",
            start=start,
            end=end,
            steps=steps
        )

        console.print(f"🔄 Running dimensional sweep: {start} → {end} ({steps} steps)")

        dimensions = np.linspace(start, end, steps)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Computing dimensional measures...", total=steps)

            for i, dim in enumerate(dimensions):
                # Compute measures
                vol = v(dim)
                surf = s(dim)
                comp = c(dim)

                point = ResearchPoint(
                    dimension=dim,
                    volume=vol,
                    surface=surf,
                    complexity=comp,
                    timestamp=datetime.now(),
                    notes=notes
                )

                sweep.results.append(point)
                progress.update(task, advance=1)

        return sweep


class PublicationExporter:
    """Export research results for publication-quality figures."""

    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path.cwd() / "exports"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def export_csv_data(self, sweep: ParameterSweep) -> Path:
        """Export sweep data as CSV for external plotting."""
        filepath = self.base_path / f"sweep_{sweep.sweep_id}.csv"

        with open(filepath, 'w') as f:
            f.write("dimension,volume,surface,complexity,timestamp,notes\n")
            for point in sweep.results:
                f.write(f"{point.dimension},{point.volume},{point.surface},"
                       f"{point.complexity},{point.timestamp.isoformat()},"
                       f'"{point.notes}"\n')

        return filepath

    def export_json_analysis(self, session: ResearchSession) -> Path:
        """Export complete session analysis as JSON."""
        filepath = self.base_path / f"analysis_{session.session_id}.json"

        analysis = {
            "session": asdict(session),
            "summary": {
                "total_points": len(session.points),
                "dimension_range": [
                    min(p.dimension for p in session.points),
                    max(p.dimension for p in session.points)
                ] if session.points else [0, 0],
                "peak_findings": self._analyze_peaks(session),
                "critical_crossings": self._find_critical_crossings(session)
            }
        }

        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        return filepath

    def _analyze_peaks(self, session: ResearchSession) -> dict:
        """Analyze peaks found in session data."""
        if not session.points:
            return {}

        dims = [p.dimension for p in session.points]
        vols = [p.volume for p in session.points]
        surfs = [p.surface for p in session.points]
        comps = [p.complexity for p in session.points]

        return {
            "volume_peak": {"dimension": dims[np.argmax(vols)], "value": max(vols)},
            "surface_peak": {"dimension": dims[np.argmax(surfs)], "value": max(surfs)},
            "complexity_peak": {"dimension": dims[np.argmax(comps)], "value": max(comps)}
        }

    def _find_critical_crossings(self, session: ResearchSession) -> list[dict]:
        """Find where measures cross critical values."""
        crossings = []

        for critical_name, critical_dim in CRITICAL_DIMENSIONS.items():
            nearby_points = [
                p for p in session.points
                if abs(p.dimension - critical_dim) < 0.1
            ]

            if nearby_points:
                closest_point = min(nearby_points, key=lambda p: abs(p.dimension - critical_dim))
                crossings.append({
                    "critical_dimension": critical_name,
                    "target_dimension": critical_dim,
                    "actual_dimension": closest_point.dimension,
                    "measures": {
                        "volume": closest_point.volume,
                        "surface": closest_point.surface,
                        "complexity": closest_point.complexity
                    }
                })

        return crossings


# Enhanced Research Functions
def enhanced_lab(start_dimension: float = 4.0, session_id: str = None) -> ResearchSession:
    """Enhanced interactive research laboratory with persistence."""

    # Initialize or load session
    persistence = ResearchPersistence()

    if session_id:
        session = persistence.load_session(session_id)
        if not session:
            console.print(f"[red]Session {session_id} not found. Creating new session.[/red]")
            session_id = None

    if not session_id:
        session_id = f"lab_{int(time.time())}"
        session = ResearchSession(
            session_id=session_id,
            start_time=datetime.now(),
            points=[],
            sweeps=[],
            bookmarks={},
            notes="Interactive research lab session",
            exports=[]
        )

    visualizer = RichVisualizer()
    sweeper = InteractiveParameterSweep(visualizer)
    exporter = PublicationExporter()

    console.print(Panel(
        f"🎮 [bold green]Enhanced Research Laboratory[/bold green]\n"
        f"Session: [cyan]{session.session_id}[/cyan]\n"
        f"Starting dimension: [yellow]{start_dimension}[/yellow]\n\n"
        f"Commands: explore <dim>, sweep <start> <end> <steps>, bookmark <name>, "
        f"export, save, load <id>, quit",
        title="Research Lab",
        border_style="green"
    ))

    current_dimension = start_dimension

    while True:
        try:
            command = Prompt.ask(f"[cyan]Lab[{current_dimension:.3f}]>[/cyan]").strip().lower()

            if command == "quit" or command == "exit":
                break
            elif command.startswith("explore "):
                try:
                    dim = float(command.split()[1])
                    current_dimension = dim

                    # Create research point
                    point = ResearchPoint(
                        dimension=dim,
                        volume=v(dim),
                        surface=s(dim),
                        complexity=c(dim),
                        timestamp=datetime.now()
                    )

                    session.points.append(point)
                    visualizer.show_dimensional_analysis(point)

                except (ValueError, IndexError):
                    console.print("[red]Usage: explore <dimension>[/red]")

            elif command.startswith("sweep "):
                try:
                    parts = command.split()
                    start = float(parts[1])
                    end = float(parts[2])
                    steps = int(parts[3]) if len(parts) > 3 else 50

                    sweep = sweeper.run_dimension_sweep(start, end, steps)
                    session.sweeps.append(sweep)
                    visualizer.show_parameter_sweep_analysis(sweep)

                except (ValueError, IndexError):
                    console.print("[red]Usage: sweep <start> <end> [steps][/red]")

            elif command.startswith("bookmark "):
                try:
                    name = command.split()[1]
                    session.bookmarks[name] = current_dimension
                    console.print(f"[green]Bookmarked dimension {current_dimension:.6f} as '{name}'[/green]")
                except IndexError:
                    console.print("[red]Usage: bookmark <name>[/red]")

            elif command == "bookmarks":
                if session.bookmarks:
                    table = Table(title="🔖 Bookmarks")
                    table.add_column("Name", style="cyan")
                    table.add_column("Dimension", style="yellow")

                    for name, dim in session.bookmarks.items():
                        table.add_row(name, f"{dim:.8f}")
                    console.print(table)
                else:
                    console.print("[yellow]No bookmarks yet[/yellow]")

            elif command == "export":
                if session.sweeps:
                    for sweep in session.sweeps:
                        csv_path = exporter.export_csv_data(sweep)
                        session.exports.append(str(csv_path))
                        console.print(f"[green]Exported sweep to {csv_path}[/green]")

                json_path = exporter.export_json_analysis(session)
                session.exports.append(str(json_path))
                console.print(f"[green]Exported analysis to {json_path}[/green]")

            elif command == "save":
                filepath = persistence.save_session(session)
                console.print(f"[green]Session saved to {filepath}[/green]")

            elif command.startswith("load "):
                try:
                    load_id = command.split()[1]
                    loaded_session = persistence.load_session(load_id)
                    if loaded_session:
                        session = loaded_session
                        console.print(f"[green]Loaded session {load_id}[/green]")
                        visualizer.show_session_overview(session)
                    else:
                        console.print(f"[red]Session {load_id} not found[/red]")
                except IndexError:
                    console.print("[red]Usage: load <session_id>[/red]")

            elif command == "sessions":
                sessions = persistence.list_sessions()
                if sessions:
                    table = Table(title="💾 Available Sessions")
                    table.add_column("Session ID", style="cyan")
                    table.add_column("Start Time", style="yellow")

                    for sess_id, start_time in sessions:
                        table.add_row(sess_id, start_time.strftime('%Y-%m-%d %H:%M:%S'))
                    console.print(table)
                else:
                    console.print("[yellow]No saved sessions found[/yellow]")

            elif command == "critical":
                visualizer.show_critical_dimensions_tree()

            elif command == "overview":
                visualizer.show_session_overview(session)

            elif command == "help":
                console.print("""
[bold]Enhanced Research Lab Commands:[/bold]
  explore <dim>     - Analyze dimension in detail
  sweep <start> <end> [steps] - Parameter sweep
  bookmark <name>   - Save current dimension
  bookmarks        - Show all bookmarks
  critical         - Show critical dimensions
  export           - Export data for publication
  save             - Save session
  load <id>        - Load session
  sessions         - List available sessions
  overview         - Show session overview
  help             - Show this help
  quit/exit        - Exit lab
                """)

            else:
                console.print(f"[red]Unknown command: {command}. Type 'help' for commands.[/red]")

        except KeyboardInterrupt:
            if Confirm.ask("\n[yellow]Save session before exiting?[/yellow]"):
                persistence.save_session(session)
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    return session


def enhanced_explore(dimension: float, context: str = "general") -> dict[str, Any]:
    """Enhanced exploration with guided discovery paths."""

    visualizer = RichVisualizer()

    console.print(f"🔍 [bold cyan]Enhanced Exploration: Dimension {dimension}[/bold cyan]")

    # Create research point
    point = ResearchPoint(
        dimension=dimension,
        volume=v(dimension),
        surface=s(dimension),
        complexity=c(dimension),
        timestamp=datetime.now(),
        notes=f"Exploration in {context} context"
    )

    # Show main analysis
    visualizer.show_dimensional_analysis(point)

    # Guided discovery paths
    discovery_paths = {
        "peaks": _explore_peak_proximity(dimension),
        "ratios": _explore_dimensional_ratios(dimension),
        "derivatives": _explore_derivative_analysis(dimension),
        "neighborhoods": _explore_local_neighborhood(dimension),
        "critical": _explore_critical_proximity(dimension)
    }

    console.print("\n🎯 [bold]Guided Discovery Paths:[/bold]")
    for path_name, path_data in discovery_paths.items():
        if path_data["significance"] > 0.5:  # Only show significant paths
            console.print(f"  • {path_name.title()}: {path_data['description']}")

    return {
        "point": asdict(point),
        "discovery_paths": discovery_paths,
        "recommendations": _generate_exploration_recommendations(dimension, discovery_paths)
    }


def enhanced_instant(configuration: str = "research") -> dict[str, Any]:
    """Enhanced instant analysis with multiple view configurations."""

    configurations = {
        "research": {
            "panels": ["dimensional_measures", "peak_analysis", "critical_dimensions", "ratio_analysis"],
            "focus": "comprehensive research overview"
        },
        "peaks": {
            "panels": ["peak_locations", "peak_values", "peak_derivatives", "peak_neighborhoods"],
            "focus": "peak analysis and characterization"
        },
        "discovery": {
            "panels": ["anomaly_detection", "pattern_analysis", "trend_identification", "hypothesis_generation"],
            "focus": "mathematical discovery and pattern recognition"
        },
        "publication": {
            "panels": ["summary_statistics", "key_findings", "publication_plots", "export_ready_data"],
            "focus": "publication-ready analysis and visualization"
        }
    }

    if configuration not in configurations:
        configuration = "research"

    config = configurations[configuration]
    visualizer = RichVisualizer()

    console.print("⚡ [bold red]Enhanced Instant Analysis[/bold red]")
    console.print(f"Configuration: [yellow]{configuration}[/yellow] - {config['focus']}")

    # Generate instant analysis
    results = {}

    for panel in config["panels"]:
        if panel == "dimensional_measures":
            results[panel] = _instant_dimensional_measures()
        elif panel == "peak_analysis":
            results[panel] = _instant_peak_analysis()
        elif panel == "critical_dimensions":
            results[panel] = _instant_critical_analysis()
        elif panel == "ratio_analysis":
            results[panel] = _instant_ratio_analysis()
        # Add more panels as needed

    # Display results with Rich formatting
    for panel_name, panel_data in results.items():
        _display_instant_panel(panel_name, panel_data, visualizer)

    return results


# Helper functions for discovery paths and instant analysis
def _explore_peak_proximity(dimension: float) -> dict[str, Any]:
    """Analyze proximity to known peaks."""
    # Peak functions return dimensions only
    v_peak_dim = v_peak()
    s_peak_dim = s_peak()
    c_peak_dim = c_peak()

    distances = {
        "volume_peak": abs(dimension - v_peak_dim),
        "surface_peak": abs(dimension - s_peak_dim),
        "complexity_peak": abs(dimension - c_peak_dim)
    }

    min_distance = min(distances.values())
    closest_peak = min(distances, key=distances.get)

    return {
        "significance": 1.0 / (1.0 + min_distance),  # Closer = more significant
        "closest_peak": closest_peak,
        "distance": min_distance,
        "description": f"Close to {closest_peak} (distance: {min_distance:.6f})"
    }

def _explore_dimensional_ratios(dimension: float) -> dict[str, Any]:
    """Analyze dimensional ratios and relationships."""
    vol = v(dimension)
    surf = s(dimension)
    comp = c(dimension)

    ratios = {
        "volume_to_surface": vol / surf if surf != 0 else float('inf'),
        "complexity_to_volume": comp / vol if vol != 0 else float('inf'),
        "complexity_to_surface": comp / surf if surf != 0 else float('inf')
    }

    # Check for interesting ratio values
    interesting_ratios = []
    for ratio_name, ratio_value in ratios.items():
        if abs(ratio_value - np.pi) < 0.01:
            interesting_ratios.append(f"{ratio_name} ≈ π")
        elif abs(ratio_value - np.e) < 0.01:
            interesting_ratios.append(f"{ratio_name} ≈ e")
        elif abs(ratio_value - (1 + np.sqrt(5))/2) < 0.01:
            interesting_ratios.append(f"{ratio_name} ≈ φ")

    significance = len(interesting_ratios) * 0.3  # More interesting ratios = higher significance

    return {
        "significance": min(significance, 1.0),
        "ratios": ratios,
        "interesting_ratios": interesting_ratios,
        "description": f"Found {len(interesting_ratios)} interesting mathematical ratios"
    }

def _explore_derivative_analysis(dimension: float) -> dict[str, Any]:
    """Analyze derivatives and rate of change."""
    h = 1e-8  # Small step for numerical differentiation

    # Numerical derivatives
    v_derivative = (v(dimension + h) - v(dimension - h)) / (2 * h)
    s_derivative = (s(dimension + h) - s(dimension - h)) / (2 * h)
    c_derivative = (c(dimension + h) - c(dimension - h)) / (2 * h)

    derivatives = {
        "volume": v_derivative,
        "surface": s_derivative,
        "complexity": c_derivative
    }

    # Check for critical points (derivative ≈ 0)
    critical_points = []
    for measure, derivative in derivatives.items():
        if abs(derivative) < 1e-6:
            critical_points.append(measure)

    significance = len(critical_points) * 0.4  # Critical points are significant

    return {
        "significance": min(significance, 1.0),
        "derivatives": derivatives,
        "critical_points": critical_points,
        "description": f"Found {len(critical_points)} measures at critical points"
    }

def _explore_local_neighborhood(dimension: float, radius: float = 0.1) -> dict[str, Any]:
    """Explore local neighborhood around dimension."""
    points = []
    test_dims = np.linspace(dimension - radius, dimension + radius, 21)

    for test_dim in test_dims:
        if test_dim > 0:  # Only positive dimensions
            points.append({
                "dimension": test_dim,
                "volume": v(test_dim),
                "surface": s(test_dim),
                "complexity": c(test_dim)
            })

    if not points:
        return {"significance": 0, "description": "No valid neighborhood points"}

    # Analyze neighborhood properties
    volumes = [p["volume"] for p in points]
    surfaces = [p["surface"] for p in points]
    complexities = [p["complexity"] for p in points]

    vol_variance = np.var(volumes)
    surf_variance = np.var(surfaces)
    comp_variance = np.var(complexities)

    # High variance indicates interesting behavior
    total_variance = vol_variance + surf_variance + comp_variance
    significance = min(np.log10(total_variance + 1) / 10, 1.0)

    return {
        "significance": significance,
        "points": points,
        "variances": {
            "volume": vol_variance,
            "surface": surf_variance,
            "complexity": comp_variance
        },
        "description": f"Neighborhood analysis shows variance {total_variance:.6f}"
    }

def _explore_critical_proximity(dimension: float) -> dict[str, Any]:
    """Check proximity to critical mathematical dimensions."""
    critical_distances = {}
    for name, critical_dim in CRITICAL_DIMENSIONS.items():
        critical_distances[name] = abs(dimension - critical_dim)

    min_distance = min(critical_distances.values())
    closest_critical = min(critical_distances, key=critical_distances.get)

    significance = 1.0 / (1.0 + min_distance * 10)  # Closer = more significant

    return {
        "significance": significance,
        "closest_critical": closest_critical,
        "distance": min_distance,
        "all_distances": critical_distances,
        "description": f"Close to critical dimension {closest_critical} (distance: {min_distance:.6f})"
    }

def _generate_exploration_recommendations(dimension: float, discovery_paths: dict) -> list[str]:
    """Generate personalized exploration recommendations."""
    recommendations = []

    # Sort paths by significance
    sorted_paths = sorted(discovery_paths.items(), key=lambda x: x[1]["significance"], reverse=True)

    for path_name, path_data in sorted_paths[:3]:  # Top 3 most significant
        if path_data["significance"] > 0.3:
            if path_name == "peaks":
                recommendations.append(f"Explore dimensions near {path_data['closest_peak']}")
            elif path_name == "critical":
                recommendations.append(f"Investigate {path_data['closest_critical']} critical behavior")
            elif path_name == "derivatives":
                recommendations.append("Analyze critical point behavior with derivative analysis")
            elif path_name == "ratios":
                recommendations.append("Study mathematical constant relationships in ratios")

    return recommendations

def _instant_dimensional_measures() -> dict:
    """Generate instant dimensional measures analysis."""
    key_dimensions = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    results = []
    for dim in key_dimensions:
        results.append({
            "dimension": dim,
            "volume": v(dim),
            "surface": s(dim),
            "complexity": c(dim)
        })

    return {"key_dimensions": results}

def _instant_peak_analysis() -> dict:
    """Generate instant peak analysis."""
    # Peak functions return single values (dimensions), we need to compute values
    v_peak_dim = v_peak()
    s_peak_dim = s_peak()
    c_peak_dim = c_peak()

    # Compute the actual values at peak dimensions
    v_peak_val = v(v_peak_dim)
    s_peak_val = s(s_peak_dim)
    c_peak_val = c(c_peak_dim)

    return {
        "volume_peak": {"dimension": v_peak_dim, "value": v_peak_val},
        "surface_peak": {"dimension": s_peak_dim, "value": s_peak_val},
        "complexity_peak": {"dimension": c_peak_dim, "value": c_peak_val}
    }

def _instant_critical_analysis() -> dict:
    """Generate instant critical dimensions analysis."""
    critical_analysis = {}

    for name, dim in CRITICAL_DIMENSIONS.items():
        critical_analysis[name] = {
            "dimension": dim,
            "volume": v(dim),
            "surface": s(dim),
            "complexity": c(dim)
        }

    return critical_analysis

def _instant_ratio_analysis() -> dict:
    """Generate instant ratio analysis."""
    test_dimensions = [1.0, 2.0, 3.0, 4.0, 5.0]

    ratio_analysis = []
    for dim in test_dimensions:
        vol = v(dim)
        surf = s(dim)
        comp = c(dim)

        ratio_analysis.append({
            "dimension": dim,
            "volume_surface_ratio": vol / surf if surf != 0 else float('inf'),
            "complexity_volume_ratio": comp / vol if vol != 0 else float('inf'),
            "surface_complexity_ratio": surf / comp if comp != 0 else float('inf')
        })

    return {"ratio_analysis": ratio_analysis}

def _display_instant_panel(panel_name: str, panel_data: dict, visualizer: RichVisualizer):
    """Display instant analysis panel with Rich formatting."""
    console.print(f"\n📊 [bold]{panel_name.replace('_', ' ').title()}[/bold]")

    if panel_name == "dimensional_measures":
        table = Table()
        table.add_column("Dimension", style="cyan")
        table.add_column("Volume", style="yellow")
        table.add_column("Surface", style="green")
        table.add_column("Complexity", style="magenta")

        for result in panel_data["key_dimensions"]:
            table.add_row(
                f"{result['dimension']:.1f}",
                f"{result['volume']:.6f}",
                f"{result['surface']:.6f}",
                f"{result['complexity']:.6f}"
            )
        console.print(table)

    elif panel_name == "peak_analysis":
        table = Table(title="🏔️ Peak Analysis")
        table.add_column("Measure", style="cyan")
        table.add_column("Peak Dimension", style="yellow")
        table.add_column("Peak Value", style="green")

        table.add_row("Volume", f"{panel_data['volume_peak']['dimension']:.8f}",
                     f"{panel_data['volume_peak']['value']:.8f}")
        table.add_row("Surface", f"{panel_data['surface_peak']['dimension']:.8f}",
                     f"{panel_data['surface_peak']['value']:.8f}")
        table.add_row("Complexity", f"{panel_data['complexity_peak']['dimension']:.8f}",
                     f"{panel_data['complexity_peak']['value']:.8f}")

        console.print(table)

    # Add more panel display logic as needed


# Backwards compatibility - enhanced versions replace originals
lab = enhanced_lab
explore = enhanced_explore
instant = enhanced_instant
