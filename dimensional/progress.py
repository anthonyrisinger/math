#!/usr/bin/env python3
"""Progress indicators and timing utilities for blazing-fast operations."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()


class MicrosecondTimer:
    """High-precision timer for microsecond measurements."""

    def __init__(self):
        self.start_time: float = 0.0
        self.lap_times: list[float] = []

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.lap_times = []

    def lap(self) -> float:
        """Record a lap time and return elapsed microseconds."""
        current = time.perf_counter()
        elapsed = (current - self.start_time) * 1_000_000  # Convert to microseconds
        self.lap_times.append(elapsed)
        return elapsed

    def elapsed(self) -> float:
        """Get elapsed microseconds since start."""
        return (time.perf_counter() - self.start_time) * 1_000_000

    def format_time(self, microseconds: float) -> str:
        """Format microseconds into human-readable string."""
        if microseconds < 1000:
            return f"{microseconds:.1f}μs"
        elif microseconds < 1_000_000:
            return f"{microseconds/1000:.2f}ms"
        else:
            return f"{microseconds/1_000_000:.3f}s"


class PerformanceProgress:
    """Enhanced progress tracking with performance metrics."""

    def __init__(self):
        self.timer = MicrosecondTimer()

    def create_progress(self) -> Progress:
        """Create a customized progress bar with our branding."""
        return Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(bar_width=30, style="green", complete_style="bold green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("[yellow]{task.fields[speed]}[/yellow]"),
            console=console,
            transient=False,
        )

    @contextmanager
    def track_operation(self, description: str, total: int = 100):
        """Context manager for tracking operations with progress."""
        progress = self.create_progress()
        self.timer.start()

        with progress:
            task = progress.add_task(description, total=total, speed="")

            def update(n: int = 1, speed_text: str = ""):
                """Update progress and speed."""
                progress.update(task, advance=n, speed=speed_text)

            yield update

            # Final timing
            elapsed = self.timer.elapsed()
            ops_per_sec = (total / elapsed) * 1_000_000 if elapsed > 0 else 0
            final_speed = f"{ops_per_sec:.0f} ops/sec" if ops_per_sec < 1_000_000 else f"{ops_per_sec/1_000_000:.1f}M ops/sec"
            progress.update(task, speed=final_speed)

            console.print(f"[green]✓ Completed:[/green] {self.timer.format_time(elapsed)} • [yellow]{final_speed}[/yellow]")


def progress_batch(items: list[Any], operation: Callable, description: str = "Processing") -> list[Any]:
    """Process items in batch with progress tracking."""
    results = []
    tracker = PerformanceProgress()
    timer = MicrosecondTimer()

    with tracker.track_operation(description, total=len(items)) as update:
        timer.start()
        for i, item in enumerate(items, 1):
            result = operation(item)
            results.append(result)

            # Update with current speed
            elapsed = timer.elapsed()
            current_rate = (i / elapsed) * 1_000_000 if elapsed > 0 else 0
            if current_rate < 1_000_000:
                speed_text = f"{current_rate:.0f} items/sec"
            else:
                speed_text = f"{current_rate/1_000_000:.1f}M items/sec"

            update(1, speed_text)

    return results


class LiveStatusDisplay:
    """Live status display for real-time updates."""

    def __init__(self):
        self.timer = MicrosecondTimer()
        self.stats: dict[str, Any] = {}

    @contextmanager
    def live_panel(self, title: str = "Processing"):
        """Display live updating panel."""
        self.timer.start()

        def generate_panel() -> Panel:
            elapsed = self.timer.elapsed()

            content = [f"[yellow]Time:[/yellow] {self.timer.format_time(elapsed)}"]

            for key, value in self.stats.items():
                if isinstance(value, float):
                    content.append(f"[cyan]{key}:[/cyan] {value:.6f}")
                else:
                    content.append(f"[cyan]{key}:[/cyan] {value}")

            return Panel("\n".join(content), title=f"[bold blue]{title}[/bold blue]", border_style="green")

        with Live(generate_panel(), refresh_per_second=10, console=console) as live:
            def update_stats(**kwargs):
                """Update displayed statistics."""
                self.stats.update(kwargs)
                live.update(generate_panel())

            yield update_stats


def create_batch_progress() -> Progress:
    """Create a progress bar optimized for batch operations."""
    return Progress(
        SpinnerColumn(spinner_name="bouncingBall", style="bold blue"),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(
            bar_width=40,
            style="cyan",
            complete_style="bold green",
            finished_style="bold green",
        ),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("•"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[yellow]{task.fields[rate]}[/yellow]"),
        console=console,
    )


def measure_performance(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Measure function performance and return result with timing."""
    timer = MicrosecondTimer()
    timer.start()
    result = func(*args, **kwargs)
    elapsed = timer.elapsed()
    return result, elapsed


def format_performance_table(timings: dict[str, float]) -> Table:
    """Create a formatted table of performance timings."""
    table = Table(title="⚡ Performance Metrics", box=None)
    table.add_column("Operation", style="cyan", no_wrap=True)
    table.add_column("Time", style="yellow", justify="right")
    table.add_column("Speed", style="green", justify="right")

    timer = MicrosecondTimer()

    for operation, microseconds in sorted(timings.items(), key=lambda x: x[1]):
        time_str = timer.format_time(microseconds)

        # Calculate operations per second
        if microseconds > 0:
            ops_per_sec = 1_000_000 / microseconds
            if ops_per_sec < 1000:
                speed_str = f"{ops_per_sec:.0f} ops/sec"
            elif ops_per_sec < 1_000_000:
                speed_str = f"{ops_per_sec/1000:.1f}K ops/sec"
            else:
                speed_str = f"{ops_per_sec/1_000_000:.1f}M ops/sec"
        else:
            speed_str = "∞ ops/sec"

        table.add_row(operation, time_str, speed_str)

    return table


class BatchProcessor:
    """High-performance batch processor with progress tracking."""

    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress
        self.timer = MicrosecondTimer()

    def process(
        self,
        items: list[Any],
        operation: Callable,
        description: str = "Processing",
        chunk_size: int = 100
    ) -> list[Any]:
        """Process items in chunks with optimal performance."""
        if not self.show_progress:
            return [operation(item) for item in items]

        results = []
        progress = create_batch_progress()
        self.timer.start()

        with progress:
            task = progress.add_task(
                description,
                total=len(items),
                rate="calculating..."
            )

            for i, item in enumerate(items):
                result = operation(item)
                results.append(result)

                # Update progress and rate
                if (i + 1) % chunk_size == 0 or i == len(items) - 1:
                    elapsed = self.timer.elapsed()
                    rate = ((i + 1) / elapsed) * 1_000_000 if elapsed > 0 else 0

                    if rate < 1000:
                        rate_str = f"{rate:.0f} items/sec"
                    elif rate < 1_000_000:
                        rate_str = f"{rate/1000:.1f}K items/sec"
                    else:
                        rate_str = f"{rate/1_000_000:.2f}M items/sec"

                    progress.update(
                        task,
                        advance=chunk_size if (i + 1) % chunk_size == 0 else (i + 1) % chunk_size,
                        rate=rate_str
                    )

        # Show final performance
        total_time = self.timer.elapsed()
        total_rate = (len(items) / total_time) * 1_000_000 if total_time > 0 else 0

        console.print(
            f"[green]✨ Completed {len(items)} items in {self.timer.format_time(total_time)}[/green] • "
            f"[yellow]{total_rate:.0f} items/sec average[/yellow]"
        )

        return results


def with_spinner(text: str = "Processing..."):
    """Decorator to show spinner during function execution."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with console.status(f"[bold blue]{text}[/bold blue]", spinner="dots"):
                timer = MicrosecondTimer()
                timer.start()
                result = func(*args, **kwargs)
                elapsed = timer.elapsed()
                console.print(f"[green]✓[/green] {text} completed in {timer.format_time(elapsed)}")
                return result
        return wrapper
    return decorator


# Convenience functions
def quick_progress(items: Iterator, description: str = "Processing") -> Iterator:
    """Quick progress wrapper for iterables."""
    from rich.progress import track
    return track(items, description=description, console=console)


def show_performance_summary(operations: list[tuple[str, float]]) -> None:
    """Display a summary of performance metrics."""
    table = Table(title="🚀 Performance Summary", box=None)
    table.add_column("Operation", style="cyan")
    table.add_column("Time", style="yellow", justify="right")
    table.add_column("Relative", style="green", justify="right")

    timer = MicrosecondTimer()

    # Find baseline (fastest operation)
    if operations:
        baseline = min(t for _, t in operations if t > 0)

        for name, microseconds in operations:
            time_str = timer.format_time(microseconds)
            relative = f"{microseconds/baseline:.1f}x" if baseline > 0 else "N/A"
            table.add_row(name, time_str, relative)

        console.print(table)

        # Overall stats
        total = sum(t for _, t in operations)
        average = total / len(operations)
        console.print(
            f"\n[bold]Total:[/bold] {timer.format_time(total)} • "
            f"[bold]Average:[/bold] {timer.format_time(average)}"
        )
