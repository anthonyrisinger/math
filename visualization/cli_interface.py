#!/usr/bin/env python3
"""
CLI Interface for Modern Visualization
=====================================

Maintains CLI compatibility while eliminating matplotlib dependencies.
Provides command-line access to modern Kingdon and Plotly backends.
"""

from pathlib import Path
from typing import Optional

import click

from .modernized_dashboard import BackendType, ModernDashboard


@click.group()
@click.option(
    "--backend",
    type=click.Choice(["kingdon", "plotly", "auto"]),
    default="auto",
    help="Visualization backend to use",
)
@click.option("--dimension", type=float, default=4.0, help="Starting dimension value")
@click.option("--time", type=float, default=0.0, help="Starting time value")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def viz(ctx, backend: str, dimension: float, time: float, verbose: bool):
    """Modern visualization interface - matplotlib eliminated."""
    ctx.ensure_object(dict)
    ctx.obj["backend"] = backend
    ctx.obj["dimension"] = dimension
    ctx.obj["time"] = time
    ctx.obj["verbose"] = verbose

    if verbose:
        click.echo("🚀 Modern visualization CLI initialized")
        click.echo(f"🔧 Backend: {backend}")
        click.echo(f"📊 Dimension: {dimension}")
        click.echo(f"⏰ Time: {time}")


@viz.command()
@click.option(
    "--layout",
    type=click.Choice(["mathematical_grid", "unified", "default"]),
    default="mathematical_grid",
    help="Dashboard layout",
)
@click.option(
    "--export",
    type=click.Choice(["html", "json", "ganja", "auto"]),
    help="Export format after launch",
)
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
def dashboard(ctx, layout: str, export: Optional[str], output: Optional[str]):
    """Launch modern interactive dashboard."""
    click.echo("🚀 Launching modern dashboard...")

    # Create dashboard with CLI parameters
    dashboard_obj = ModernDashboard(backend=ctx.obj["backend"])
    dashboard_obj.update_parameters(
        dimension=ctx.obj["dimension"], time=ctx.obj["time"]
    )

    # Configure backend-specific options
    if dashboard_obj.backend_type == BackendType.PLOTLY:
        success = dashboard_obj.backend.initialize(layout=layout)
    else:
        success = dashboard_obj.backend.initialize()

    if not success:
        click.echo("❌ Dashboard initialization failed", err=True)
        return

    # Launch dashboard
    result = dashboard_obj.launch()

    if result and export:
        # Export scene if requested
        exported = dashboard_obj.export_scene(export)

        if exported and output:
            output_path = Path(output)
            try:
                if export == "html":
                    output_path.write_text(exported, encoding="utf-8")
                elif export in ["json", "ganja"]:
                    output_path.write_text(str(exported), encoding="utf-8")
                else:
                    output_path.write_bytes(exported)

                click.echo(f"✅ Scene exported to: {output_path}")
            except Exception as e:
                click.echo(f"❌ Export failed: {e}", err=True)

    # Display backend info
    info = dashboard_obj.get_backend_info()
    if ctx.obj["verbose"]:
        click.echo("\n📊 Dashboard Information:")
        for key, value in info.items():
            click.echo(f"   {key}: {value}")


@viz.command()
@click.option(
    "--scene-type",
    type=click.Choice(["landscape", "topology", "morphic", "cascade"]),
    default="landscape",
    help="Scene type to render",
)
@click.option(
    "--export",
    type=click.Choice(["html", "json", "ganja", "auto"]),
    default="auto",
    help="Export format",
)
@click.option(
    "--output", "-o", type=click.Path(), required=True, help="Output file path"
)
@click.pass_context
def render(ctx, scene_type: str, export: str, output: str):
    """Render specific scene to file."""
    click.echo(f"🎨 Rendering {scene_type} scene...")

    dashboard_obj = ModernDashboard(backend=ctx.obj["backend"])
    dashboard_obj.update_parameters(
        dimension=ctx.obj["dimension"], time=ctx.obj["time"]
    )

    # Initialize backend
    if not dashboard_obj.backend.initialize():
        click.echo("❌ Backend initialization failed", err=True)
        return

    # Create scene-specific data
    if scene_type == "landscape":
        scene_data = dashboard_obj._prepare_scene_data()
        # Filter to only landscape data
        scene_data = {
            "geometry": {},
            "topology": {},
            "measures": {"landscape": scene_data["measures"]["landscape"]},
            "parameters": {},
        }
    elif scene_type == "topology":
        scene_data = dashboard_obj._prepare_scene_data()
        scene_data = {
            "geometry": {"topology_view": scene_data["geometry"]["topology_view"]},
            "topology": {},
            "measures": {},
            "parameters": {},
        }
    elif scene_type == "morphic":
        scene_data = dashboard_obj._prepare_scene_data()
        scene_data = {
            "geometry": {"morphic": scene_data["geometry"]["morphic"]},
            "topology": {},
            "measures": {},
            "parameters": {},
        }
    else:  # cascade
        scene_data = dashboard_obj._prepare_scene_data()
        scene_data = {
            "geometry": {},
            "topology": {"cascade": scene_data["topology"]["cascade"]},
            "measures": {},
            "parameters": {},
        }

    # Render scene
    dashboard_obj.backend.render_scene(scene_data)

    # Export
    exported = dashboard_obj.export_scene(export)

    output_path = Path(output)
    try:
        if export == "html":
            output_path.write_text(exported, encoding="utf-8")
        elif export in ["json", "ganja"]:
            output_path.write_text(str(exported), encoding="utf-8")
        else:
            output_path.write_bytes(exported)

        click.echo(f"✅ {scene_type.title()} scene exported to: {output_path}")
    except Exception as e:
        click.echo(f"❌ Export failed: {e}", err=True)


@viz.command()
@click.option(
    "--control-type",
    type=click.Choice(["additive", "multiplicative", "boundary"]),
    required=True,
    help="Control semantic type",
)
@click.option(
    "--value", required=True, help="Control value (JSON format for complex values)"
)
@click.pass_context
def control(ctx, control_type: str, value: str):
    """Apply control operation with semantic validation."""
    click.echo(f"🎮 Applying {control_type} control...")

    # Parse value
    try:
        import json

        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        # Try as simple numeric value
        try:
            parsed_value = float(value)
        except ValueError:
            # Try as boolean
            if value.lower() in ["true", "false"]:
                parsed_value = value.lower() == "true"
            else:
                parsed_value = value  # Keep as string

    # Create dashboard and apply control
    dashboard_obj = ModernDashboard(backend=ctx.obj["backend"])

    if not dashboard_obj.backend.initialize():
        click.echo("❌ Backend initialization failed", err=True)
        return

    success = dashboard_obj.apply_control(control_type, parsed_value)

    if success:
        click.echo("✅ Control applied successfully")
    else:
        click.echo("❌ Control application failed", err=True)


@viz.command()
@click.pass_context
def info(ctx):
    """Display backend and system information."""
    dashboard_obj = ModernDashboard(backend=ctx.obj["backend"])

    click.echo("📊 Modern Visualization System Information")
    click.echo("=" * 50)

    # Backend info
    info_data = dashboard_obj.get_backend_info()
    click.echo(f"Backend: {info_data.get('backend', 'unknown')}")
    click.echo(f"Status: {info_data.get('status', 'unknown')}")

    # Camera configuration
    camera = info_data.get("camera_config", {})
    if camera:
        click.echo(f"Camera projection: {camera.get('projection', 'unknown')}")
        click.echo(f"Aspect ratio: {camera.get('aspect_ratio', 'unknown')}")
        click.echo(
            f"Golden ratio view: {info_data.get('orthographic_view', 'unknown')}"
        )

    # Dependencies check
    click.echo("\n🔍 Dependencies:")

    import importlib.util

    if importlib.util.find_spec("plotly"):
        click.echo("✅ Plotly available")
    else:
        click.echo("❌ Plotly not available")

    if importlib.util.find_spec("kingdon"):
        click.echo("✅ Kingdon available")
    else:
        click.echo("❌ Kingdon not available")

    try:
        import numpy

        click.echo(f"✅ NumPy {numpy.__version__}")
    except ImportError:
        click.echo("❌ NumPy not available")

    try:
        import scipy

        click.echo(f"✅ SciPy {scipy.__version__}")
    except ImportError:
        click.echo("❌ SciPy not available")

    # Matplotlib elimination status
    click.echo("\n💀 Matplotlib Elimination Status:")
    if importlib.util.find_spec("matplotlib"):
        click.echo("⚠️  Matplotlib still present (not used in modern backends)")
    else:
        click.echo("✅ Matplotlib successfully eliminated")


@viz.command()
@click.option(
    "--new-backend",
    type=click.Choice(["kingdon", "plotly"]),
    required=True,
    help="New backend to switch to",
)
@click.pass_context
def switch(ctx, new_backend: str):
    """Switch visualization backend."""
    click.echo(f"🔄 Switching to {new_backend} backend...")

    dashboard_obj = ModernDashboard(backend=ctx.obj["backend"])
    success = dashboard_obj.switch_backend(new_backend)

    if success:
        click.echo(f"✅ Successfully switched to {new_backend}")
        ctx.obj["backend"] = new_backend
    else:
        click.echo(f"❌ Failed to switch to {new_backend}", err=True)


@viz.command()
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["html", "json", "ganja", "auto"]),
    default="auto",
    help="Export format",
)
@click.option(
    "--output", "-o", type=click.Path(), required=True, help="Output file path"
)
@click.pass_context
def export(ctx, export_format: str, output: str):
    """Export current scene data."""
    click.echo(f"📤 Exporting scene in {export_format} format...")

    dashboard_obj = ModernDashboard(backend=ctx.obj["backend"])
    dashboard_obj.update_parameters(
        dimension=ctx.obj["dimension"], time=ctx.obj["time"]
    )

    if not dashboard_obj.backend.initialize():
        click.echo("❌ Backend initialization failed", err=True)
        return

    # Prepare and render scene
    scene_data = dashboard_obj._prepare_scene_data()
    dashboard_obj.backend.render_scene(scene_data)

    # Export
    exported = dashboard_obj.export_scene(export_format)

    output_path = Path(output)
    try:
        if export_format == "html":
            output_path.write_text(exported, encoding="utf-8")
        elif export_format in ["json", "ganja"]:
            output_path.write_text(str(exported), encoding="utf-8")
        else:
            output_path.write_bytes(exported)

        click.echo(f"✅ Scene exported to: {output_path}")
        click.echo(f"📁 File size: {output_path.stat().st_size} bytes")
    except Exception as e:
        click.echo(f"❌ Export failed: {e}", err=True)


def main():
    """CLI entry point."""
    viz()


if __name__ == "__main__":
    main()
