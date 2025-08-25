#!/usr/bin/env python3
"""
3D Visualization Standards
==========================

Standard 3D visualization setup for all mathematical visualizations
in the dimensional emergence framework. Ensures consistent orthographic
projection, golden ratio viewing angles, and unified appearance.

Standard settings:
- Orthographic projection (no perspective distortion)
- Box aspect ratio (1,1,1) for accurate spatial relationships
- View elevation: degrees(φ-1) ≈ 35.4° (golden ratio viewing angle)
- View azimuth: -45° (symmetric diagonal view)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .constants import PHI, VIEW_ELEV, VIEW_AZIM, BOX_ASPECT

def setup_3d_axis(ax, title="", xlim=None, ylim=None, zlim=None,
                  grid=True, grid_alpha=0.3):
    """
    Set up 3D axis with standard orthographic projection and golden viewing angle.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axis object
    title : str
        Axis title
    xlim, ylim, zlim : tuple, optional
        Axis limits as (min, max)
    grid : bool
        Whether to show grid
    grid_alpha : float
        Grid transparency

    Returns
    -------
    Axes3D
        Configured 3D axis
    """
    # Set orthographic projection
    ax.set_proj_type('ortho')

    # Set golden ratio viewing angle
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    # Set box aspect ratio for accurate spatial representation
    ax.set_box_aspect(BOX_ASPECT)

    # Set title
    if title:
        ax.set_title(title, fontsize=12, pad=15)

    # Set limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)

    # Configure grid
    if grid:
        ax.grid(True, alpha=grid_alpha)

    return ax

def create_3d_figure(figsize=(10, 8), dpi=100):
    """
    Create figure with 3D axis using standard settings.

    Parameters
    ----------
    figsize : tuple
        Figure size in inches
    dpi : int
        Dots per inch resolution

    Returns
    -------
    tuple
        (figure, axis) pair
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    return fig, setup_3d_axis(ax)

def set_equal_aspect_3d(ax):
    """
    Set equal aspect ratio for 3D axis.

    Parameters
    ----------
    ax : Axes3D
        3D axis to configure
    """
    # Get current limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # Find the range of each axis
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    zrange = zlim[1] - zlim[0]

    # Find maximum range
    max_range = max(xrange, yrange, zrange)

    # Calculate centers
    xcenter = (xlim[0] + xlim[1]) / 2
    ycenter = (ylim[0] + ylim[1]) / 2
    zcenter = (zlim[0] + zlim[1]) / 2

    # Set equal limits
    half_range = max_range / 2
    ax.set_xlim(xcenter - half_range, xcenter + half_range)
    ax.set_ylim(ycenter - half_range, ycenter + half_range)
    ax.set_zlim(zcenter - half_range, zcenter + half_range)

    return ax

def golden_view_rotation(t, base_elev=VIEW_ELEV, base_azim=VIEW_AZIM, phi=PHI):
    """
    Generate rotation sequence based on golden ratio.

    Creates smooth camera rotation using golden ratio proportions
    for aesthetically pleasing animations.

    Parameters
    ----------
    t : float or array
        Time parameter [0, 1]
    base_elev : float
        Base elevation angle
    base_azim : float
        Base azimuth angle
    phi : float
        Golden ratio

    Returns
    -------
    tuple
        (elevation, azimuth) angles
    """
    t = np.asarray(t)

    # Golden ratio modulation
    elev_variation = 20 * np.sin(2 * np.pi * t / phi)
    azim_variation = 360 * t / phi

    elevation = base_elev + elev_variation
    azimuth = base_azim + azim_variation

    return elevation, azimuth

def standard_colormap(name='viridis'):
    """
    Get standard colormap for consistent visualization.

    Parameters
    ----------
    name : str
        Colormap name

    Returns
    -------
    Colormap
        Matplotlib colormap object
    """
    return plt.cm.get_cmap(name)

def add_coordinate_frame(ax, origin=(0, 0, 0), scale=1.0, alpha=0.7):
    """
    Add coordinate frame (xyz axes) to 3D plot.

    Parameters
    ----------
    ax : Axes3D
        3D axis
    origin : tuple
        Origin point for coordinate frame
    scale : float
        Scale factor for axes length
    alpha : float
        Transparency of axes

    Returns
    -------
    list
        List of axis line objects
    """
    x0, y0, z0 = origin

    # X axis (red)
    x_line = ax.plot([x0, x0 + scale], [y0, y0], [z0, z0],
                     color='red', alpha=alpha, linewidth=2)[0]

    # Y axis (green)
    y_line = ax.plot([x0, x0], [y0, y0 + scale], [z0, z0],
                     color='green', alpha=alpha, linewidth=2)[0]

    # Z axis (blue)
    z_line = ax.plot([x0, x0], [y0, y0], [z0, z0 + scale],
                     color='blue', alpha=alpha, linewidth=2)[0]

    return [x_line, y_line, z_line]

def save_3d_figure(fig, filename, dpi=300, bbox_inches='tight'):
    """
    Save 3D figure with standard settings.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    filename : str
        Output filename
    dpi : int
        Resolution
    bbox_inches : str
        Bounding box setting
    """
    # Ensure directory exists
    import os
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,
                facecolor='white', edgecolor='none')

def create_sphere_wireframe(center=(0, 0, 0), radius=1.0, resolution=20):
    """
    Create wireframe sphere coordinates.

    Parameters
    ----------
    center : tuple
        Sphere center
    radius : float
        Sphere radius
    resolution : int
        Number of wireframe lines

    Returns
    -------
    tuple
        (X, Y, Z) coordinate arrays for wireframe
    """
    # Create sphere coordinates
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution//2)
    U, V = np.meshgrid(u, v)

    X = center[0] + radius * np.cos(U) * np.sin(V)
    Y = center[1] + radius * np.sin(U) * np.sin(V)
    Z = center[2] + radius * np.cos(V)

    return X, Y, Z

def plot_sphere_wireframe(ax, center=(0, 0, 0), radius=1.0,
                          resolution=20, color='gray', alpha=0.3):
    """
    Plot wireframe sphere on 3D axis.

    Parameters
    ----------
    ax : Axes3D
        3D axis
    center : tuple
        Sphere center
    radius : float
        Sphere radius
    resolution : int
        Wireframe resolution
    color : str
        Line color
    alpha : float
        Transparency

    Returns
    -------
    list
        List of plotted line objects
    """
    X, Y, Z = create_sphere_wireframe(center, radius, resolution)

    lines = []

    # Plot longitude lines
    for i in range(X.shape[1]):
        line = ax.plot(X[:, i], Y[:, i], Z[:, i],
                      color=color, alpha=alpha, linewidth=0.5)[0]
        lines.append(line)

    # Plot latitude lines
    for i in range(X.shape[0]):
        line = ax.plot(X[i, :], Y[i, :], Z[i, :],
                      color=color, alpha=alpha, linewidth=0.5)[0]
        lines.append(line)

    return lines

def add_integer_badge(ax, value, tolerance=1e-6, position='upper left'):
    """
    Add integer badge showing value and residual.

    Format: "Value ≈ N (residual=δ)"
    Green if |residual| < tolerance, red otherwise.

    Parameters
    ----------
    ax : Axes3D
        3D axis
    value : float
        Value to display
    tolerance : float
        Tolerance for "good" integer approximation
    position : str
        Badge position

    Returns
    -------
    Text
        Text object for the badge
    """
    # Find nearest integer and residual
    nearest_int = int(round(value))
    residual = abs(value - nearest_int)

    # Format text
    badge_text = f"Value ≈ {nearest_int} (residual={residual:.2e})"

    # Choose color
    color = 'green' if residual < tolerance else 'red'

    # Position mapping
    positions = {
        'upper left': (0.02, 0.98),
        'upper right': (0.98, 0.98),
        'lower left': (0.02, 0.02),
        'lower right': (0.98, 0.02)
    }

    if position in positions:
        x, y = positions[position]
        ha = 'left' if 'left' in position else 'right'
        va = 'top' if 'upper' in position else 'bottom'
    else:
        x, y, ha, va = 0.02, 0.98, 'left', 'top'

    # Add text
    text = ax.text2D(x, y, badge_text, transform=ax.transAxes,
                     fontsize=10, color=color, weight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                     ha=ha, va=va)

    return text

def view_preserving_limits(data_points, margin=0.1):
    """
    Calculate axis limits that preserve the standard view while containing all data.

    Parameters
    ----------
    data_points : array-like
        Points to contain, shape (N, 3)
    margin : float
        Extra margin as fraction of data range

    Returns
    -------
    tuple
        (xlim, ylim, zlim) tuples
    """
    data_points = np.asarray(data_points)

    # Find data ranges
    min_coords = np.min(data_points, axis=0)
    max_coords = np.max(data_points, axis=0)
    ranges = max_coords - min_coords
    centers = (max_coords + min_coords) / 2

    # Add margin
    half_ranges = ranges * (1 + margin) / 2

    # Create symmetric limits around centers
    xlim = (centers[0] - half_ranges[0], centers[0] + half_ranges[0])
    ylim = (centers[1] - half_ranges[1], centers[1] + half_ranges[1])
    zlim = (centers[2] - half_ranges[2], centers[2] + half_ranges[2])

    return xlim, ylim, zlim

class View3DManager:
    """
    Manager class for 3D visualization consistency.

    Ensures all 3D plots follow the same standards and provides
    convenient methods for common 3D visualization tasks.
    """

    def __init__(self):
        self.elev = VIEW_ELEV
        self.azim = VIEW_AZIM
        self.box_aspect = BOX_ASPECT

    def setup_axis(self, ax, **kwargs):
        """Set up axis with standard settings."""
        return setup_3d_axis(ax, **kwargs)

    def create_figure(self, **kwargs):
        """Create figure with standard 3D axis."""
        return create_3d_figure(**kwargs)

    def animate_rotation(self, ax, frames=100, phi=PHI):
        """Create rotation animation."""
        def update(frame):
            t = frame / frames
            elev, azim = golden_view_rotation(t, self.elev, self.azim, phi)
            ax.view_init(elev=elev, azim=azim)
            return []

        return update

if __name__ == "__main__":
    print("3D VISUALIZATION STANDARDS TEST")
    print("=" * 50)

    # Test figure creation
    fig, ax = create_3d_figure(figsize=(8, 6))

    # Add coordinate frame
    add_coordinate_frame(ax)

    # Plot a sphere wireframe
    plot_sphere_wireframe(ax, radius=0.8)

    # Add integer badge
    add_integer_badge(ax, 2.99999, tolerance=1e-4)

    # Set title
    ax.set_title("3D Visualization Standards Test")

    print(f"View elevation: {VIEW_ELEV:.1f}°")
    print(f"View azimuth: {VIEW_AZIM:.1f}°")
    print(f"Box aspect: {BOX_ASPECT}")
    print(f"Golden ratio: {PHI:.6f}")

    # Show plot
    plt.tight_layout()
    plt.show()