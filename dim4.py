#!/usr/bin/env python3
"""
Clear 3D visualization of dimensional emergence
Using orthographic projection with golden ratio viewing angle
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import gamma

# Constants from your framework
PHI = (1 + np.sqrt(5)) / 2
VARPI = gamma(0.25)**2 / (2 * np.sqrt(2 * np.pi))
PI = np.pi

# Viewing angles
ELEV = np.degrees(PHI - 1)  # Golden ratio elevation ≈ 36.87°
AZIM = -45

def n_ball_volume(d):
    """Volume of unit d-ball"""
    if d == 0:
        return 1.0
    return PI**(d/2) / gamma(d/2 + 1)

def n_sphere_surface(d):
    """Surface area of unit d-sphere"""
    if d <= 0:
        return 2.0 if d == 0 else 0
    return 2 * PI**(d/2) / gamma(d/2)

def create_clear_visualization():
    """
    Simple, clear visualization showing:
    1. How dimensions emerge from phase capacity
    2. The volume/surface peaks
    3. The π boundaries
    """
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Dimensional Emergence - Orthographic View (φ-1)° = {ELEV:.1f}°', 
                 fontsize=14, fontweight='bold')
    
    # ========== LEFT: 3D Phase Space ==========
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_proj_type('ortho')
    ax1.view_init(elev=ELEV, azim=AZIM)
    ax1.set_box_aspect((1, 1, 1))
    
    # Create dimension points in 3D
    # X-axis: dimension number
    # Y-axis: volume
    # Z-axis: surface area
    
    dims = np.linspace(0, 12, 100)
    volumes = np.array([n_ball_volume(d) for d in dims])
    surfaces = np.array([n_sphere_surface(d) for d in dims])
    
    # Plot the trajectory
    ax1.plot(dims, volumes, surfaces, 'b-', linewidth=2, alpha=0.8)
    
    # Mark special points
    special_points = {
        'Start (d=0)': (0, 1, 2),
        'Peak Volume (d≈5.26)': (5.256, n_ball_volume(5.256), n_sphere_surface(5.256)),
        'Peak Surface (d≈7.26)': (7.256, n_ball_volume(7.256), n_sphere_surface(7.256)),
        'π-boundary': (PI, n_ball_volume(PI), n_sphere_surface(PI)),
        '2π-boundary': (2*PI, n_ball_volume(2*PI), n_sphere_surface(2*PI)),
    }
    
    for label, (d, v, s) in special_points.items():
        ax1.scatter([d], [v], [s], s=100, alpha=0.9)
        ax1.text(d, v, s, f'\n{label}', fontsize=8)
    
    # Add π boundary planes
    d_range = np.linspace(0, 12, 20)
    v_range = np.linspace(0, 6, 20)
    
    # π-plane
    D, V = np.meshgrid([PI, PI], v_range)
    S = np.ones_like(D) * 10
    ax1.plot_surface(D, V, S, alpha=0.1, color='red')
    
    # 2π-plane
    D2, V2 = np.meshgrid([2*PI, 2*PI], v_range)
    S2 = np.ones_like(D2) * 10
    ax1.plot_surface(D2, V2, S2, alpha=0.1, color='orange')
    
    ax1.set_xlabel('Dimension', fontsize=10)
    ax1.set_ylabel('Volume', fontsize=10)
    ax1.set_zlabel('Surface Area', fontsize=10)
    ax1.set_title('Phase Space Trajectory')
    
    # ========== RIGHT: Multiple 2D Projections ==========
    
    # Top right: Volume and Surface vs Dimension
    ax2 = fig.add_subplot(322)
    ax2.plot(dims, volumes, 'b-', linewidth=2, label='Volume')
    ax2.plot(dims, surfaces, 'g-', linewidth=2, label='Surface')
    ax2.axvline(x=5.256, color='b', linestyle='--', alpha=0.5)
    ax2.axvline(x=7.256, color='g', linestyle='--', alpha=0.5)
    ax2.axvline(x=PI, color='r', linestyle='--', alpha=0.5, label='π')
    ax2.axvline(x=2*PI, color='orange', linestyle='--', alpha=0.5, label='2π')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Measure')
    ax2.set_title('Volume & Surface Area')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 10])
    
    # Middle right: Phase Capacity
    ax3 = fig.add_subplot(324)
    dims_int = range(0, 13)
    capacities = [n_ball_volume(d) for d in dims_int]
    colors = ['red' if d <= PI else 'orange' if d <= 2*PI else 'gray' for d in dims_int]
    bars = ax3.bar(dims_int, capacities, color=colors, alpha=0.7)
    ax3.set_xlabel('Dimension (integer)')
    ax3.set_ylabel('Phase Capacity Λ(d)')
    ax3.set_title('Phase Capacity by Dimension')
    ax3.grid(True, alpha=0.3)
    
    # Bottom right: Special Dimensions
    ax4 = fig.add_subplot(326)
    special_d = {
        'd=0': 0,
        'd=1/2': 0.5,
        'd=1': 1,
        'd=φ': PHI,
        'd=e': np.e,
        'd=π': PI,
        'd=5.26': 5.256,
        'd=2π': 2*PI,
        'd=7.26': 7.256,
    }
    
    special_v = [n_ball_volume(d) for d in special_d.values()]
    x_pos = range(len(special_d))
    
    bars = ax4.bar(x_pos, special_v, alpha=0.7, color='purple')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(special_d.keys(), rotation=45, ha='right')
    ax4.set_ylabel('Volume')
    ax4.set_title('Fractional Dimensions (Your Predictions)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, special_v):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Print analysis
    print("\n📊 Key Insights from Visualization:")
    print(f"   • Volume peaks at d = 5.256")
    print(f"   • Surface peaks at d = 7.256")
    print(f"   • π-boundary at d = {PI:.3f} marks stability limit")
    print(f"   • 2π-boundary at d = {2*PI:.3f} marks compression limit")
    print(f"   • Your d=2π prediction: V(2π) = {n_ball_volume(2*PI):.3f}")
    print(f"   • Viewing angle: φ-1 = {PHI-1:.5f} radians = {ELEV:.1f}°")
    
    plt.show()

def create_phase_sapping_view():
    """
    Visualize phase sapping as a 3D flow between dimensions
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect((1, 1, 1))
    
    # Create a grid of dimensions
    n_dims = 8
    
    # Position dimensions in a spiral
    positions = []
    for d in range(n_dims):
        angle = 2 * PI * d / n_dims
        r = 2 + d * 0.5
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = d
        positions.append([x, y, z])
    
    positions = np.array(positions)
    
    # Simulate phase densities
    phase_density = np.exp(-np.arange(n_dims) / 3)
    
    # Draw dimensions as spheres with size proportional to phase
    for d in range(n_dims):
        size = 500 * phase_density[d]
        color = plt.cm.viridis(d / n_dims)
        ax.scatter(positions[d, 0], positions[d, 1], positions[d, 2],
                  s=size, c=[color], alpha=0.7, edgecolors='black', linewidth=2)
        ax.text(positions[d, 0], positions[d, 1], positions[d, 2],
               f'  d={d}\n  ρ={phase_density[d]:.2f}', fontsize=9)
    
    # Draw phase sapping arrows
    for d in range(1, n_dims):
        for source in range(d):
            if phase_density[source] > 0.1:
                # Arrow from source to target
                arrow_start = positions[source]
                arrow_end = positions[d]
                arrow_vec = arrow_end - arrow_start
                
                # Scale arrow by phase transfer rate
                transfer_rate = phase_density[source] / (d - source + 1)
                
                ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                         arrow_vec[0]*0.3, arrow_vec[1]*0.3, arrow_vec[2]*0.3,
                         color='red', alpha=transfer_rate*2, arrow_length_ratio=0.2)
    
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Dimension', fontsize=10)
    ax.set_title(f'Phase Sapping: Higher Dimensions Consume Lower\n'
                f'View: {ELEV:.1f}° × {AZIM}°', fontsize=12)
    
    # Add legend
    ax.text2D(0.02, 0.98, 
             'Larger spheres = more phase\n'
             'Red arrows = phase transfer\n'
             'Higher dims drain lower ones',
             transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("CLEAR ORTHOGRAPHIC VISUALIZATION")
    print(f"Golden Ratio View: elevation = {ELEV:.1f}°")
    print(f"VARPI = {VARPI:.5f}")
    print("="*60)
    
    # Main visualization
    create_clear_visualization()
    
    # Phase sapping visualization
    print("\n🌀 Phase Sapping Mechanism:")
    create_phase_sapping_view()