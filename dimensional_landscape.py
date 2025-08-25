#!/usr/bin/env python3
"""
Dimensional Landscape Explorer
==============================

Interactive 3D visualization of how geometric measures change with dimension.
Shows the fundamental landscape where dimension creates reality.

Run: python dimensional_landscape.py
Controls: Dimension slider, rotation controls, measure toggles
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib import cm
import matplotlib.animation as animation
from core_measures import DimensionalMeasures, setup_3d_axis, print_critical_info

class DimensionalLandscape:
    """Interactive 3D landscape of dimensional measures."""
    
    def __init__(self, d_max=12, resolution=1000):
        self.d_max = d_max
        self.resolution = resolution
        self.measures = DimensionalMeasures()
        
        # Compute the landscape
        self.d_range = np.linspace(0.01, d_max, resolution)
        self.volumes = np.array([self.measures.ball_volume(d) for d in self.d_range])
        self.surfaces = np.array([self.measures.sphere_surface(d) for d in self.d_range])
        self.complexity = self.volumes * self.surfaces
        self.ratios = self.surfaces / np.maximum(self.volumes, 1e-10)
        
        # Find critical points
        self.crits = self.measures.critical_dimensions()
        
        # Current state
        self.current_d = 4.0  # Start at our universe
        self.show_volume = True
        self.show_surface = True
        self.show_complexity = True
        self.show_critical = True
        
        # Animation
        self.animating = False
        self.anim = None
        
    def create_figure(self):
        """Create the interactive figure."""
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('Dimensional Landscape: Where Geometry Meets Reality', 
                         fontsize=14, fontweight='bold')
        
        # Main 3D plot
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        setup_3d_axis(self.ax3d, "The Fundamental Geometric Landscape")
        
        # Control panels
        self._create_controls()
        
        # Initial plot
        self.update_plot()
        
        print_critical_info()
        print("\nINTERACTIVE CONTROLS:")
        print("• Dimension slider: Explore different dimensional values")
        print("• Checkboxes: Toggle different measures on/off")
        print("• Auto button: Animate through dimensional space")
        print("• Reset button: Return to d=4 (our universe)")
        
    def _create_controls(self):
        """Create all interactive controls."""
        # Dimension slider
        ax_dim = plt.axes([0.15, 0.02, 0.5, 0.03])
        self.dim_slider = Slider(ax_dim, 'Dimension', 0.1, self.d_max, 
                                valinit=self.current_d, valfmt='%.2f')
        self.dim_slider.on_changed(self.on_dimension_change)
        
        # Measure toggles
        ax_checks = plt.axes([0.02, 0.4, 0.12, 0.2])
        self.checkboxes = CheckButtons(ax_checks, 
                                     ['Volume', 'Surface', 'Complexity', 'Critical'],
                                     [self.show_volume, self.show_surface, 
                                      self.show_complexity, self.show_critical])
        self.checkboxes.on_clicked(self.on_toggle_measure)
        
        # Control buttons
        ax_auto = plt.axes([0.7, 0.02, 0.08, 0.03])
        ax_reset = plt.axes([0.8, 0.02, 0.08, 0.03])
        
        self.btn_auto = Button(ax_auto, 'Auto')
        self.btn_reset = Button(ax_reset, 'Reset')
        
        self.btn_auto.on_clicked(self.toggle_animation)
        self.btn_reset.on_clicked(self.reset_view)
        
    def on_dimension_change(self, val):
        """Handle dimension slider change."""
        self.current_d = val
        self.update_plot()
        
    def on_toggle_measure(self, label):
        """Handle measure toggle."""
        if label == 'Volume':
            self.show_volume = not self.show_volume
        elif label == 'Surface':
            self.show_surface = not self.show_surface
        elif label == 'Complexity':
            self.show_complexity = not self.show_complexity
        elif label == 'Critical':
            self.show_critical = not self.show_critical
        self.update_plot()
        
    def toggle_animation(self, event):
        """Toggle automatic animation."""
        self.animating = not self.animating
        self.btn_auto.label.set_text('Stop' if self.animating else 'Auto')
        
        if self.animating and self.anim is None:
            self.anim = animation.FuncAnimation(self.fig, self.animate_frame,
                                               interval=50, blit=False)
        elif not self.animating and self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
            
    def animate_frame(self, frame):
        """Animation frame update."""
        if self.animating:
            # Slowly cycle through dimensions
            new_d = (self.current_d + 0.02) % self.d_max
            if new_d < 0.1:
                new_d = 0.1
            self.dim_slider.set_val(new_d)
        
    def reset_view(self, event):
        """Reset to default view."""
        self.current_d = 4.0
        self.dim_slider.set_val(self.current_d)
        self.update_plot()
        
    def update_plot(self):
        """Update the 3D plot with current settings."""
        self.ax3d.clear()
        setup_3d_axis(self.ax3d, "The Fundamental Geometric Landscape")
        
        # Create parametric surface showing all three measures
        phi = np.linspace(0, 2*np.pi, 60)
        
        # Volume surface (inner)
        if self.show_volume:
            V_phi = np.outer(self.volumes, np.ones_like(phi))
            D_phi = np.outer(self.d_range, np.ones_like(phi))
            X_v = D_phi
            Y_v = V_phi * np.cos(phi) * 0.3  # Scale for visibility
            Z_v = V_phi * np.sin(phi) * 0.3
            
            self.ax3d.plot_surface(X_v, Y_v, Z_v, 
                                  facecolors=cm.Blues(0.3 * np.ones_like(X_v)), 
                                  alpha=0.4, linewidth=0, antialiased=True)
        
        # Surface area surface (middle)
        if self.show_surface:
            S_phi = np.outer(self.surfaces, np.ones_like(phi))
            D_phi = np.outer(self.d_range, np.ones_like(phi))
            X_s = D_phi
            Y_s = S_phi * np.cos(phi) * 0.15
            Z_s = S_phi * np.sin(phi) * 0.15
            
            self.ax3d.plot_surface(X_s, Y_s, Z_s,
                                  facecolors=cm.Greens(0.5 * np.ones_like(X_s)), 
                                  alpha=0.5, linewidth=0, antialiased=True)
        
        # Complexity surface (outer)
        if self.show_complexity:
            C_phi = np.outer(self.complexity, np.ones_like(phi))
            D_phi = np.outer(self.d_range, np.ones_like(phi))
            X_c = D_phi
            Y_c = C_phi * np.cos(phi) * 0.05  # Smaller scale - complexity is large
            Z_c = C_phi * np.sin(phi) * 0.05
            
            # Color by complexity value
            colors = cm.plasma(C_phi / np.max(C_phi))
            self.ax3d.plot_surface(X_c, Y_c, Z_c,
                                  facecolors=colors, alpha=0.7,
                                  linewidth=0, antialiased=True)
        
        # Critical dimensions
        if self.show_critical:
            # Mark critical points
            critical_dims = [
                (self.crits['complexity_peak'][0], 'V×S Peak', 'gold', '*'),
                (self.crits['pi_boundary'], 'π', 'red', 'o'),
                (self.crits['tau_boundary'], '2π', 'orange', 's'),
                (self.crits['phi_golden'], 'φ', 'green', '^'),
            ]
            
            for d_crit, label, color, marker in critical_dims:
                if d_crit <= self.d_max:
                    v = self.measures.ball_volume(d_crit)
                    s = self.measures.sphere_surface(d_crit)
                    c = v * s
                    
                    # Draw vertical line
                    self.ax3d.plot([d_crit, d_crit], [0, 0], [-c*0.05, c*0.05],
                                  color=color, linestyle='--', alpha=0.6)
                    
                    # Mark at top
                    self.ax3d.scatter([d_crit], [0], [c*0.05], 
                                    c=color, marker=marker, s=100, 
                                    edgecolors='black', linewidth=1)
                    
                    # Label
                    self.ax3d.text(d_crit, 0, c*0.06, f' {label}', 
                                  fontsize=9, color=color)
        
        # Current dimension indicator
        v_current = self.measures.ball_volume(self.current_d)
        s_current = self.measures.sphere_surface(self.current_d)
        c_current = v_current * s_current
        
        # Current position marker
        self.ax3d.scatter([self.current_d], [0], [0], 
                         c='red', marker='o', s=200,
                         edgecolors='black', linewidth=2)
        
        # Draw trajectory showing current values
        trajectory_phi = np.linspace(0, 2*np.pi, 100)
        
        if self.show_volume:
            traj_y = v_current * np.cos(trajectory_phi) * 0.3
            traj_z = v_current * np.sin(trajectory_phi) * 0.3
            self.ax3d.plot([self.current_d] * len(trajectory_phi), traj_y, traj_z,
                          'b-', linewidth=3, alpha=0.8, label=f'V={v_current:.2f}')
        
        if self.show_surface:
            traj_y = s_current * np.cos(trajectory_phi) * 0.15
            traj_z = s_current * np.sin(trajectory_phi) * 0.15
            self.ax3d.plot([self.current_d] * len(trajectory_phi), traj_y, traj_z,
                          'g-', linewidth=3, alpha=0.8, label=f'S={s_current:.2f}')
        
        if self.show_complexity:
            traj_y = c_current * np.cos(trajectory_phi) * 0.05
            traj_z = c_current * np.sin(trajectory_phi) * 0.05
            self.ax3d.plot([self.current_d] * len(trajectory_phi), traj_y, traj_z,
                          'r-', linewidth=4, alpha=0.9, label=f'C={c_current:.2f}')
        
        # Info text
        info_text = f"d = {self.current_d:.2f}\n"
        if v_current > 0:
            info_text += f"V = {v_current:.3f}\nS = {s_current:.3f}\nC = {c_current:.3f}"
        
        self.ax3d.text2D(0.02, 0.98, info_text, transform=self.ax3d.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Set limits based on complexity peak
        max_c = self.crits['complexity_peak'][1]
        self.ax3d.set_xlim([0, self.d_max])
        self.ax3d.set_ylim([-max_c*0.06, max_c*0.06])
        self.ax3d.set_zlim([-max_c*0.06, max_c*0.06])
        
        self.ax3d.set_xlabel('Dimension')
        self.ax3d.set_ylabel('Geometric Y')
        self.ax3d.set_zlabel('Geometric Z')
        
        self.fig.canvas.draw_idle()
    
    def run(self):
        """Run the interactive visualization."""
        self.create_figure()
        plt.show()

def main():
    """Launch the dimensional landscape explorer."""
    print("DIMENSIONAL LANDSCAPE EXPLORER")
    print("=" * 50)
    
    landscape = DimensionalLandscape()
    landscape.run()

if __name__ == "__main__":
    main()