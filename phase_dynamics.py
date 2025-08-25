#!/usr/bin/env python3
"""
Phase Dynamics Visualizer
==========================

Interactive 3D visualization of phase sapping between dimensions.
Shows how higher dimensions literally "feed" on lower ones, creating
the arrow of time and driving dimensional emergence.

Run: python phase_dynamics.py
Controls: Time slider, injection controls, emergence monitoring
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import cm
import matplotlib.animation as animation
from core_measures import DimensionalMeasures, setup_3d_axis, PHI, PI

class PhaseDynamics:
    """Phase sapping dynamics between dimensional levels."""
    
    def __init__(self, max_dimensions=12):
        self.max_dim = max_dimensions
        self.measures = DimensionalMeasures()
        
        # Phase state - complex values representing phase density at each dimension
        self.phase_density = np.zeros(max_dimensions, dtype=complex)
        self.phase_density[0] = 1.0  # Start with unity at the void
        
        # Energy flow matrix - tracks sapping between dimensions
        self.flow_matrix = np.zeros((max_dimensions, max_dimensions))
        
        # Emerged dimensions - those that have reached critical phase
        self.emerged = {0}
        
        # Time and dynamics
        self.time = 0.0
        self.dimension = 0.0  # Current effective dimension
        self.auto_evolve = False
        
        # History for trails
        self.history = []
        
    def phase_capacity(self, d):
        """Phase capacity threshold for dimension d."""
        return self.measures.ball_volume(max(d, 0.01))
    
    def sap_rate(self, source, target):
        """Calculate phase sapping rate from source to target dimension."""
        if source >= target:
            return 0.0
        
        # Deficit in target dimension drives sapping
        capacity = self.phase_capacity(target)
        current = abs(self.phase_density[target])
        deficit = max(0, capacity - current)
        
        # Distance factor with golden ratio regularization
        distance_factor = 1 / (target - source + PHI)
        
        # Frequency ratio - higher dimensions oscillate faster
        frequency_ratio = np.sqrt((target + 1) / (source + 1))
        
        return deficit * distance_factor * frequency_ratio
    
    def evolve_phase(self, dt):
        """Evolve the phase dynamics by time step dt."""
        self.time += dt
        
        # Calculate all sapping rates
        for target in range(1, self.max_dim):
            total_inflow = 0.0
            
            for source in range(target):
                if abs(self.phase_density[source]) > 0.01:
                    rate = self.sap_rate(source, target)
                    
                    # Transfer phase with rotation
                    transfer = rate * abs(self.phase_density[source]) * dt * 0.1
                    transfer = min(transfer, abs(self.phase_density[source]) * 0.5)  # Prevent overdrain
                    
                    if transfer > 0:
                        # Remove from source
                        phase_angle = np.angle(self.phase_density[source])
                        transfer_complex = transfer * np.exp(1j * phase_angle)
                        self.phase_density[source] -= transfer_complex
                        
                        # Add to target with phase rotation
                        rotation = np.exp(1j * PI * target / 6)
                        self.phase_density[target] += transfer_complex * rotation
                        
                        # Track flow for visualization
                        self.flow_matrix[source, target] = transfer
                        total_inflow += transfer
        
        # Decay flow matrix
        self.flow_matrix *= 0.9
        
        # Check for new emergences
        self._check_emergences()
        
        # Update effective dimension
        self._update_dimension()
        
        # Record history
        self._record_history()
    
    def _check_emergences(self):
        """Check if any new dimensions have emerged."""
        for d in range(1, self.max_dim):
            if d not in self.emerged:
                current_phase = abs(self.phase_density[d])
                capacity = self.phase_capacity(d)
                
                if current_phase >= capacity * 0.8:  # 80% threshold
                    self.emerged.add(d)
                    print(f"✨ Dimension {d} emerged! (phase: {current_phase:.3f}, capacity: {capacity:.3f})")
                    
                    # Seed next dimension
                    if d + 1 < self.max_dim:
                        self.phase_density[d + 1] += 0.05 * np.exp(1j * PI / 4)
    
    def _update_dimension(self):
        """Update effective dimension based on phase distribution."""
        # Weighted average of dimensions by phase magnitude
        total_phase = 0.0
        weighted_dim = 0.0
        
        for d in range(self.max_dim):
            phase_mag = abs(self.phase_density[d])
            if phase_mag > 0.01:
                total_phase += phase_mag
                weighted_dim += d * phase_mag
        
        if total_phase > 0:
            self.dimension = weighted_dim / total_phase
    
    def _record_history(self):
        """Record current state for trails."""
        self.history.append({
            'time': self.time,
            'dimension': self.dimension,
            'phase': self.phase_density.copy(),
            'emerged': self.emerged.copy()
        })
        
        # Limit history length
        if len(self.history) > 500:
            self.history = self.history[-400:]
    
    def inject_energy(self, dimension, amount=0.3):
        """Inject phase energy at specified dimension."""
        if 0 <= dimension < self.max_dim:
            # Random phase injection
            random_phase = np.random.random() * 2 * PI
            self.phase_density[dimension] += amount * np.exp(1j * random_phase)
            print(f"💉 Injected {amount:.2f} energy at dimension {dimension}")
    
    def reset(self):
        """Reset to initial state."""
        self.phase_density = np.zeros(self.max_dim, dtype=complex)
        self.phase_density[0] = 1.0
        self.flow_matrix = np.zeros((self.max_dim, self.max_dim))
        self.emerged = {0}
        self.time = 0.0
        self.dimension = 0.0
        self.history = []

class PhaseDynamicsVisualizer:
    """Interactive visualizer for phase dynamics."""
    
    def __init__(self):
        self.dynamics = PhaseDynamics()
        self.anim = None
        
    def create_figure(self):
        """Create interactive figure."""
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('Phase Sapping Dynamics: The Mechanism of Dimensional Emergence',
                         fontsize=14, fontweight='bold')
        
        # Main 3D plot
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        setup_3d_axis(self.ax3d, "Phase Flow Between Dimensions")
        
        # Controls
        self._create_controls()
        
        # Initial plot
        self.update_plot()
        
        print("PHASE DYNAMICS VISUALIZER")
        print("=" * 50)
        print("CONTROLS:")
        print("• Time slider: Manual time evolution")
        print("• Auto button: Automatic evolution")
        print("• Inject buttons: Add energy to specific dimensions")
        print("• Reset: Return to initial state")
        print("\nWATCH:")
        print("• Blue bars: Phase magnitude at each dimension")
        print("• Red arrows: Energy flow between dimensions")
        print("• Gold markers: Emerged dimensions")
        print("• Phase trails show dimensional evolution")
        
    def _create_controls(self):
        """Create interactive controls."""
        # Time slider
        ax_time = plt.axes([0.15, 0.02, 0.4, 0.03])
        self.time_slider = Slider(ax_time, 'Time', 0, 50, valinit=0)
        self.time_slider.on_changed(self.on_time_change)
        
        # Control buttons
        ax_auto = plt.axes([0.6, 0.02, 0.06, 0.03])
        ax_reset = plt.axes([0.67, 0.02, 0.06, 0.03])
        ax_inject0 = plt.axes([0.74, 0.02, 0.04, 0.03])
        ax_inject1 = plt.axes([0.79, 0.02, 0.04, 0.03])
        ax_inject2 = plt.axes([0.84, 0.02, 0.04, 0.03])
        ax_inject3 = plt.axes([0.89, 0.02, 0.04, 0.03])
        ax_inject4 = plt.axes([0.94, 0.02, 0.04, 0.03])
        
        self.btn_auto = Button(ax_auto, 'Auto')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_inject0 = Button(ax_inject0, 'E0')
        self.btn_inject1 = Button(ax_inject1, 'E1')
        self.btn_inject2 = Button(ax_inject2, 'E2')
        self.btn_inject3 = Button(ax_inject3, 'E3')
        self.btn_inject4 = Button(ax_inject4, 'E4')
        
        self.btn_auto.on_clicked(self.toggle_auto)
        self.btn_reset.on_clicked(self.reset)
        self.btn_inject0.on_clicked(lambda x: self.inject_energy(0))
        self.btn_inject1.on_clicked(lambda x: self.inject_energy(1))
        self.btn_inject2.on_clicked(lambda x: self.inject_energy(2))
        self.btn_inject3.on_clicked(lambda x: self.inject_energy(3))
        self.btn_inject4.on_clicked(lambda x: self.inject_energy(4))
        
    def on_time_change(self, val):
        """Handle manual time change."""
        # Evolve to target time
        target_time = val
        while self.dynamics.time < target_time:
            self.dynamics.evolve_phase(0.1)
        self.update_plot()
        
    def toggle_auto(self, event):
        """Toggle automatic evolution."""
        self.dynamics.auto_evolve = not self.dynamics.auto_evolve
        self.btn_auto.label.set_text('Stop' if self.dynamics.auto_evolve else 'Auto')
        
        if self.dynamics.auto_evolve and self.anim is None:
            self.anim = animation.FuncAnimation(self.fig, self.animate_frame,
                                               interval=50, blit=False)
        elif not self.dynamics.auto_evolve and self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
        
    def animate_frame(self, frame):
        """Animation frame update."""
        if self.dynamics.auto_evolve:
            self.dynamics.evolve_phase(0.05)
            self.time_slider.set_val(self.dynamics.time)
            self.update_plot()
        
    def reset(self, event):
        """Reset dynamics."""
        self.dynamics.reset()
        self.time_slider.set_val(0)
        self.update_plot()
        
    def inject_energy(self, dim):
        """Inject energy at dimension."""
        self.dynamics.inject_energy(dim)
        self.update_plot()
        
    def update_plot(self):
        """Update the 3D visualization."""
        self.ax3d.clear()
        setup_3d_axis(self.ax3d, "Phase Flow Between Dimensions")
        
        # Position dimensions in a spiral around the origin
        max_vis_dim = min(10, self.dynamics.max_dim)
        
        positions = []
        for d in range(max_vis_dim):
            # Spiral positioning
            theta = 2 * PI * d / max_vis_dim
            radius = 1 + d * 0.2
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = 0
            positions.append((x, y, z))
        
        # Draw phase bars
        for d in range(max_vis_dim):
            x, y, z = positions[d]
            phase_mag = abs(self.dynamics.phase_density[d])
            
            if phase_mag > 0.01:
                # Vertical bar showing phase magnitude
                bar_height = phase_mag * 3
                self.ax3d.plot([x, x], [y, y], [z, z + bar_height],
                              'b-', linewidth=6, alpha=0.7)
                
                # Top marker
                color = 'gold' if d in self.dynamics.emerged else 'cyan'
                size = 150 if d in self.dynamics.emerged else 100
                self.ax3d.scatter([x], [y], [z + bar_height], 
                                 c=color, s=size, marker='o',
                                 edgecolors='black', linewidth=1)
                
                # Dimension label
                self.ax3d.text(x, y, z + bar_height + 0.2, f'{d}',
                              fontsize=10, ha='center', va='bottom')
                
                # Phase capacity reference (gray line)
                capacity = self.dynamics.phase_capacity(d)
                cap_height = capacity * 3
                self.ax3d.plot([x - 0.1, x + 0.1], [y, y], [z + cap_height, z + cap_height],
                              'gray', linewidth=2, alpha=0.5)
        
        # Draw energy flows
        for source in range(max_vis_dim):
            for target in range(source + 1, min(source + 3, max_vis_dim)):
                flow = self.dynamics.flow_matrix[source, target]
                if flow > 0.001:
                    x1, y1, z1 = positions[source]
                    x2, y2, z2 = positions[target]
                    
                    # Get heights
                    h1 = abs(self.dynamics.phase_density[source]) * 3
                    h2 = abs(self.dynamics.phase_density[target]) * 3
                    
                    # Arrow from source to target
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    mid_z = (z1 + h1 + z2 + h2) / 2
                    
                    # Flow strength determines alpha and width
                    alpha = min(1.0, flow * 50)
                    width = max(1, min(3, int(flow * 100)))
                    
                    self.ax3d.plot([x1, mid_x, x2], 
                                  [y1, mid_y, y2],
                                  [z1 + h1, mid_z, z2 + h2],
                                  'r-', alpha=alpha, linewidth=width)
                    
                    # Arrow head at target
                    self.ax3d.scatter([x2], [y2], [z2 + h2],
                                     c='red', marker='>', s=50, alpha=alpha)
        
        # Draw phase trail (dimensional evolution)
        if len(self.dynamics.history) > 10:
            # Extract dimensional trajectory
            times = [h['time'] for h in self.dynamics.history[-50:]]
            dims = [h['dimension'] for h in self.dynamics.history[-50:]]
            
            # Convert to 3D coordinates
            trail_coords = []
            for dim in dims:
                # Position based on effective dimension
                int_dim = int(dim) % max_vis_dim
                frac = dim - int(dim)
                
                if int_dim < max_vis_dim - 1:
                    x1, y1, z1 = positions[int_dim]
                    x2, y2, z2 = positions[min(int_dim + 1, max_vis_dim - 1)]
                    
                    # Interpolate
                    x = x1 + frac * (x2 - x1)
                    y = y1 + frac * (y2 - y1)
                    z = z1 + frac * (z2 - z1) + 2  # Elevated trail
                else:
                    x, y, z = positions[int_dim]
                    z += 2
                
                trail_coords.append((x, y, z))
            
            if len(trail_coords) > 1:
                trail_x, trail_y, trail_z = zip(*trail_coords)
                self.ax3d.plot(trail_x, trail_y, trail_z,
                              'gold', linewidth=3, alpha=0.8,
                              label=f'Dimensional Evolution')
        
        # Current state info
        info_text = (f"Time: {self.dynamics.time:.1f}\n"
                    f"Effective Dim: {self.dynamics.dimension:.2f}\n"
                    f"Emerged: {sorted(list(self.dynamics.emerged))}")
        
        self.ax3d.text2D(0.02, 0.98, info_text, transform=self.ax3d.transAxes,
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Set axis limits
        self.ax3d.set_xlim([-3, 3])
        self.ax3d.set_ylim([-3, 3])
        self.ax3d.set_zlim([0, 8])
        
        self.ax3d.set_xlabel('X Position')
        self.ax3d.set_ylabel('Y Position')
        self.ax3d.set_zlabel('Phase Magnitude')
        
        self.fig.canvas.draw_idle()
    
    def run(self):
        """Run the interactive visualization."""
        self.create_figure()
        plt.show()

def main():
    """Launch the phase dynamics visualizer."""
    visualizer = PhaseDynamicsVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()