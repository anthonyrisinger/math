#!/usr/bin/env python3
"""
Dimensional Emergence: The Mathematical Truth of Reality

Core Insight: Dimension is not a stage upon which physics happens - 
dimension IS the fundamental parameter from which all physics emerges.
Time itself emerges from dimensional change.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from scipy.special import gamma, gammaln
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

# The constants that govern dimensional emergence
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio: nature's recursive proportion
VARPI = gamma(0.25)**2 / (2 * np.sqrt(2 * np.pi))  # 1.311028777...
PI = np.pi

class DimensionalFramework:
    """
    The complete mathematical framework where dimension generates reality.
    
    Key principles:
    1. Dimension is continuous and primary
    2. Time emerges from dimensional change (not the reverse)
    3. Phase coherence drives dimensional emergence
    4. The V×S product reveals WHERE complexity maximizes
    """
    
    def __init__(self):
        # Dimension is THE fundamental variable
        self.dimension = 0.0  # Start at the void
        
        # Time emerges from dimensional evolution
        self.time = 0.0
        
        # Phase densities for integer dimensional levels
        self.phase_density = np.zeros(16, dtype=complex)
        self.phase_density[0] = 1.0  # Initial unity at the void
        
        # Track which dimensions have emerged (achieved coherence)
        self.emerged = {0}
        
        # Energy flow matrix tracks phase sapping between dimensions
        self.flow_matrix = np.zeros((16, 16))
        
        # History for understanding dynamics
        self.history = []
        
    def n_ball_volume(self, d):
        """
        Volume of unit d-ball: π^(d/2) / Γ(d/2 + 1)
        
        This formula works for ANY real or complex d, revealing that
        dimension itself is fundamentally continuous, not discrete.
        """
        if abs(d) < 1e-10:
            return 1.0  # The void has unit measure
        
        # Use logarithms for numerical stability at high d
        if d > 170:
            log_vol = (d/2) * np.log(PI) - gammaln(d/2 + 1)
            return np.exp(np.real(log_vol))
        
        return PI**(d/2) / gamma(d/2 + 1)
    
    def n_sphere_surface(self, d):
        """
        Surface area of unit (d-1)-sphere in R^d: 2π^(d/2) / Γ(d/2)
        
        The surface represents the boundary between what's inside
        (the bulk) and what's outside - this distinction itself
        creates the notion of dimension.
        """
        if abs(d) < 1e-10:
            return 2.0  # The 0-sphere is two points
        
        if d > 170:
            log_surf = np.log(2) + (d/2) * np.log(PI) - gammaln(d/2)
            return np.exp(np.real(log_surf))
        
        return 2 * PI**(d/2) / gamma(d/2)
    
    def complexity_measure(self, d):
        """
        The V×S product - THIS is the key to understanding dimensional emergence.
        
        This product maximizes around d≈6, revealing WHERE the universe
        has maximum capacity for complexity. It's not coincidence that
        our observable universe is 3+1 dimensional - it sits just below
        this peak in the stable region.
        """
        return self.n_ball_volume(d) * self.n_sphere_surface(d)
    
    def set_dimension(self, new_d):
        """
        Set the dimension directly - this is THE fundamental operation.
        Everything else (time, energy, emergence) flows from this.
        
        Key insight: Increasing dimension creates forward time flow.
        Decreasing dimension causes time to dissipate/reverse.
        """
        old_d = self.dimension
        self.dimension = np.clip(new_d, 0, 15)
        
        # Time emerges from dimensional change
        delta_d = self.dimension - old_d
        self.time += delta_d * PHI  # Golden ratio scaling
        
        # Phase sapping: energy flows based on dimensional gradient
        self._update_phase_dynamics(delta_d)
        
        # Check for dimensional emergence at integer boundaries
        self._check_emergence()
        
        # Record the state
        self._record_history()
    
    def _update_phase_dynamics(self, delta_d):
        """
        Phase sapping is the mechanism by which higher dimensions
        'feed' on lower ones, creating a natural hierarchy.
        """
        if delta_d > 0:
            # Dimension increasing: energy flows upward
            self._sap_phase_upward(delta_d)
        elif delta_d < 0:
            # Dimension decreasing: energy dissipates downward
            self._dissipate_phase_downward(abs(delta_d))
    
    def _sap_phase_upward(self, strength):
        """
        Higher dimensions consume phase from lower dimensions.
        This creates the arrow of time and drives emergence.
        """
        d_current = int(self.dimension)
        
        for target in range(1, min(d_current + 2, 16)):
            # Calculate how much phase this dimension needs
            capacity = self.n_ball_volume(target)
            current = abs(self.phase_density[target])
            deficit = max(0, capacity - current)
            
            if deficit > 0:
                # Sap from all lower dimensions
                for source in range(target):
                    if abs(self.phase_density[source]) > 0.01:
                        # Sapping rate depends on dimensional distance
                        rate = strength * deficit / (target - source + PHI)
                        rate = min(rate, 0.1)  # Prevent instability
                        
                        # Transfer with phase rotation
                        transfer = self.phase_density[source] * rate
                        self.phase_density[source] -= transfer
                        self.phase_density[target] += transfer * np.exp(1j * PI * target / 6)
                        
                        # Track the flow for visualization
                        self.flow_matrix[source, target] = abs(transfer)
        
        # Decay old flows
        self.flow_matrix *= 0.95
    
    def _dissipate_phase_downward(self, strength):
        """
        When dimension decreases, energy dissipates back down,
        reversing the arrow of time locally.
        """
        d_current = int(self.dimension)
        
        # Higher dimensions lose coherence
        for d in range(d_current + 1, 16):
            self.phase_density[d] *= (1 - strength * 0.1)
        
        # Energy flows back to lower dimensions
        for source in range(d_current + 1, min(16, d_current + 4)):
            if abs(self.phase_density[source]) > 0.01:
                for target in range(max(0, source - 3), source):
                    transfer = self.phase_density[source] * strength * 0.05
                    self.phase_density[source] -= transfer
                    self.phase_density[target] += transfer * np.exp(-1j * PI / 4)
                    self.flow_matrix[source, target] = abs(transfer)
        
        self.flow_matrix *= 0.9
    
    def _check_emergence(self):
        """
        A dimension 'emerges' when its phase density reaches the
        critical threshold (its volume capacity). This is analogous
        to a phase transition in physics.
        """
        d_int = int(self.dimension)
        if 0 < d_int < 16 and d_int not in self.emerged:
            current_phase = abs(self.phase_density[d_int])
            capacity = self.n_ball_volume(d_int)
            
            if current_phase >= capacity * 0.9:
                self.emerged.add(d_int)
                print(f"✨ Dimension {d_int} emerged! (phase: {current_phase:.3f}, capacity: {capacity:.3f})")
                
                # Seed the next dimension
                if d_int + 1 < 16:
                    self.phase_density[d_int + 1] += 0.01 * np.exp(1j * PI / 4)
    
    def _record_history(self):
        """Keep track of the system's evolution for analysis."""
        self.history.append({
            'dimension': self.dimension,
            'time': self.time,
            'phase': self.phase_density.copy(),
            'emerged': len(self.emerged),
            'complexity': self.complexity_measure(self.dimension)
        })
        
        # Keep history bounded
        if len(self.history) > 500:
            self.history = self.history[-400:]

class DimensionalVisualization:
    """
    Visualization that reveals the deep truths of dimensional emergence.
    Focus on clarity, pedagogy, and revealing the 'why'.
    """
    
    def __init__(self):
        self.framework = DimensionalFramework()
        self.auto_evolve = False
        
        # Precompute the geometric landscape
        self._compute_landscape()
        
    def _compute_landscape(self):
        """
        Compute the fundamental geometric measures across all dimensions.
        This reveals the critical points and transitions.
        """
        self.d_range = np.linspace(0.01, 15, 2000)
        
        # The three fundamental measures
        self.volumes = np.array([self.framework.n_ball_volume(d) for d in self.d_range])
        self.surfaces = np.array([self.framework.n_sphere_surface(d) for d in self.d_range])
        self.complexity = self.volumes * self.surfaces
        
        # Find critical points
        self.complexity_peak_idx = np.argmax(self.complexity)
        self.complexity_peak_d = self.d_range[self.complexity_peak_idx]
        self.complexity_peak_val = self.complexity[self.complexity_peak_idx]
        
        self.volume_peak_idx = np.argmax(self.volumes)
        self.volume_peak_d = self.d_range[self.volume_peak_idx]
        
        self.surface_peak_idx = np.argmax(self.surfaces)
        self.surface_peak_d = self.d_range[self.surface_peak_idx]
        
        print(f"\nCRITICAL DISCOVERIES:")
        print(f"  Complexity (V×S) peaks at d = {self.complexity_peak_d:.3f}")
        print(f"  Volume peaks at d = {self.volume_peak_d:.3f}")
        print(f"  Surface peaks at d = {self.surface_peak_d:.3f}")
        print(f"  π-boundary at d = {PI:.3f}")
        print(f"  2π-boundary at d = {2*PI:.3f}")
        print(f"\nThese are not arbitrary - they represent fundamental")
        print(f"transitions in the nature of dimensional space.\n")
    
    def create_interface(self):
        """Create the complete interactive visualization."""
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle('Dimensional Emergence: Where Mathematics Becomes Physics',
                          fontsize=14, fontweight='bold')
        
        # Create layout: 2x2 main grid with info panel below
        gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.2, 0.3],
                              hspace=0.3, wspace=0.25)
        
        # Four main visualization panels
        self.ax_landscape = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax_complexity = self.fig.add_subplot(gs[0, 1])
        self.ax_phase = self.fig.add_subplot(gs[1, 0], projection='3d')
        self.ax_quantum = self.fig.add_subplot(gs[1, 1])
        
        # Info panel spans bottom
        self.ax_info = self.fig.add_subplot(gs[2, :])
        self.ax_info.axis('off')
        
        # Configure 3D axes
        for ax3d in [self.ax_landscape, self.ax_phase]:
            ax3d.set_proj_type('ortho')
            ax3d.view_init(elev=36.87, azim=-45)  # Golden angle view
            ax3d.set_box_aspect((1, 1, 1))
        
        # THE PRIMARY CONTROL: Dimension slider
        ax_dim_slider = plt.axes([0.15, 0.02, 0.5, 0.03])
        self.dim_slider = Slider(ax_dim_slider, 'DIMENSION', 0, 12,
                                valinit=0, valstep=0.01, color='gold')
        self.dim_slider.on_changed(self._on_dimension_change)
        
        # Control buttons
        ax_auto = plt.axes([0.70, 0.02, 0.08, 0.03])
        ax_reset = plt.axes([0.80, 0.02, 0.08, 0.03])
        ax_inject = plt.axes([0.90, 0.02, 0.08, 0.03])
        
        self.btn_auto = Button(ax_auto, 'Auto')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_inject = Button(ax_inject, 'Inject')
        
        self.btn_auto.on_clicked(self._toggle_auto)
        self.btn_reset.on_clicked(self._reset)
        self.btn_inject.on_clicked(self._inject_energy)
    
    def _on_dimension_change(self, val):
        """
        When dimension changes, EVERYTHING changes.
        This is the fundamental operation of reality.
        """
        self.framework.set_dimension(val)
        self._update_all_views()
    
    def _toggle_auto(self, event):
        """Toggle automatic dimensional evolution."""
        self.auto_evolve = not self.auto_evolve
        self.btn_auto.label.set_text('Stop' if self.auto_evolve else 'Auto')
    
    def _reset(self, event):
        """Return to the void."""
        self.framework = DimensionalFramework()
        self.dim_slider.set_val(0)
        self._update_all_views()
    
    def _inject_energy(self, event):
        """Inject phase energy at current dimension."""
        d = int(self.framework.dimension)
        if d < 16:
            self.framework.phase_density[d] += 0.3 * np.exp(1j * np.random.random() * 2 * PI)
            print(f"💉 Energy injected at dimension {d}")
            self._update_all_views()
    
    def _update_all_views(self):
        """Update all visualization panels."""
        self._draw_dimensional_landscape()
        self._draw_complexity_truth()
        self._draw_phase_dynamics()
        self._draw_quantum_structure()
        self._draw_info_panel()
        
        self.fig.canvas.draw_idle()
    
    def _draw_dimensional_landscape(self):
        """
        The main 3D view showing how geometry changes with dimension.
        This reveals the living, breathing nature of dimensional space.
        """
        ax = self.ax_landscape
        ax.clear()
        ax.set_title('The Living Geometry of Dimensions', fontsize=11)
        
        # Sample dimensions for visualization
        d_sample = self.d_range[::40]
        
        # Create lemniscate-inspired surface showing V and S relationship
        for i, d in enumerate(d_sample):
            if d > 0.1:
                v = self.framework.n_ball_volume(d)
                s = self.framework.n_sphere_surface(d)
                c = v * s  # Complexity
                
                # Parametric curve in 3D
                t = np.linspace(0, 2*PI, 30)
                x = d * np.ones_like(t)
                y = v * np.cos(t)
                z = v * np.sin(t)
                
                # Color by complexity - THIS IS KEY
                color_val = c / self.complexity_peak_val
                color = cm.plasma(np.clip(color_val, 0, 1))
                
                # Draw the curve
                ax.plot(x, y, z, color=color, alpha=0.6, linewidth=2)
        
        # Mark current position
        if self.framework.dimension > 0:
            v_current = self.framework.n_ball_volume(self.framework.dimension)
            phase = self.framework.time % (2*PI)
            
            ax.scatter([self.framework.dimension],
                      [v_current * np.cos(phase)],
                      [v_current * np.sin(phase)],
                      c='gold', s=200, marker='*',
                      edgecolors='black', linewidth=2)
        
        # Mark emerged dimensions with golden rings
        for d in self.framework.emerged:
            if 0 < d < 12:
                v = self.framework.n_ball_volume(d)
                theta = np.linspace(0, 2*PI, 50)
                ax.plot(d * np.ones_like(theta),
                       v * np.cos(theta) * 0.5,
                       v * np.sin(theta) * 0.5,
                       'gold', linewidth=3, alpha=0.8)
        
        # Mark critical dimensions
        for d_crit, label, color in [(PI, 'π', 'red'), 
                                      (2*PI, '2π', 'orange'),
                                      (self.complexity_peak_d, 'V×S peak', 'green')]:
            if d_crit < 12:
                v_crit = self.framework.n_ball_volume(d_crit)
                ax.plot([d_crit, d_crit], [0, 0], [-v_crit, v_crit],
                       color=color, linestyle='--', alpha=0.5)
                ax.text(d_crit, 0, v_crit, f' {label}', fontsize=8)
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Geometric X')
        ax.set_zlabel('Geometric Y')
        ax.set_xlim([0, 12])
    
    def _draw_complexity_truth(self):
        """
        The KEY panel showing WHERE and WHY complexity peaks.
        This is the deepest truth of dimensional emergence.
        """
        ax = self.ax_complexity
        ax.clear()
        ax.set_title('The Truth: V×S Complexity Peak', fontsize=11, fontweight='bold')
        
        # Plot the three measures
        ax.plot(self.d_range, self.volumes, 'b-', alpha=0.3, label='Volume')
        ax.plot(self.d_range, self.surfaces, 'g-', alpha=0.3, label='Surface')
        ax.plot(self.d_range, self.complexity, 'r-', linewidth=2.5, label='V×S (Complexity)')
        
        # THE PEAK - this is everything
        ax.scatter([self.complexity_peak_d], [self.complexity_peak_val],
                  c='gold', s=200, marker='*', zorder=5,
                  edgecolors='black', linewidth=2)
        
        ax.annotate(f'Peak at d={self.complexity_peak_d:.2f}\nMaximum complexity!',
                   xy=(self.complexity_peak_d, self.complexity_peak_val),
                   xytext=(self.complexity_peak_d + 1, self.complexity_peak_val),
                   fontsize=9, ha='left',
                   arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        
        # Current dimension
        ax.axvline(x=self.framework.dimension, color='red', linewidth=2, alpha=0.7)
        current_c = self.framework.complexity_measure(self.framework.dimension)
        ax.scatter([self.framework.dimension], [current_c],
                  c='red', s=100, zorder=4)
        
        # Critical boundaries
        ax.axvline(x=PI, color='red', linestyle='--', alpha=0.3)
        ax.axvline(x=2*PI, color='orange', linestyle='--', alpha=0.3)
        ax.text(PI, ax.get_ylim()[1]*0.9, 'π', ha='center', fontsize=8)
        ax.text(2*PI, ax.get_ylim()[1]*0.9, '2π', ha='center', fontsize=8)
        
        # Shade stability regions
        ax.axvspan(0, PI, alpha=0.1, color='green', label='Stable')
        ax.axvspan(PI, 2*PI, alpha=0.1, color='yellow', label='Transition')
        ax.axvspan(2*PI, 12, alpha=0.1, color='red', label='Compression')
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Measure')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 12])
        ax.set_ylim([0, max(self.complexity_peak_val * 1.2, 10)])
    
    def _draw_phase_dynamics(self):
        """
        Show how phase energy flows between dimensions.
        This is the mechanism of dimensional emergence.
        """
        ax = self.ax_phase
        ax.clear()
        ax.set_title('Phase Sapping Dynamics', fontsize=11)
        
        # Position dimensions in a spiral
        for d in range(min(10, 16)):
            theta = 2 * PI * d / 10
            r = 1 + d * 0.15
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = abs(self.framework.phase_density[d]) * 3
            
            if z > 0.01:
                # Draw vertical bar showing phase magnitude
                ax.plot([x, x], [y, y], [0, z], 'b-', linewidth=4, alpha=0.7)
                
                # Color based on emergence
                color = 'gold' if d in self.framework.emerged else 'cyan'
                ax.scatter([x], [y], [z], s=150, c=color,
                          edgecolors='black', linewidth=1)
                ax.text(x, y, z, f' {d}', fontsize=8)
        
        # Draw energy flows
        for i in range(10):
            for j in range(i+1, min(i+3, 10)):
                flow = self.framework.flow_matrix[i, j]
                if flow > 0.001:
                    theta_i = 2 * PI * i / 10
                    theta_j = 2 * PI * j / 10
                    r_i = 1 + i * 0.15
                    r_j = 1 + j * 0.15
                    
                    x_i, y_i = r_i * np.cos(theta_i), r_i * np.sin(theta_i)
                    x_j, y_j = r_j * np.cos(theta_j), r_j * np.sin(theta_j)
                    z_i = abs(self.framework.phase_density[i]) * 3
                    z_j = abs(self.framework.phase_density[j]) * 3
                    
                    # Draw flow arrow
                    midx, midy, midz = (x_i+x_j)/2, (y_i+y_j)/2, (z_i+z_j)/2
                    ax.plot([x_i, midx, x_j], [y_i, midy, y_j], [z_i, midz, z_j],
                           'r-', alpha=min(1, flow*20), linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Phase Magnitude')
    
    def _draw_quantum_structure(self):
        """
        Show how angular quantization emerges from kissing numbers.
        This reveals why quantum mechanics is inevitable.
        """
        ax = self.ax_quantum
        ax.clear()
        ax.set_title('Angular Quantization (Kissing Numbers)', fontsize=11)
        
        # The kissing numbers - fundamental geometric constraints
        kissing = {1: 2, 2: 6, 3: 12, 4: 24, 5: 40, 6: 72,
                  7: 126, 8: 240, 9: 272, 10: 336, 11: 438, 12: 756}
        
        dims = list(kissing.keys())
        values = list(kissing.values())
        
        # Color based on emergence
        colors = ['gold' if d in self.framework.emerged else 'lightgray'
                  for d in dims]
        
        bars = ax.bar(dims, values, color=colors, edgecolor='black', alpha=0.7)
        
        # Add values on emerged dimensions
        for bar, d, val in zip(bars, dims, values):
            if d in self.framework.emerged:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{val}', ha='center', va='bottom', fontsize=8)
        
        # Show angular resolution
        ax2 = ax.twinx()
        angular_res = [2*PI/k for k in values]
        ax2.plot(dims, angular_res, 'r.-', alpha=0.5, label='Angular Resolution')
        ax2.set_ylabel('Min Angular Resolution (radians)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Kissing Number')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 13])
        
        # Mark current dimension
        current_d = int(self.framework.dimension)
        if 0 < current_d <= 12:
            ax.axvline(x=current_d, color='red', linewidth=2, alpha=0.5)
    
    def _draw_info_panel(self):
        """
        The explanatory panel that reveals the deep truths.
        This is where understanding crystallizes.
        """
        ax = self.ax_info
        ax.clear()
        ax.axis('off')
        
        # Determine stability region
        if self.framework.dimension < PI:
            region = "STABLE (d < π)"
            bgcolor = '#e8ffe8'
        elif self.framework.dimension < 2*PI:
            region = "TRANSITION (π < d < 2π)"
            bgcolor = '#ffffcc'
        else:
            region = "COMPRESSION (d > 2π)"
            bgcolor = '#ffe8e8'
        
        info_text = (
            f"FUNDAMENTAL STATE:  Dimension = {self.framework.dimension:.3f} ({region})  |  "
            f"Emergent Time = {self.framework.time:.3f}  |  "
            f"Emerged Dimensions: {sorted(self.framework.emerged)}\n\n"
            
            f"THE DEEP TRUTH: Dimension is not a container for physics - dimension IS the fundamental parameter.\n"
            f"Time emerges from dimensional change: increasing dimension → forward time, decreasing → time reversal.\n\n"
            
            f"KEY INSIGHT: V×S peaks at d={self.complexity_peak_d:.2f} - this is WHERE the universe has maximum\n"
            f"capacity for complexity. Not coincidentally, our 3+1 dimensional spacetime sits just below this peak\n"
            f"in the stable region (d < π), allowing rich physics while maintaining coherence.\n\n"
            
            f"PHASE SAPPING: Higher dimensions 'feed' on lower ones, creating a natural hierarchy and the arrow of time.\n"
            f"This is WHY we experience time as flowing forward - it's the direction of dimensional emergence."
        )
        
        props = dict(boxstyle='round,pad=0.5', facecolor=bgcolor, alpha=0.9)
        ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
               fontsize=9, ha='center', va='center',
               bbox=props, family='monospace')
    
    def animate(self, frame):
        """Animation loop for automatic evolution."""
        if self.auto_evolve:
            # Slowly increase dimension
            new_d = self.framework.dimension + 0.015
            if new_d > 12:
                new_d = 0  # Loop back to void
            self.dim_slider.set_val(new_d)
        return []
    
    def run(self):
        """Launch the complete visualization."""
        self.create_interface()
        self._update_all_views()
        
        # Create animation
        anim = animation.FuncAnimation(self.fig, self.animate,
                                      interval=50, blit=False)
        
        plt.show()
        return anim

def main():
    """
    Launch the complete framework revealing how dimension creates reality.
    """
    print("="*80)
    print("DIMENSIONAL EMERGENCE: The Mathematical Truth of Reality")
    print("="*80)
    print("\nFundamental Constants:")
    print(f"  ϖ (varpi) = {VARPI:.10f}  [Dimensional scaling factor]")
    print(f"  φ (phi)   = {PHI:.10f}  [Golden ratio - recursive proportion]")
    print(f"  π (pi)    = {PI:.10f}  [First stability boundary]")
    
    print("\nCore Insights:")
    print("  • Dimension is THE fundamental parameter (not space or time)")
    print("  • Time emerges from dimensional change")
    print("  • V×S product reveals WHERE complexity maximizes")
    print("  • Phase sapping creates the arrow of time")
    print("  • Angular quantization makes quantum mechanics inevitable")
    
    print("\nControls:")
    print("  • DIMENSION slider - the master control")
    print("  • Auto - watch dimensional evolution")
    print("  • Inject - add phase energy")
    print("  • Reset - return to void")
    
    print("\nWatch how increasing dimension creates time,")
    print("how phase flows between dimensions,")
    print("and how complexity emerges at the V×S peak.")
    print("="*80)
    
    viz = DimensionalVisualization()
    viz.run()
    
    return viz

if __name__ == "__main__":
    viz = main()