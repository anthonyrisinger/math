#!/usr/bin/env python3
"""
DIMENSIONAL EMERGENCE: The Complete Framework
Where dimension itself is the fundamental parameter from which time emerges
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import gamma, gammaln
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# The fundamental constants that govern reality
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio - nature's proportion
PSI = 1 / PHI  # Golden conjugate
VARPI = gamma(0.25)**2 / (2 * np.sqrt(2 * np.pi))  # 1.311028777...
PI = np.pi
E = np.e

class DimensionalReality:
    """
    The true framework where dimension is primary and time emerges from it
    Key insight: Increasing dimension creates time; decreasing dimension dissipates it
    """
    
    def __init__(self):
        # Dimension is THE fundamental parameter
        self.dimension = 0.0  # Start at the void
        self.max_dim = 15.0
        
        # Time emerges from dimensional change
        self.time = 0.0
        self.time_rate = 0.0  # d(time)/d(dimension)
        
        # Phase state for each integer dimension
        self.phase_density = np.zeros(16, dtype=complex)
        self.phase_density[0] = 1.0  # Initial unity
        
        # Energy flows between dimensions
        self.flow_matrix = np.zeros((16, 16))
        
        # Emergence tracking
        self.emerged = {0}  # Void always exists
        self.emerging = None
        
        # History for understanding
        self.history = []
        
    def n_ball_volume(self, d):
        """Volume of unit d-ball - works for ANY real d"""
        if abs(d) < 1e-10:
            return 1.0
        if d > 170:  # Avoid overflow
            log_vol = (d/2) * np.log(PI) - gammaln(d/2 + 1)
            return np.exp(log_vol)
        return PI**(d/2) / gamma(d/2 + 1)
    
    def n_sphere_surface(self, d):
        """Surface area of unit (d-1)-sphere in R^d"""
        if abs(d) < 1e-10:
            return 2.0  # Two points
        if d > 170:
            log_surf = np.log(2) + (d/2) * np.log(PI) - gammaln(d/2)
            return np.exp(log_surf)
        return 2 * PI**(d/2) / gamma(d/2)
    
    def energy_product(self, d):
        """V * S product - the key to understanding WHY dimensions emerge"""
        return self.n_ball_volume(d) * self.n_sphere_surface(d)
    
    def phase_capacity(self, d):
        """Maximum phase before emergence - the threshold"""
        return self.n_ball_volume(d)
    
    def set_dimension(self, new_d):
        """
        Set dimension directly - this is THE fundamental operation
        Time emerges from this change
        """
        old_d = self.dimension
        self.dimension = np.clip(new_d, 0, self.max_dim)
        
        # Time flows forward when dimension increases, backward when it decreases
        delta_d = self.dimension - old_d
        self.time_rate = delta_d * PHI  # Golden ratio scaling
        self.time += self.time_rate
        
        # Phase sapping based on dimensional change
        if delta_d > 0:
            # Increasing dimension - energy flows upward
            self.sap_phase_upward(delta_d)
        elif delta_d < 0:
            # Decreasing dimension - energy dissipates
            self.dissipate_phase(abs(delta_d))
        
        # Check emergence at integer crossings
        d_int = int(self.dimension)
        if d_int not in self.emerged and d_int > 0:
            current_phase = abs(self.phase_density[d_int])
            capacity = self.phase_capacity(d_int)
            if current_phase >= capacity * 0.9:
                self.emerged.add(d_int)
                print(f"✨ Dimension {d_int} emerged at d={self.dimension:.3f}")
        
        # Record state
        self.record_history()
    
    def sap_phase_upward(self, delta):
        """Energy flows to higher dimensions as dimension increases"""
        d_int = int(self.dimension)
        
        for target in range(1, min(d_int + 2, 16)):
            deficit = self.phase_capacity(target) - abs(self.phase_density[target])
            
            if deficit > 0:
                for source in range(target):
                    if abs(self.phase_density[source]) > 0.01:
                        # Sapping rate based on dimensional distance and delta
                        rate = delta * deficit / (target - source + PHI)
                        rate = min(rate, 0.1)
                        
                        transfer = self.phase_density[source] * rate
                        self.phase_density[source] -= transfer
                        self.phase_density[target] += transfer * np.exp(1j * PI * target / 6)
                        
                        self.flow_matrix[source, target] = abs(transfer)
        
        # Decay old flows
        self.flow_matrix *= 0.95
    
    def dissipate_phase(self, delta):
        """Energy dissipates as dimension decreases"""
        d_int = int(self.dimension)
        
        # Higher dimensions lose energy
        for d in range(d_int + 1, 16):
            self.phase_density[d] *= (1 - delta * 0.1)
        
        # Energy flows back down
        for source in range(d_int + 1, 16):
            if abs(self.phase_density[source]) > 0.01:
                for target in range(max(0, source - 3), source):
                    transfer = self.phase_density[source] * delta * 0.05
                    self.phase_density[source] -= transfer
                    self.phase_density[target] += transfer * np.exp(-1j * PI / 4)
                    
                    self.flow_matrix[source, target] = abs(transfer)
        
        self.flow_matrix *= 0.9
    
    def record_history(self):
        """Record current state"""
        self.history.append({
            'dimension': self.dimension,
            'time': self.time,
            'phase': self.phase_density.copy(),
            'emerged': len(self.emerged),
            'total_phase': np.sum(np.abs(self.phase_density)),
            'energy_product': self.energy_product(self.dimension)
        })
        
        # Keep history manageable
        if len(self.history) > 500:
            self.history = self.history[-400:]

class TruthfulVisualization:
    """
    The complete visualization showing WHY dimensional emergence matters
    """
    
    def __init__(self):
        self.reality = DimensionalReality()
        self.current_d = 0.0
        self.auto_evolve = False
        
        # Precompute critical information
        self.compute_critical_geometry()
        
    def compute_critical_geometry(self):
        """Find all critical dimensions and transitions"""
        d_range = np.linspace(0.01, 15, 3000)
        
        # Compute all measures
        self.d_range = d_range
        self.volumes = np.array([self.reality.n_ball_volume(d) for d in d_range])
        self.surfaces = np.array([self.reality.n_sphere_surface(d) for d in d_range])
        self.products = self.volumes * self.surfaces
        
        # Find critical points
        vol_peaks, _ = find_peaks(self.volumes)
        surf_peaks, _ = find_peaks(self.surfaces)
        prod_peaks, _ = find_peaks(self.products)
        
        self.critical = {
            'volume_peak': d_range[vol_peaks[0]] if len(vol_peaks) > 0 else 5.256,
            'surface_peak': d_range[surf_peaks[0]] if len(surf_peaks) > 0 else 7.256,
            'product_peak': d_range[prod_peaks[0]] if len(prod_peaks) > 0 else 6.0,
            'pi': PI,
            '2pi': 2 * PI,
            '4pi': 4 * PI,
            'phi': PHI,
            'e': E,
            'varpi': VARPI
        }
        
        # Find where V*S is maximum - this is KEY
        self.product_peak_idx = np.argmax(self.products)
        self.product_peak_d = d_range[self.product_peak_idx]
        self.product_peak_val = self.products[self.product_peak_idx]
        
        print(f"KEY INSIGHT: V*S peaks at d={self.product_peak_d:.3f} with value {self.product_peak_val:.3f}")
        print(f"This is WHERE and WHY complexity maximizes!")
    
    def create_interface(self):
        """Create the complete interface"""
        self.fig = plt.figure(figsize=(20, 12), facecolor='#f0f0f0')
        self.fig.suptitle('DIMENSIONAL EMERGENCE: The Truth Revealed', 
                          fontsize=16, fontweight='bold')
        
        # Main grid
        gs = gridspec.GridSpec(3, 4, height_ratios=[1.2, 1, 0.8], 
                              hspace=0.3, wspace=0.3)
        
        # Main 3D view - larger, spans 2x2
        self.ax_main = self.fig.add_subplot(gs[:2, :2], projection='3d')
        self.ax_main.set_proj_type('ortho')
        self.ax_main.view_init(elev=36.87, azim=-45)
        self.ax_main.set_box_aspect((1, 1, 1))
        
        # Critical views
        self.ax_product = self.fig.add_subplot(gs[0, 2])  # V*S product
        self.ax_phase = self.fig.add_subplot(gs[0, 3], projection='3d')  # Phase space
        self.ax_flow = self.fig.add_subplot(gs[1, 2], projection='3d')  # Energy flow
        self.ax_quantum = self.fig.add_subplot(gs[1, 3])  # Quantum structure
        
        # Set 3D projections
        for ax in [self.ax_phase, self.ax_flow]:
            ax.set_proj_type('ortho')
            ax.view_init(elev=36.87, azim=-45)
            ax.set_box_aspect((1, 1, 1))
        
        # Info panel
        self.ax_info = self.fig.add_subplot(gs[2, :])
        self.ax_info.axis('off')
        
        # THE PRIMARY CONTROL - Dimension slider
        ax_dim = plt.axes([0.15, 0.02, 0.5, 0.03], facecolor='lightgray')
        self.slider_dim = Slider(ax_dim, 'DIMENSION', 0, 15, valinit=0, 
                                 valstep=0.01, color='gold')
        self.slider_dim.on_changed(self.update_dimension)
        
        # Secondary controls
        ax_auto = plt.axes([0.70, 0.02, 0.08, 0.03])
        ax_reset = plt.axes([0.79, 0.02, 0.08, 0.03])
        ax_inject = plt.axes([0.88, 0.02, 0.08, 0.03])
        
        self.btn_auto = Button(ax_auto, 'Auto')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_inject = Button(ax_inject, 'Inject')
        
        self.btn_auto.on_clicked(self.toggle_auto)
        self.btn_reset.on_clicked(self.reset)
        self.btn_inject.on_clicked(self.inject_energy)
        
    def update_dimension(self, val):
        """THE fundamental update - everything flows from dimensional change"""
        self.reality.set_dimension(val)
        self.update_all()
    
    def toggle_auto(self, event):
        """Toggle automatic evolution"""
        self.auto_evolve = not self.auto_evolve
        self.btn_auto.label.set_text('Stop' if self.auto_evolve else 'Auto')
    
    def reset(self, event):
        """Reset to beginning"""
        self.reality = DimensionalReality()
        self.slider_dim.set_val(0)
        self.update_all()
    
    def inject_energy(self, event):
        """Inject energy at current dimension"""
        d_int = int(self.reality.dimension)
        if d_int < 16:
            self.reality.phase_density[d_int] += 0.3 * np.exp(1j * np.random.random() * 2 * PI)
            print(f"💉 Energy injected at dimension {d_int}")
            self.update_all()
    
    def update_all(self):
        """Update all visualizations"""
        # Clear all axes
        for ax in [self.ax_main, self.ax_product, self.ax_phase, 
                   self.ax_flow, self.ax_quantum]:
            ax.clear()
        
        # Update each view
        self.draw_main_reality()
        self.draw_product_truth()
        self.draw_phase_space()
        self.draw_energy_flow()
        self.draw_quantum_structure()
        self.draw_info_panel()
        
        self.fig.canvas.draw_idle()
    
    def draw_main_reality(self):
        """The main 3D view showing the complete picture"""
        ax = self.ax_main
        ax.set_title('The Living Geometry of Dimensional Emergence', fontsize=12, fontweight='bold')
        
        # Draw the fundamental curves
        d_sample = self.d_range[::30]
        
        # Create the lemniscate-inspired surface
        t = np.linspace(0, 2*PI, 40)
        
        for i, d in enumerate(d_sample):
            if d > 0.1:
                v = self.reality.n_ball_volume(d)
                s = self.reality.n_sphere_surface(d)
                p = v * s  # The product
                
                # Lemniscate parametrization
                x = d * np.ones_like(t)
                y = v * np.cos(t) / (1 + np.sin(t)**2)
                z = v * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
                
                # Color by product (the KEY measure)
                color_val = p / self.product_peak_val
                color = cm.plasma(np.clip(color_val, 0, 1))
                
                # Draw as parametric curve
                for j in range(len(t)-1):
                    ax.plot(x[j:j+2], y[j:j+2], z[j:j+2], 
                           color=color, alpha=0.6, linewidth=2)
        
        # Mark current dimension
        if self.reality.dimension > 0:
            v_curr = self.reality.n_ball_volume(self.reality.dimension)
            s_curr = self.reality.n_sphere_surface(self.reality.dimension)
            
            # Current position on lemniscate
            phase = self.reality.time % (2*PI)
            x_curr = self.reality.dimension
            y_curr = v_curr * np.cos(phase) / (1 + np.sin(phase)**2)
            z_curr = v_curr * np.sin(phase) * np.cos(phase) / (1 + np.sin(phase)**2)
            
            ax.scatter([x_curr], [y_curr], [z_curr], 
                      c='gold', s=200, marker='*', 
                      edgecolors='black', linewidth=2, zorder=5)
        
        # Mark critical dimensions
        for name, d_crit in self.critical.items():
            if 0 < d_crit < 15:
                v_crit = self.reality.n_ball_volume(d_crit)
                ax.plot([d_crit, d_crit], [0, 0], [-v_crit/2, v_crit/2],
                       'r--', alpha=0.3, linewidth=1)
                
                # Label key ones
                if name in ['product_peak', 'pi', '2pi']:
                    ax.text(d_crit, 0, v_crit/2, f' {name}', fontsize=8)
        
        # Emerged dimensions as golden rings
        for d in self.reality.emerged:
            if d > 0:
                v_em = self.reality.n_ball_volume(d)
                theta_ring = np.linspace(0, 2*PI, 50)
                x_ring = d * np.ones_like(theta_ring)
                y_ring = v_em * np.cos(theta_ring) / 2
                z_ring = v_em * np.sin(theta_ring) / 2
                ax.plot(x_ring, y_ring, z_ring, 'gold', 
                       linewidth=3, alpha=0.8)
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Geometric Flow')
        ax.set_zlabel('Phase Structure')
        ax.set_xlim([0, 15])
    
    def draw_product_truth(self):
        """The KEY insight - where V*S peaks and WHY it matters"""
        ax = self.ax_product
        ax.set_title('THE KEY: V×S Product (Complexity Measure)', fontsize=10, fontweight='bold')
        
        # Plot all three measures
        ax.plot(self.d_range, self.volumes, 'b-', alpha=0.5, linewidth=1, label='Volume')
        ax.plot(self.d_range, self.surfaces, 'g-', alpha=0.5, linewidth=1, label='Surface')
        ax.plot(self.d_range, self.products, 'r-', linewidth=2, label='V×S Product')
        
        # Mark the peak
        ax.scatter([self.product_peak_d], [self.product_peak_val], 
                  c='gold', s=150, marker='*', zorder=5,
                  edgecolors='black', linewidth=2)
        ax.axvline(x=self.product_peak_d, color='gold', linestyle='--', alpha=0.5)
        ax.text(self.product_peak_d, self.product_peak_val, 
               f'\n Peak at d={self.product_peak_d:.2f}\n Max complexity!', 
               fontsize=9, ha='center')
        
        # Current dimension
        ax.axvline(x=self.reality.dimension, color='red', linewidth=2, alpha=0.7)
        
        # Critical boundaries
        for name, d in [('π', PI), ('2π', 2*PI)]:
            ax.axvline(x=d, color='gray', linestyle='--', alpha=0.3)
            ax.text(d, ax.get_ylim()[1]*0.9, name, ha='center', fontsize=8)
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Measure')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 12])
        ax.set_ylim([0, max(10, self.product_peak_val * 1.2)])
    
    def draw_phase_space(self):
        """Complex phase evolution"""
        ax = self.ax_phase
        ax.set_title('Phase Space', fontsize=10)
        
        # Draw each dimension's phase
        for d in range(min(12, 16)):
            z = self.reality.phase_density[d]
            mag = abs(z)
            
            if mag > 0.001:
                # 3D embedding
                x = z.real
                y = z.imag
                z_coord = d * 0.3
                
                # Size and color
                size = 200 * mag
                color = 'gold' if d in self.reality.emerged else 'cyan'
                
                ax.scatter([x], [y], [z_coord], s=size, c=color,
                          alpha=0.8, edgecolors='black', linewidth=1)
                
                if d in self.reality.emerged:
                    ax.text(x, y, z_coord, f' {d}', fontsize=7)
        
        # Unit circles at different heights
        theta = np.linspace(0, 2*PI, 50)
        for level in [0, 1, 2]:
            ax.plot(np.cos(theta), np.sin(theta), 
                   [level]*len(theta), 'k--', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('Re')
        ax.set_ylabel('Im')
        ax.set_zlabel('Dim')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([0, 4])
    
    def draw_energy_flow(self):
        """Energy flow visualization"""
        ax = self.ax_flow
        ax.set_title('Phase Sapping', fontsize=10)
        
        # Position dimensions in spiral
        for d in range(min(8, 16)):
            theta = 2 * PI * d / 8
            r = 1 + d * 0.2
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = abs(self.reality.phase_density[d])
            
            # Vertical bars
            if z > 0.01:
                ax.plot([x, x], [y, y], [0, z], 'b-', linewidth=3, alpha=0.7)
                
                color = 'gold' if d in self.reality.emerged else 'cyan'
                ax.scatter([x], [y], [z], s=150, c=color,
                          edgecolors='black', linewidth=1)
                ax.text(x, y, z, f'{d}', fontsize=7, ha='center')
        
        # Draw flows
        for i in range(8):
            for j in range(i+1, min(i+3, 8)):
                flow = self.reality.flow_matrix[i, j]
                if flow > 0.001:
                    theta_i = 2 * PI * i / 8
                    theta_j = 2 * PI * j / 8
                    r_i = 1 + i * 0.2
                    r_j = 1 + j * 0.2
                    
                    x_i = r_i * np.cos(theta_i)
                    y_i = r_i * np.sin(theta_i)
                    x_j = r_j * np.cos(theta_j)
                    y_j = r_j * np.sin(theta_j)
                    
                    z_i = abs(self.reality.phase_density[i])
                    z_j = abs(self.reality.phase_density[j])
                    
                    # Flow arrow
                    ax.plot([x_i, x_j], [y_i, y_j], [z_i, z_j],
                           'r-', alpha=min(1, flow*10), linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Phase')
    
    def draw_quantum_structure(self):
        """Quantum numbers and angular quantization"""
        ax = self.ax_quantum
        ax.set_title('Angular Quantization', fontsize=10)
        
        # Kissing numbers
        kissing = [2, 6, 12, 24, 40, 72, 126, 240, 272, 336, 438, 756]
        dims = range(1, min(len(kissing)+1, 13))
        
        # Bar chart of kissing numbers for emerged dimensions
        emerged_dims = [d for d in dims if d in self.reality.emerged]
        emerged_kissing = [kissing[d-1] for d in emerged_dims]
        
        if emerged_dims:
            bars = ax.bar(emerged_dims, emerged_kissing, 
                         color='gold', alpha=0.7, edgecolor='black')
            
            # Add values on bars
            for bar, val in zip(bars, emerged_kissing):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{val}', ha='center', va='bottom', fontsize=8)
        
        # Potential dimensions
        potential_dims = [d for d in dims if d not in self.reality.emerged]
        if potential_dims:
            potential_kissing = [kissing[d-1] for d in potential_dims]
            ax.bar(potential_dims, potential_kissing,
                  color='lightgray', alpha=0.3, edgecolor='gray')
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Kissing Number')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 13])
    
    def draw_info_panel(self):
        """Information panel showing the WHY"""
        ax = self.ax_info
        ax.clear()
        ax.axis('off')
        
        # Create info text
        info_text = (
            f"CURRENT STATE:  Dimension = {self.reality.dimension:.3f}  |  "
            f"Time = {self.reality.time:.3f}  |  "
            f"Emerged: {sorted(self.reality.emerged)}  |  "
            f"Total Phase: {np.sum(np.abs(self.reality.phase_density)):.3f}\n\n"
            
            f"KEY INSIGHT: V×S peaks at d={self.product_peak_d:.3f} → "
            f"Maximum complexity emerges here!\n"
            f"This is WHERE the universe has the most 'room' for both extent (V) and boundary (S).\n\n"
            
            f"CRITICAL DIMENSIONS:  "
            f"π={PI:.3f} (stability boundary)  |  "
            f"2π={2*PI:.3f} (compression limit)  |  "
            f"Volume peak={self.critical['volume_peak']:.3f}  |  "
            f"Surface peak={self.critical['surface_peak']:.3f}\n\n"
            
            f"THE TRUTH: Dimension is primary. Time emerges from dimensional change. "
            f"Increasing dimension → time flows forward. "
            f"Decreasing dimension → time dissipates. "
            f"Reality is the continuous emergence of dimensions through phase coherence."
        )
        
        # Color-coded based on current dimension
        if self.reality.dimension < PI:
            bgcolor = '#e8f5e9'  # Green - stable region
        elif self.reality.dimension < 2*PI:
            bgcolor = '#fff3e0'  # Orange - transition region
        else:
            bgcolor = '#ffebee'  # Red - compression region
        
        # Draw text box
        props = dict(boxstyle='round,pad=0.5', facecolor=bgcolor, alpha=0.8)
        ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
               fontsize=10, ha='center', va='center',
               bbox=props, family='monospace')
    
    def animate(self, frame):
        """Animation loop"""
        if self.auto_evolve:
            # Slowly increase dimension
            new_d = self.reality.dimension + 0.02
            if new_d > 12:
                new_d = 0
            self.slider_dim.set_val(new_d)
        
        return []
    
    def run(self):
        """Launch the complete visualization"""
        self.create_interface()
        self.update_all()
        
        # Create animation
        anim = animation.FuncAnimation(self.fig, self.animate,
                                      interval=50, blit=False)
        
        plt.show()
        return anim

def main():
    """The complete truth of dimensional emergence"""
    
    print("="*80)
    print("DIMENSIONAL EMERGENCE: The Complete Framework")
    print("="*80)
    print(f"Fundamental Constants:")
    print(f"  ϖ (varpi) = {VARPI:.10f}")
    print(f"  φ (phi)   = {PHI:.10f}")
    print(f"  π (pi)    = {PI:.10f}")
    print()
    print("THE KEY INSIGHT:")
    print("  Dimension is THE fundamental parameter")
    print("  Time emerges from dimensional change")
    print("  V×S product shows WHERE complexity maximizes")
    print()
    print("CONTROLS:")
    print("  DIMENSION SLIDER - The primary control")
    print("  Auto - Automatic evolution")
    print("  Inject - Add energy at current dimension")
    print("  Reset - Return to void")
    print("="*80)
    print()
    
    viz = TruthfulVisualization()
    viz.run()
    
    return viz

if __name__ == "__main__":
    viz = main()