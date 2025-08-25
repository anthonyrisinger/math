#!/usr/bin/env python3
"""
Complexity Peak Explorer
========================

Interactive 3D exploration of the V×S complexity peak at d≈6.
This is THE fundamental insight: where the universe has maximum
capacity for both interior (volume) and boundary (surface).

Run: python complexity_peak.py
Controls: Focus dimension, complexity analysis, stability regions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib import cm
from core_measures import DimensionalMeasures, setup_3d_axis, PHI, PI, E

class ComplexityPeakExplorer:
    """Explore the fundamental complexity peak in dimensional space."""
    
    def __init__(self, focus_range=(3, 9), resolution=2000):
        self.focus_range = focus_range
        self.resolution = resolution
        self.measures = DimensionalMeasures()
        
        # Compute high-resolution measures around the peak
        self.d_range = np.linspace(focus_range[0], focus_range[1], resolution)
        self.volumes = np.array([self.measures.ball_volume(d) for d in self.d_range])
        self.surfaces = np.array([self.measures.sphere_surface(d) for d in self.d_range])
        self.complexity = self.volumes * self.surfaces
        self.ratios = self.surfaces / np.maximum(self.volumes, 1e-10)
        
        # Find the exact peak
        self.peak_idx = np.argmax(self.complexity)
        self.peak_dimension = self.d_range[self.peak_idx]
        self.peak_complexity = self.complexity[self.peak_idx]
        
        # Critical analysis
        self.critical_dims = self._analyze_critical_dimensions()
        
        # Visualization state
        self.focus_dim = self.peak_dimension
        self.show_volume = True
        self.show_surface = True
        self.show_complexity = True
        self.show_derivatives = False
        self.show_stability = True
        
        print(f"COMPLEXITY PEAK DISCOVERED:")
        print(f"  Peak at d = {self.peak_dimension:.6f}")
        print(f"  Peak value = {self.peak_complexity:.6f}")
        print(f"  This is WHERE the universe maximizes information capacity!")
    
    def _analyze_critical_dimensions(self):
        """Analyze critical dimensional points."""
        # Find derivatives for stability analysis
        d_vol = np.gradient(self.volumes, self.d_range[1] - self.d_range[0])
        d_surf = np.gradient(self.surfaces, self.d_range[1] - self.d_range[0])
        d_complexity = np.gradient(self.complexity, self.d_range[1] - self.d_range[0])
        
        # Find local extrema
        vol_extrema = self._find_extrema(d_vol)
        surf_extrema = self._find_extrema(d_surf)
        comp_extrema = self._find_extrema(d_complexity)
        
        return {
            'volume_extrema': vol_extrema,
            'surface_extrema': surf_extrema,
            'complexity_extrema': comp_extrema,
            'derivatives': {
                'volume': d_vol,
                'surface': d_surf,
                'complexity': d_complexity
            }
        }
    
    def _find_extrema(self, derivative):
        """Find extrema from derivative zero-crossings."""
        extrema = []
        for i in range(1, len(derivative) - 1):
            if derivative[i-1] * derivative[i+1] < 0:  # Sign change
                extrema.append(i)
        return extrema
    
    def get_stability_regions(self):
        """Classify dimensional regions by stability."""
        regions = []
        
        for i, d in enumerate(self.d_range):
            if d < PI:
                region = "Stable"
                color = 'green'
            elif d < 2 * PI:
                region = "Transition"
                color = 'yellow'
            elif d < self.peak_dimension:
                region = "Pre-Peak"
                color = 'orange'
            elif d > self.peak_dimension * 1.2:
                region = "Compression"
                color = 'red'
            else:
                region = "Peak Zone"
                color = 'purple'
            
            regions.append((region, color))
        
        return regions
    
    def get_complexity_surface(self, theta_resolution=100):
        """Generate 3D complexity surface."""
        # Parametric surface: (d, V×cos(θ), V×sin(θ)) colored by S
        theta = np.linspace(0, 2 * PI, theta_resolution)
        D, Theta = np.meshgrid(self.d_range[::10], theta)
        
        # Sample volumes and surfaces at mesh points
        d_sample = self.d_range[::10]
        V_sample = self.volumes[::10]
        S_sample = self.surfaces[::10]
        C_sample = self.complexity[::10]
        
        # Create surface coordinates
        X = D
        Y = np.outer(np.ones(theta_resolution), V_sample) * np.cos(Theta)
        Z = np.outer(np.ones(theta_resolution), V_sample) * np.sin(Theta)
        
        # Color by complexity
        Colors = np.outer(np.ones(theta_resolution), C_sample)
        
        return X, Y, Z, Colors

class ComplexityPeakVisualizer:
    """Interactive visualizer for complexity peak exploration."""
    
    def __init__(self):
        self.explorer = ComplexityPeakExplorer()
        
    def create_figure(self):
        """Create the interactive figure."""
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.suptitle('The Complexity Peak: Where Reality Maximizes Information Capacity',
                         fontsize=14, fontweight='bold')
        
        # Create subplot layout
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, hspace=0.25, wspace=0.2)
        
        # Main 3D complexity surface
        self.ax_main = self.fig.add_subplot(gs[0, :], projection='3d')
        setup_3d_axis(self.ax_main, "The Fundamental Complexity Landscape")
        
        # 2D analysis plots
        self.ax_measures = self.fig.add_subplot(gs[1, 0])
        self.ax_stability = self.fig.add_subplot(gs[1, 1])
        
        # Controls
        self._create_controls()
        
        # Initial plot
        self.update_all_plots()
        
        print("COMPLEXITY PEAK EXPLORER")
        print("=" * 60)
        print("THE FUNDAMENTAL TRUTH:")
        print(f"  Maximum complexity at d = {self.explorer.peak_dimension:.3f}")
        print("  This is NOT coincidence - it's where the universe")
        print("  balances interior capacity (volume) with boundary")
        print("  interface (surface) for maximum information processing.")
        print()
        print("OUR UNIVERSE (d=4) sits just below this peak in the")
        print("stable region, allowing rich physics while maintaining")
        print("coherence. Higher dimensions (d>6) become compressed.")
        print()
        print("CONTROLS:")
        print("• Focus slider: Zoom into specific dimensional region")
        print("• Checkboxes: Toggle different visualizations")
        print("• Peak button: Jump to exact complexity maximum")
        print("• Universe button: Jump to our universe (d=4)")
        
    def _create_controls(self):
        """Create interactive controls."""
        # Focus dimension slider
        ax_focus = plt.axes([0.15, 0.02, 0.3, 0.03])
        self.focus_slider = Slider(ax_focus, 'Focus', 
                                  self.explorer.focus_range[0], 
                                  self.explorer.focus_range[1],
                                  valinit=self.explorer.peak_dimension)
        self.focus_slider.on_changed(self.on_focus_change)
        
        # Control buttons
        ax_peak = plt.axes([0.5, 0.02, 0.08, 0.03])
        ax_universe = plt.axes([0.6, 0.02, 0.08, 0.03])
        ax_analyze = plt.axes([0.7, 0.02, 0.08, 0.03])
        
        self.btn_peak = Button(ax_peak, 'Peak')
        self.btn_universe = Button(ax_universe, 'Universe')
        self.btn_analyze = Button(ax_analyze, 'Analyze')
        
        self.btn_peak.on_clicked(self.goto_peak)
        self.btn_universe.on_clicked(self.goto_universe)
        self.btn_analyze.on_clicked(self.detailed_analysis)
        
        # Visualization toggles
        ax_checks = plt.axes([0.02, 0.3, 0.12, 0.25])
        self.checkboxes = CheckButtons(ax_checks,
                                     ['Volume', 'Surface', 'Complexity', 'Derivatives', 'Stability'],
                                     [self.explorer.show_volume, self.explorer.show_surface,
                                      self.explorer.show_complexity, self.explorer.show_derivatives,
                                      self.explorer.show_stability])
        self.checkboxes.on_clicked(self.on_toggle_display)
    
    def on_focus_change(self, val):
        """Handle focus dimension change."""
        self.explorer.focus_dim = val
        self.update_all_plots()
    
    def goto_peak(self, event):
        """Jump to complexity peak."""
        self.focus_slider.set_val(self.explorer.peak_dimension)
        self.update_all_plots()
    
    def goto_universe(self, event):
        """Jump to our universe (d=4)."""
        self.focus_slider.set_val(4.0)
        self.update_all_plots()
    
    def detailed_analysis(self, event):
        """Print detailed analysis of current focus."""
        d = self.explorer.focus_dim
        v = self.explorer.measures.ball_volume(d)
        s = self.explorer.measures.sphere_surface(d)
        c = v * s
        r = s / v
        
        print(f"\nDETAILED ANALYSIS at d = {d:.3f}")
        print("=" * 50)
        print(f"Volume (V):      {v:.6f}")
        print(f"Surface (S):     {s:.6f}")
        print(f"Complexity (C):  {c:.6f}")
        print(f"Ratio (S/V):     {r:.6f}")
        print(f"Peak ratio:      {c / self.explorer.peak_complexity:.6f}")
        
        # Distance from critical points
        print(f"\nDistances from critical points:")
        print(f"  From π:        {abs(d - PI):.3f}")
        print(f"  From 2π:       {abs(d - 2*PI):.3f}")
        print(f"  From peak:     {abs(d - self.explorer.peak_dimension):.3f}")
        print(f"  From φ:        {abs(d - PHI):.3f}")
        print(f"  From e:        {abs(d - E):.3f}")
        
        # Stability classification
        if d < PI:
            stability = "STABLE - Linear expansion phase"
        elif d < 2*PI:
            stability = "TRANSITION - Approaching compression"
        elif d < self.explorer.peak_dimension:
            stability = "PRE-PEAK - Building complexity"
        elif d < self.explorer.peak_dimension * 1.2:
            stability = "PEAK ZONE - Maximum complexity"
        else:
            stability = "COMPRESSION - Shell concentration"
        
        print(f"\nStability:       {stability}")
    
    def on_toggle_display(self, label):
        """Handle display toggles."""
        if label == 'Volume':
            self.explorer.show_volume = not self.explorer.show_volume
        elif label == 'Surface':
            self.explorer.show_surface = not self.explorer.show_surface
        elif label == 'Complexity':
            self.explorer.show_complexity = not self.explorer.show_complexity
        elif label == 'Derivatives':
            self.explorer.show_derivatives = not self.explorer.show_derivatives
        elif label == 'Stability':
            self.explorer.show_stability = not self.explorer.show_stability
        
        self.update_all_plots()
    
    def update_all_plots(self):
        """Update all visualization panels."""
        self.update_main_3d()
        self.update_measures_plot()
        self.update_stability_plot()
        self.fig.canvas.draw_idle()
    
    def update_main_3d(self):
        """Update the main 3D complexity surface."""
        self.ax_main.clear()
        setup_3d_axis(self.ax_main, "The Fundamental Complexity Landscape")
        
        # Get complexity surface
        X, Y, Z, Colors = self.explorer.get_complexity_surface()
        
        # Main complexity surface
        if self.explorer.show_complexity:
            # Normalize colors
            norm_colors = Colors / np.max(Colors)
            surface_colors = cm.plasma(norm_colors)
            
            self.ax_main.plot_surface(X, Y, Z, facecolors=surface_colors,
                                     alpha=0.7, linewidth=0, antialiased=True)
        
        # Volume and surface contours
        if self.explorer.show_volume or self.explorer.show_surface:
            theta = np.linspace(0, 2*PI, 60)
            
            for i, d in enumerate(self.explorer.d_range[::50]):
                v = self.explorer.volumes[i*50]
                s = self.explorer.surfaces[i*50]
                
                if self.explorer.show_volume:
                    # Volume contour (inner)
                    x_v = np.full_like(theta, d)
                    y_v = v * np.cos(theta) * 0.5
                    z_v = v * np.sin(theta) * 0.5
                    self.ax_main.plot(x_v, y_v, z_v, 'b-', alpha=0.4, linewidth=2)
                
                if self.explorer.show_surface:
                    # Surface contour (outer)
                    x_s = np.full_like(theta, d)
                    y_s = s * np.cos(theta) * 0.3
                    z_s = s * np.sin(theta) * 0.3
                    self.ax_main.plot(x_s, y_s, z_s, 'g-', alpha=0.4, linewidth=2)
        
        # Mark the complexity peak
        peak_v = self.explorer.measures.ball_volume(self.explorer.peak_dimension)
        peak_theta = np.linspace(0, 2*PI, 100)
        peak_x = np.full_like(peak_theta, self.explorer.peak_dimension)
        peak_y = peak_v * np.cos(peak_theta) * 0.8
        peak_z = peak_v * np.sin(peak_theta) * 0.8
        
        self.ax_main.plot(peak_x, peak_y, peak_z, 'gold', linewidth=5, alpha=0.9)
        
        # Peak marker
        self.ax_main.scatter([self.explorer.peak_dimension], [0], [0],
                           c='gold', s=300, marker='*',
                           edgecolors='black', linewidth=3)
        
        # Critical dimension markers
        for d_crit, label, color in [
            (PI, 'π', 'red'),
            (2*PI, '2π', 'orange'),
            (4.0, 'Universe', 'cyan'),
            (PHI, 'φ', 'green'),
            (E, 'e', 'blue')
        ]:
            if self.explorer.focus_range[0] <= d_crit <= self.explorer.focus_range[1]:
                self.ax_main.scatter([d_crit], [0], [0],
                                   c=color, s=150, marker='o',
                                   edgecolors='black', linewidth=2)
                self.ax_main.text(d_crit, 0, 0.5, f' {label}',
                                 fontsize=10, color=color)
        
        # Current focus marker
        focus_v = self.explorer.measures.ball_volume(self.explorer.focus_dim)
        self.ax_main.scatter([self.explorer.focus_dim], [0], [0],
                           c='red', s=200, marker='x',
                           linewidth=4)
        
        # Info text
        focus_c = self.explorer.measures.complexity_measure(self.explorer.focus_dim)
        info_text = (f"Focus: d = {self.explorer.focus_dim:.3f}\n"
                    f"Complexity: {focus_c:.4f}\n"
                    f"Peak ratio: {focus_c/self.explorer.peak_complexity:.3f}")
        
        self.ax_main.text2D(0.02, 0.98, info_text, transform=self.ax_main.transAxes,
                           fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        # Set limits
        max_v = np.max(self.explorer.volumes)
        self.ax_main.set_xlim(self.explorer.focus_range)
        self.ax_main.set_ylim([-max_v*0.9, max_v*0.9])
        self.ax_main.set_zlim([-max_v*0.9, max_v*0.9])
        
        self.ax_main.set_xlabel('Dimension')
        self.ax_main.set_ylabel('Geometric Y')
        self.ax_main.set_zlabel('Geometric Z')
    
    def update_measures_plot(self):
        """Update the 2D measures analysis plot."""
        self.ax_measures.clear()
        self.ax_measures.set_title('Geometric Measures Analysis', fontsize=12)
        
        # Plot measures
        if self.explorer.show_volume:
            self.ax_measures.plot(self.explorer.d_range, self.explorer.volumes,
                                 'b-', linewidth=2, label='Volume', alpha=0.7)
        
        if self.explorer.show_surface:
            self.ax_measures.plot(self.explorer.d_range, self.explorer.surfaces,
                                 'g-', linewidth=2, label='Surface', alpha=0.7)
        
        if self.explorer.show_complexity:
            self.ax_measures.plot(self.explorer.d_range, self.explorer.complexity,
                                 'r-', linewidth=3, label='Complexity (V×S)')
        
        # Mark the peak
        self.ax_measures.scatter([self.explorer.peak_dimension], [self.explorer.peak_complexity],
                                c='gold', s=200, marker='*', zorder=5,
                                edgecolors='black', linewidth=2)
        
        # Critical dimensions
        y_max = max(np.max(self.explorer.complexity), np.max(self.explorer.surfaces))
        for d_crit, label, color in [(PI, 'π', 'red'), (2*PI, '2π', 'orange'),
                                     (4.0, 'Universe', 'cyan')]:
            if self.explorer.focus_range[0] <= d_crit <= self.explorer.focus_range[1]:
                self.ax_measures.axvline(d_crit, color=color, linestyle='--', alpha=0.5)
                self.ax_measures.text(d_crit, y_max*0.9, f' {label}',
                                     rotation=90, color=color, fontsize=9)
        
        # Focus line
        self.ax_measures.axvline(self.explorer.focus_dim, color='red', linewidth=2, alpha=0.7)
        
        self.ax_measures.set_xlabel('Dimension')
        self.ax_measures.set_ylabel('Measure Value')
        self.ax_measures.legend()
        self.ax_measures.grid(True, alpha=0.3)
        self.ax_measures.set_xlim(self.explorer.focus_range)
    
    def update_stability_plot(self):
        """Update the stability regions analysis."""
        self.ax_stability.clear()
        self.ax_stability.set_title('Stability Regions', fontsize=12)
        
        # Stability regions
        regions = self.explorer.get_stability_regions()
        region_colors = [r[1] for r in regions]
        
        # Plot complexity with stability coloring
        for i in range(len(self.explorer.d_range) - 1):
            d1, d2 = self.explorer.d_range[i], self.explorer.d_range[i+1]
            c1, c2 = self.explorer.complexity[i], self.explorer.complexity[i+1]
            color = region_colors[i]
            
            self.ax_stability.plot([d1, d2], [c1, c2], color=color, linewidth=3, alpha=0.7)
        
        # Add region labels
        region_labels = {}
        for i, (region, color) in enumerate(regions):
            if region not in region_labels:
                d = self.explorer.d_range[i]
                c = self.explorer.complexity[i]
                region_labels[region] = (d, c, color)
        
        for region, (d, c, color) in region_labels.items():
            self.ax_stability.text(d, c, f' {region}', color=color, fontsize=9,
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Peak marker
        self.ax_stability.scatter([self.explorer.peak_dimension], [self.explorer.peak_complexity],
                                 c='gold', s=200, marker='*', zorder=5,
                                 edgecolors='black', linewidth=2)
        
        # Focus marker
        focus_c = self.explorer.measures.complexity_measure(self.explorer.focus_dim)
        self.ax_stability.scatter([self.explorer.focus_dim], [focus_c],
                                 c='red', s=150, marker='x', linewidth=3, zorder=5)
        
        self.ax_stability.set_xlabel('Dimension')
        self.ax_stability.set_ylabel('Complexity (V×S)')
        self.ax_stability.grid(True, alpha=0.3)
        self.ax_stability.set_xlim(self.explorer.focus_range)
    
    def run(self):
        """Run the interactive visualization."""
        self.create_figure()
        plt.show()

def main():
    """Launch the complexity peak explorer."""
    visualizer = ComplexityPeakVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()