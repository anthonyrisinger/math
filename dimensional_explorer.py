#!/usr/bin/env python3
"""
Dimensional Explorer - Unified Launcher
========================================

Central hub for exploring the dimensional emergence framework.
Launch any visualization or explore the mathematical foundations.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from core_measures import DimensionalMeasures, PHI, PI, PSI, VARPI

class DimensionalExplorer:
    """Main explorer interface for the dimensional emergence framework."""

    def __init__(self):
        self.measures = DimensionalMeasures()
        self.fig = None

    def create_interface(self):
        """Create the main launcher interface."""
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('Dimensional Emergence Framework Explorer',
                         fontsize=16, fontweight='bold')

        # Main info display
        ax_info = plt.axes([0.1, 0.4, 0.8, 0.5])
        ax_info.axis('off')

        # Create info text
        info_text = """
DIMENSIONAL EMERGENCE FRAMEWORK
═══════════════════════════════

Core Insight: Dimension is not a container for physics - dimension IS physics.
             Time emerges from dimensional change through dt/dd = φ (golden ratio).

Key Discoveries:
• V×S complexity peaks at d ≈ 6 (maximum information capacity)
• Our universe (d=4) sits optimally below this peak
• Higher dimensions "feed" on lower ones through phase sapping
• Quantum mechanics emerges from angular quantization

Critical Dimensions:
• d = 0: The void (pure potential, ρ=1)
• d = π ≈ 3.14: First stability boundary
• d = 4: Our universe (3+1 spacetime)
• d ≈ 6: Complexity peak (V×S maximum)
• d = 2π ≈ 6.28: Compression boundary
• d = 24: Leech lattice limit

Fundamental Constants:
• φ = 1.618... (golden ratio - recursive proportion)
• ψ = 0.618... (golden conjugate)
• ϖ = 1.311... (dimensional coupling constant)
• π boundaries mark phase transitions
"""

        ax_info.text(0.05, 0.5, info_text, transform=ax_info.transAxes,
                    fontsize=11, family='monospace', verticalalignment='center')

        # Visualization launcher buttons
        button_width = 0.15
        button_height = 0.04
        button_spacing = 0.02
        start_x = 0.1
        start_y = 0.25

        # Create launch buttons for each visualization
        visualizations = [
            ("Landscape", "dimensional_landscape", "Basic geometric measures in 3D"),
            ("Phase Flow", "phase_dynamics", "Energy sapping between dimensions"),
            ("Emergence", "emergence_cascade", "Sequential dimensional birth"),
            ("Peak Explorer", "complexity_peak", "Deep dive into V×S peak"),
            ("Constants", None, "Explore mathematical constants"),
        ]

        self.buttons = []
        for i, (name, module, description) in enumerate(visualizations):
            x = start_x + (i % 3) * (button_width + button_spacing)
            y = start_y - (i // 3) * (button_height + button_spacing)

            ax_btn = plt.axes([x, y, button_width, button_height])
            btn = Button(ax_btn, name)

            if module:
                btn.on_clicked(lambda event, m=module: self.launch_visualization(m))
            else:
                btn.on_clicked(self.show_constants)

            self.buttons.append(btn)

            # Add description below button
            ax_desc = plt.axes([x, y - 0.02, button_width, 0.02])
            ax_desc.axis('off')
            ax_desc.text(0.5, 0.5, description, transform=ax_desc.transAxes,
                        fontsize=8, ha='center', va='center', style='italic')

        # Mathematical analysis section
        ax_math = plt.axes([0.1, 0.05, 0.35, 0.15])
        ax_math.axis('off')

        # Compute some key values
        v4 = self.measures.ball_volume(4)
        s4 = self.measures.sphere_surface(4)
        c4 = v4 * s4

        peak_d, peak_val = self.measures.find_peak(self.measures.complexity_measure)

        math_text = f"""Current Analysis (d=4, Our Universe):
─────────────────────────────────
Volume:     {v4:.6f}
Surface:    {s4:.6f}
Complexity: {c4:.6f}
Peak ratio: {c4/peak_val:.3f}

Peak occurs at d = {peak_d:.3f}
Peak value = {peak_val:.3f}"""

        ax_math.text(0, 1, math_text, transform=ax_math.transAxes,
                    fontsize=10, family='monospace', va='top')

        # Quick formulas reference
        ax_formulas = plt.axes([0.55, 0.05, 0.35, 0.15])
        ax_formulas.axis('off')

        formulas_text = """Key Formulas:
─────────────
V_d = π^(d/2) / Γ(d/2 + 1)
S_d = 2π^(d/2) / Γ(d/2)
C_d = V_d × S_d

Phase Sapping:
R(s→t) = [Λ(t)-|ρ_t|]/[t-s+φ]
        × √[(t+1)/(s+1)]"""

        ax_formulas.text(0, 1, formulas_text, transform=ax_formulas.transAxes,
                        fontsize=10, family='monospace', va='top')

    def launch_visualization(self, module_name):
        """Launch a specific visualization module."""
        print(f"\nLaunching {module_name}...")
        print("=" * 50)

        # Import and run the module
        try:
            if module_name == "dimensional_landscape":
                from dimensional_landscape import main
                main()
            elif module_name == "phase_dynamics":
                from phase_dynamics import main
                main()
            elif module_name == "emergence_cascade":
                from emergence_cascade import main
                main()
            elif module_name == "complexity_peak":
                from complexity_peak import main
                main()
        except ImportError as e:
            print(f"Error: Could not import {module_name}")
            print(f"Details: {e}")
        except Exception as e:
            print(f"Error running {module_name}: {e}")

    def show_constants(self, event):
        """Show detailed constants analysis."""
        # Create new figure for constants
        fig_const = plt.figure(figsize=(12, 8))
        fig_const.suptitle('Dimensional Constants Explorer', fontsize=14, fontweight='bold')

        # Plot various constants and their relationships
        ax1 = fig_const.add_subplot(221)
        ax2 = fig_const.add_subplot(222)
        ax3 = fig_const.add_subplot(223)
        ax4 = fig_const.add_subplot(224)

        # 1. Volume and surface peaks
        d_range = np.linspace(0.1, 10, 1000)
        volumes = [self.measures.ball_volume(d) for d in d_range]
        surfaces = [self.measures.sphere_surface(d) for d in d_range]
        complexity = [v*s for v,s in zip(volumes, surfaces)]

        ax1.plot(d_range, volumes, 'b-', label='Volume', alpha=0.7)
        ax1.plot(d_range, surfaces, 'g-', label='Surface', alpha=0.7)
        ax1.plot(d_range, complexity, 'r-', label='V×S', linewidth=2)
        ax1.axvline(PI, color='red', linestyle='--', alpha=0.5, label='π')
        ax1.axvline(2*PI, color='orange', linestyle='--', alpha=0.5, label='2π')
        ax1.axvline(PHI, color='gold', linestyle='--', alpha=0.5, label='φ')
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Measure')
        ax1.set_title('Geometric Measures')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 10])

        # 2. Critical dimensions bar chart
        critical_dims = {
            'void': 0,
            'φ': PHI,
            'π': PI,
            'our universe': 4,
            'V peak': 5.256,
            'C peak': 6,
            '2π': 2*PI,
            'S peak': 7.256,
        }

        names = list(critical_dims.keys())
        values = list(critical_dims.values())
        colors_crit = ['black', 'gold', 'red', 'cyan', 'blue', 'purple', 'orange', 'green']

        ax2.bar(names, values, color=colors_crit, alpha=0.7)
        ax2.set_ylabel('Dimension')
        ax2.set_title('Critical Dimensions')
        ax2.tick_params(axis='x', rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Phase capacity thresholds
        dims = range(0, 8)
        capacities = [self.measures.ball_volume(d) for d in dims]

        ax3.plot(dims, capacities, 'o-', color='purple', linewidth=2, markersize=8)
        ax3.fill_between(dims, 0, capacities, alpha=0.3, color='purple')
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Phase Capacity Λ(d)')
        ax3.set_title('Emergence Thresholds')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(dims)

        # 4. Constants relationships
        ax4.axis('off')
        const_text = f"""Fundamental Constants:
══════════════════════

φ = {PHI:.10f}  (Golden Ratio)
ψ = {PSI:.10f}  (Golden Conjugate)
ϖ = {VARPI:.10f}  (Dimensional Coupling)

Key Relationships:
─────────────────
φ² = φ + 1 = {PHI**2:.6f}
ψ² = 1 - ψ = {PSI**2:.6f}
φ × ψ = 1 = {PHI * PSI:.6f}
φ - ψ = √5 = {PHI - PSI:.6f}

π Boundaries:
────────────
π = {PI:.10f}
2π = {2*PI:.10f}
√π = {np.sqrt(PI):.10f}

Special Values:
──────────────
Γ(1/2) = √π = {np.sqrt(PI):.6f}
Γ(1/4) = {np.math.gamma(0.25):.6f}
e^(π/4) = {np.exp(PI/4):.6f}
"""

        ax4.text(0.1, 0.9, const_text, transform=ax4.transAxes,
                fontsize=10, family='monospace', va='top')

        plt.tight_layout()
        plt.show()

    def run(self):
        """Run the explorer interface."""
        self.create_interface()
        plt.show()

def main():
    """Launch the dimensional explorer."""
    print("\n" + "="*60)
    print("DIMENSIONAL EMERGENCE FRAMEWORK EXPLORER")
    print("="*60)
    print("\nThis is the central hub for exploring how dimension creates reality.")
    print("Select any visualization to begin your exploration.")
    print("\nKey insight: Dimension is not a stage for physics -")
    print("             dimension IS the fundamental parameter.\n")

    explorer = DimensionalExplorer()
    explorer.run()

if __name__ == "__main__":
    main()