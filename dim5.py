#!/usr/bin/env python3
"""
Dimensional Emergence Framework - Corrected Implementation
Built from your actual mathematical framework
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Correct constants
def compute_varpi():
    """Γ(1/4)² / (2√(2π))"""
    return gamma(0.25)**2 / (2 * np.sqrt(2 * np.pi))

VARPI = compute_varpi()  # Should be ≈ 1.31102, not 2.62
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi

class DimensionalEmergence:
    """
    Correct implementation based on your framework:
    - Phase capacity Λ(d) = π^(d/2) / Γ(d/2 + 1)
    - Emergence when ∫ρ_d dV ≥ Λ(d)
    - Phase sapping from all lower dimensions
    """
    
    def __init__(self, dt=0.01):
        self.dt = dt
        self.time = 0.0
        
        # Start with dimension 0 (pure potential)
        self.current_dim = 0
        self.emerging_dim = None
        self.emergence_progress = 0.0  # 0 to 1 during emergence
        
        # Phase densities for dimensions 0-12
        self.max_dim = 13
        self.phase_density = np.zeros(self.max_dim)
        self.phase_density[0] = 1.0  # Initial singularity
        
        # Clock rates (all start at 1)
        self.clock_rates = np.ones(self.max_dim)
        
        self.history = []
        
    def phase_capacity(self, d):
        """Λ(d) = π^(d/2) / Γ(d/2 + 1)"""
        if d == 0:
            return 1.0
        return PI**(d/2) / gamma(d/2 + 1)
    
    def integrated_phase(self, d):
        """Total phase in dimension d"""
        # Since we're working with unit volumes, integral = density
        return self.phase_density[d]
    
    def can_emerge(self, d):
        """Check if dimension d can emerge"""
        if d >= self.max_dim or d < 0:
            return False
        return self.integrated_phase(d) >= self.phase_capacity(d)
    
    def phase_sap_rate(self, d_source, d_target):
        """Rate at which d_target saps from d_source"""
        if d_source >= d_target:
            return 0.0
        
        # Higher dimensions sap more aggressively
        # Rate proportional to the deficit needed
        deficit = self.phase_capacity(d_target) - self.integrated_phase(d_target)
        if deficit <= 0:
            return 0.0
            
        # Sapping rate decreases with dimensional distance
        distance_factor = 1.0 / (1 + (d_target - d_source))
        
        # Higher dimensions have higher "frequency" so sap faster
        frequency_ratio = np.sqrt((d_target + 1) / (d_source + 1))
        
        return deficit * distance_factor * frequency_ratio * self.dt
    
    def step(self):
        """Single evolution step"""
        
        # Check for emergence
        if self.emerging_dim is None:
            # Find the highest dimension that can emerge
            for d in range(self.current_dim + 1, self.max_dim):
                if d == self.current_dim + 1:  # Only allow sequential emergence
                    if self.can_emerge(self.current_dim):
                        self.emerging_dim = d
                        self.emergence_progress = 0.0
                        print(f"t={self.time:.3f}: Dimension {d} begins emerging")
                        break
        
        # Handle ongoing emergence
        if self.emerging_dim is not None:
            # Emerging dimension saps from all lower dimensions
            for d in range(self.emerging_dim):
                if self.phase_density[d] > 0:
                    sap_amount = self.phase_sap_rate(d, self.emerging_dim)
                    
                    # Can't sap more than available
                    sap_amount = min(sap_amount, self.phase_density[d])
                    
                    # Transfer phase
                    self.phase_density[d] -= sap_amount
                    self.phase_density[self.emerging_dim] += sap_amount
                    
                    # Slow down the sapped dimension's clock
                    self.clock_rates[d] *= (1 - sap_amount / max(self.phase_density[d], 0.01))
            
            # Update emergence progress (takes π time units)
            self.emergence_progress += self.dt / PI
            
            if self.emergence_progress >= 1.0:
                print(f"t={self.time:.3f}: Dimension {self.emerging_dim} fully emerged")
                self.current_dim = self.emerging_dim
                self.emerging_dim = None
                self.emergence_progress = 0.0
        
        # Record state
        self.history.append({
            'time': self.time,
            'current_dim': self.current_dim,
            'emerging_dim': self.emerging_dim,
            'phase_density': self.phase_density.copy(),
            'clock_rates': self.clock_rates.copy()
        })
        
        self.time += self.dt
    
    def run(self, steps=1000):
        """Run simulation"""
        print(f"\n🌀 Starting with VARPI = {VARPI:.5f}")
        print(f"   Phase capacities: Λ(3) = {self.phase_capacity(3):.4f}, "
              f"Λ(5) = {self.phase_capacity(5):.4f}")
        
        for _ in range(steps):
            self.step()
            
            # Add energy at specific times to trigger emergence
            if abs(self.time - 1.0) < self.dt:
                self.phase_density[0] += 2.0
                print(f"t={self.time:.3f}: Energy injection")
            if abs(self.time - 5.0) < self.dt:
                self.phase_density[1] += 1.5
            if abs(self.time - 10.0) < self.dt:
                self.phase_density[2] += 1.0
        
        return self.history

class GeometricAnalysis:
    """Analyze the geometric properties correctly"""
    
    @staticmethod
    def n_ball_volume(d):
        """V_n = π^(n/2) / Γ(n/2 + 1)"""
        if d == 0:
            return 1.0
        return PI**(d/2) / gamma(d/2 + 1)
    
    @staticmethod
    def n_sphere_surface(d):
        """S_n = 2π^(n/2) / Γ(n/2)"""
        if d <= 0:
            return 2.0 if d == 0 else 0.0
        return 2 * PI**(d/2) / gamma(d/2)
    
    @staticmethod
    def find_peaks():
        """Find where volume and surface peak"""
        dims = np.linspace(0.1, 15, 1000)
        volumes = [GeometricAnalysis.n_ball_volume(d) for d in dims]
        surfaces = [GeometricAnalysis.n_sphere_surface(d) for d in dims]
        
        vol_peak = dims[np.argmax(volumes)]
        surf_peak = dims[np.argmax(surfaces)]
        
        return vol_peak, surf_peak
    
    @staticmethod
    def critical_dimensions():
        """Find the critical dimensional boundaries"""
        results = {}
        
        # Where volume = 1
        for d in np.linspace(0, 20, 1000):
            v = GeometricAnalysis.n_ball_volume(d)
            if abs(v - 1.0) < 0.01:
                if 'volume_equals_1' not in results:
                    results['volume_equals_1'] = []
                results['volume_equals_1'].append(d)
        
        # π boundaries
        results['pi_boundary'] = PI
        results['2pi_boundary'] = 2 * PI
        results['4pi_boundary'] = 4 * PI
        
        return results

def visualize_framework():
    """Create correct visualizations"""
    
    # Run simulation
    sim = DimensionalEmergence(dt=0.01)
    history = sim.run(steps=2000)
    
    # Set up figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Dimensional Emergence: Corrected Framework', fontsize=14, fontweight='bold')
    
    # Extract history
    times = [h['time'] for h in history]
    current_dims = [h['current_dim'] for h in history]
    phase_densities = np.array([h['phase_density'] for h in history])
    
    # 1. Phase density evolution
    ax = axes[0, 0]
    for d in range(min(6, phase_densities.shape[1])):
        if np.any(phase_densities[:, d] > 0.01):
            ax.plot(times, phase_densities[:, d], label=f'd={d}', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Phase Density ρ(d)')
    ax.set_title('Phase Density Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Current dimension
    ax = axes[0, 1]
    ax.plot(times, current_dims, 'r-', linewidth=2)
    ax.fill_between(times, 0, current_dims, alpha=0.3, color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Current Dimension')
    ax.set_title('Dimensional Emergence')
    ax.grid(True, alpha=0.3)
    
    # 3. Volume and Surface
    ax = axes[0, 2]
    dims = np.linspace(0.1, 15, 1000)
    volumes = [GeometricAnalysis.n_ball_volume(d) for d in dims]
    surfaces = [GeometricAnalysis.n_sphere_surface(d) for d in dims]
    
    ax.plot(dims, volumes, 'b-', label='Volume', linewidth=2)
    ax.plot(dims, surfaces, 'g-', label='Surface', linewidth=2)
    
    vol_peak, surf_peak = GeometricAnalysis.find_peaks()
    ax.axvline(x=vol_peak, color='b', linestyle='--', alpha=0.5, label=f'V peak: {vol_peak:.2f}')
    ax.axvline(x=surf_peak, color='g', linestyle='--', alpha=0.5, label=f'S peak: {surf_peak:.2f}')
    ax.axvline(x=PI, color='r', linestyle='--', alpha=0.5, label='π')
    ax.axvline(x=2*PI, color='orange', linestyle='--', alpha=0.5, label='2π')
    
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Measure')
    ax.set_title('n-Ball/n-Sphere Geometry')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 10])
    
    # 4. Phase Capacity
    ax = axes[1, 0]
    dims_int = range(0, 13)
    capacities = [sim.phase_capacity(d) for d in dims_int]
    ax.bar(dims_int, capacities, alpha=0.7, color='purple')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Phase Capacity Λ(d)')
    ax.set_title('Phase Capacity by Dimension')
    ax.grid(True, alpha=0.3)
    
    # 5. Special dimensions
    ax = axes[1, 1]
    special_d = {
        '0': 0,
        '1/2': 0.5,
        '1': 1.0,
        'φ': PHI,
        'e': np.e,
        'π': PI,
        '2π': 2*PI,
        'ϖ': VARPI
    }
    
    special_v = [GeometricAnalysis.n_ball_volume(d) for d in special_d.values()]
    x_pos = range(len(special_d))
    
    bars = ax.bar(x_pos, special_v, alpha=0.7, color='teal')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(special_d.keys(), rotation=45)
    ax.set_ylabel('Volume')
    ax.set_title('Fractional Dimension Volumes')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, special_v)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Clock rates
    ax = axes[1, 2]
    final_clocks = history[-1]['clock_rates'][:6]
    dims_clock = range(len(final_clocks))
    ax.bar(dims_clock, final_clocks, alpha=0.7, color='green')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Clock Rate')
    ax.set_title('Final Clock Rates (Phase Sapping Effect)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print(f"\n📊 Analysis Complete:")
    print(f"   Volume peaks at d = {vol_peak:.3f}")
    print(f"   Surface peaks at d = {surf_peak:.3f}")
    print(f"   VARPI = {VARPI:.5f}")
    print(f"   Final dimension reached: {current_dims[-1]}")
    print(f"   Phase conservation: {np.sum(phase_densities[-1]):.3f}")

if __name__ == "__main__":
    print("="*60)
    print("DIMENSIONAL EMERGENCE - CORRECTED")
    print("="*60)
    visualize_framework()