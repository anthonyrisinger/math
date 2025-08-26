#!/usr/bin/env python3
"""
Dimensional Emergence Framework
Computational implementation of phase-driven dimensional genesis
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.special import gamma, gammaln

warnings.filterwarnings("ignore")

# Mathematical Constants
PHI = (1 + np.sqrt(5)) / 2
PSI = 1 / PHI
VARPI = gamma(0.25) ** 2 / (2 * np.sqrt(2 * np.pi))  # 1.311028777...
PI = np.pi

# Viewing parameters
ELEV = np.degrees(PHI - 1)  # 36.87°
AZIM = -45


class DimensionalFramework:
    """
    Core computational engine for dimensional emergence via phase dynamics
    """

    def __init__(self, max_dim=24, dt=0.001):
        self.max_dim = max_dim
        self.dt = dt
        self.time = 0.0

        # Phase state (complex for rotational dynamics)
        self.phase_density = np.zeros(max_dim, dtype=complex)
        self.phase_density[0] = 1.0  # Initial singularity

        # Dimensional tracking
        self.emerged = set([0])
        self.clock_rates = np.ones(max_dim)

        # Kissing numbers (angular quantization thresholds)
        self.kissing = {
            1: 2,
            2: 6,
            3: 12,
            4: 24,
            5: 40,
            6: 72,
            7: 126,
            8: 240,
            9: 272,
            10: 336,
            11: 438,
            12: 756,
            16: 4320,
            24: 196560,
        }

    @staticmethod
    def n_ball_volume(d, safe=True):
        """
        Volume of unit d-ball: π^(d/2) / Γ(d/2 + 1)
        Works for complex d; handles edge cases properly
        """
        if np.abs(d) < 1e-10:
            return 1.0

        if safe and np.real(d) > 170:  # Avoid overflow
            # Use Stirling approximation in log space
            log_vol = (d / 2) * np.log(PI) - gammaln(d / 2 + 1)
            return np.exp(log_vol)

        return PI ** (d / 2) / gamma(d / 2 + 1)

    @staticmethod
    def n_sphere_surface(d, safe=True):
        """
        Surface area of unit d-sphere: 2π^((d+1)/2) / Γ((d+1)/2)
        Note: This is correct formula, previous had error
        """
        if np.abs(d) < 1e-10:
            return 2.0  # S^0 = two points

        if safe and np.real(d) > 170:
            # Use log space for large d
            log_surf = np.log(2) + ((d + 1) / 2) * np.log(PI) - gammaln((d + 1) / 2)
            return np.exp(log_surf)

        # Correct formula: S^(d-1) embedded in R^d
        return 2 * PI ** ((d + 1) / 2) / gamma((d + 1) / 2)

    def phase_capacity(self, d):
        """Phase capacity Λ(d) - threshold for emergence"""
        return np.abs(self.n_ball_volume(d))

    def angular_resolution(self, d):
        """Minimum distinguishable angle - drives quantization"""
        d_int = int(np.real(d))
        k = self.kissing.get(d_int, 2 * d_int)
        return 2 * PI / k

    def phase_sapping_rate(self, source, target):
        """Rate at which dimension 'target' saps from 'source'"""
        if source >= target:
            return 0.0

        # Energy cost increases with dimensional distance
        distance_factor = 1.0 / (target - source + PHI)

        # Higher dimensions have higher frequency
        frequency_ratio = np.sqrt((target + 1) / (source + 1))

        # Phase deficit drives sapping
        deficit = self.phase_capacity(target) - np.abs(self.phase_density[target])
        deficit = max(0, deficit)

        return deficit * distance_factor * frequency_ratio

    def evolve(self):
        """Single evolution step"""
        self.time += self.dt

        # Involution: complex conjugation on even dimensions
        for d in range(0, self.max_dim, 2):
            self.phase_density[d] = np.conj(self.phase_density[d])

        # Phase sapping dynamics
        transfers = np.zeros(self.max_dim, dtype=complex)

        for target in range(1, self.max_dim):
            for source in range(target):
                if np.abs(self.phase_density[source]) > 1e-6:
                    rate = self.phase_sapping_rate(source, target) * self.dt

                    # Angular quantization affects transfer
                    angle = self.angular_resolution(target)
                    rotation = np.exp(1j * angle * self.time)

                    transfer = self.phase_density[source] * rate * rotation
                    transfers[source] -= transfer
                    transfers[target] += transfer

                    # Clock rate modulation
                    self.clock_rates[source] *= 1 - rate / 100

        self.phase_density += transfers

        # Check emergence conditions
        for d in range(self.max_dim):
            if np.abs(self.phase_density[d]) >= self.phase_capacity(d):
                if d not in self.emerged:
                    self.emerged.add(d)
                    # Seed next dimension
                    if d + 1 < self.max_dim:
                        self.phase_density[d + 1] += 0.01 * np.exp(1j * PI / 4)

    def run_simulation(self, steps=5000, injections=None):
        """Run complete simulation with optional energy injections"""
        history = {"time": [], "phase": [], "emerged": [], "total_phase": []}

        for step in range(steps):
            self.evolve()

            # Energy injections
            if injections:
                for t_inject, d_inject, amount in injections:
                    if abs(self.time - t_inject) < self.dt:
                        self.phase_density[d_inject] += amount

            # Record periodically
            if step % 10 == 0:
                history["time"].append(self.time)
                history["phase"].append(self.phase_density.copy())
                history["emerged"].append(len(self.emerged))
                history["total_phase"].append(np.sum(np.abs(self.phase_density)))

        return history


def analyze_geometric_structure():
    """
    Complete analysis of n-ball/n-sphere geometry
    Identifies critical dimensions and phase transitions
    """

    # High-resolution dimension scan
    d_continuous = np.linspace(0.01, 15, 2000)

    # Compute measures (handling edge cases)
    volumes = []
    surfaces = []

    for d in d_continuous:
        v = DimensionalFramework.n_ball_volume(d, safe=True)
        s = DimensionalFramework.n_sphere_surface(d, safe=True)
        volumes.append(v)
        surfaces.append(s)

    volumes = np.array(volumes)
    surfaces = np.array(surfaces)

    # Find critical points
    vol_max_idx = np.argmax(volumes)
    surf_max_idx = np.argmax(surfaces)

    critical_dims = {
        "volume_peak": d_continuous[vol_max_idx],
        "surface_peak": d_continuous[surf_max_idx],
        "volume_peak_value": volumes[vol_max_idx],
        "surface_peak_value": surfaces[surf_max_idx],
        "pi_boundary": PI,
        "2pi_boundary": 2 * PI,
        "4pi_boundary": 4 * PI,
        "volume_unity_crossings": [],
    }

    # Find where volume = 1
    for i in range(len(volumes) - 1):
        if (volumes[i] - 1) * (volumes[i + 1] - 1) < 0:
            # Linear interpolation for crossing point
            d_cross = d_continuous[i] + (1 - volumes[i]) * (
                d_continuous[i + 1] - d_continuous[i]
            ) / (volumes[i + 1] - volumes[i])
            critical_dims["volume_unity_crossings"].append(d_cross)

    # Fractional dimensions analysis
    special_d = {
        "0": 0.0,
        "1/2": 0.5,
        "1": 1.0,
        "φ": PHI,
        "e": np.e,
        "π": PI,
        "ϖ": VARPI,
        "2π": 2 * PI,
    }

    fractional_analysis = {}
    for name, d in special_d.items():
        if d > 0:  # Avoid d=0 singularity
            fractional_analysis[name] = {
                "d": d,
                "volume": DimensionalFramework.n_ball_volume(d),
                "surface": DimensionalFramework.n_sphere_surface(d),
                "ratio": DimensionalFramework.n_sphere_surface(d)
                / DimensionalFramework.n_ball_volume(d),
            }

    return d_continuous, volumes, surfaces, critical_dims, fractional_analysis


def create_comprehensive_visualization():
    """
    Professional visualization suite for dimensional emergence
    """

    # Run analysis
    d_cont, volumes, surfaces, critical, fractional = analyze_geometric_structure()

    # Initialize framework and run simulation
    framework = DimensionalFramework(max_dim=13)

    # Strategic energy injections to trigger emergence
    injections = [(1.0, 0, 0.5), (2.0, 1, 0.3), (3.0, 2, 0.2), (4.0, 3, 0.15)]

    history = framework.run_simulation(steps=5000, injections=injections)

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.25)

    # ========== Main 3D Plot ==========
    ax1 = fig.add_subplot(gs[:, 0], projection="3d")
    ax1.set_proj_type("ortho")
    ax1.view_init(elev=ELEV, azim=AZIM)
    ax1.set_box_aspect((1, 1, 1))

    # Plot (dimension, volume, surface) trajectory
    ax1.plot(d_cont, volumes, surfaces, "b-", linewidth=1.5, alpha=0.8)

    # Mark critical points
    ax1.scatter(
        [critical["volume_peak"]],
        [critical["volume_peak_value"]],
        [DimensionalFramework.n_sphere_surface(critical["volume_peak"])],
        c="red",
        s=100,
        marker="o",
        label=f"V peak: d={critical['volume_peak']:.3f}",
    )

    ax1.scatter(
        [critical["surface_peak"]],
        [DimensionalFramework.n_ball_volume(critical["surface_peak"])],
        [critical["surface_peak_value"]],
        c="green",
        s=100,
        marker="s",
        label=f"S peak: d={critical['surface_peak']:.3f}",
    )

    # π-boundaries as vertical planes
    pi_dims = [PI, 2 * PI]
    for pi_d in pi_dims:
        v_range = np.linspace(0, 6, 10)
        s_range = np.linspace(0, 35, 10)
        V, S = np.meshgrid(v_range, s_range)
        D = np.ones_like(V) * pi_d
        ax1.plot_surface(D, V, S, alpha=0.1, color="orange")

    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("Volume")
    ax1.set_zlabel("Surface Area")
    ax1.set_title(f"Geometric Structure\n(View: {ELEV:.1f}°, {AZIM}°)")
    ax1.legend(loc="upper left", fontsize=8)

    # ========== Volume/Surface Plot ==========
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.plot(d_cont, volumes, "b-", linewidth=2, label="Volume")
    ax2.plot(d_cont, surfaces, "g-", linewidth=2, label="Surface")

    # Critical dimensions
    for d_crit in [PI, 2 * PI, 4 * PI]:
        ax2.axvline(x=d_crit, color="red", alpha=0.3, linestyle="--")
        ax2.text(
            d_crit,
            ax2.get_ylim()[1] * 0.9,
            f"{d_crit/PI:.0f}π",
            ha="center",
            fontsize=8,
        )

    ax2.axvline(x=critical["volume_peak"], color="blue", alpha=0.5, linestyle=":")
    ax2.axvline(x=critical["surface_peak"], color="green", alpha=0.5, linestyle=":")
    ax2.axhline(y=1, color="black", alpha=0.3, linestyle="-")

    ax2.set_xlabel("Dimension")
    ax2.set_ylabel("Measure")
    ax2.set_title("n-Ball Volume and n-Sphere Surface Area")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 15])
    ax2.set_ylim([0, 10])

    # ========== Phase Evolution ==========
    ax3 = fig.add_subplot(gs[1, 1])

    times = history["time"]
    phase_mags = np.array([np.abs(p) for p in history["phase"]])

    # Plot first few dimensions
    for d in range(min(6, phase_mags.shape[1])):
        if np.any(phase_mags[:, d] > 0.01):
            ax3.plot(times, phase_mags[:, d], label=f"d={d}", alpha=0.7)

    ax3.set_xlabel("Time")
    ax3.set_ylabel("|ρ(d)|")
    ax3.set_title("Phase Density Evolution")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ========== Emergence Timeline ==========
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(times, history["emerged"], "r-", linewidth=2)
    ax4.fill_between(times, 0, history["emerged"], alpha=0.3, color="red")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Emerged Dimensions")
    ax4.set_title("Dimensional Cascade")
    ax4.grid(True, alpha=0.3)

    # ========== Fractional Dimensions ==========
    ax5 = fig.add_subplot(gs[2, 1])

    frac_names = list(fractional.keys())
    frac_vols = [fractional[k]["volume"] for k in frac_names]
    frac_surfs = [fractional[k]["surface"] for k in frac_names]

    x = np.arange(len(frac_names))
    width = 0.35

    ax5.bar(x - width / 2, frac_vols, width, label="Volume", alpha=0.7, color="blue")
    ax5.bar(x + width / 2, frac_surfs, width, label="Surface", alpha=0.7, color="green")

    ax5.set_xlabel("Dimension")
    ax5.set_ylabel("Measure")
    ax5.set_title("Fractional Dimensions")
    ax5.set_xticks(x)
    ax5.set_xticklabels(frac_names)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ========== Angular Quantization ==========
    ax6 = fig.add_subplot(gs[2, 2])

    dims_quant = range(1, 13)
    kissing_nums = [framework.kissing.get(d, 2 * d) for d in dims_quant]
    angular_res = [2 * PI / k for k in kissing_nums]

    ax6.bar(dims_quant, angular_res, alpha=0.7, color="purple")
    ax6.set_xlabel("Dimension")
    ax6.set_ylabel("Angular Resolution (radians)")
    ax6.set_title("Quantization Threshold")
    ax6.grid(True, alpha=0.3)

    # Add second y-axis for kissing numbers
    ax6_twin = ax6.twinx()
    ax6_twin.plot(dims_quant, kissing_nums, "ro-", alpha=0.5, markersize=4)
    ax6_twin.set_ylabel("Kissing Number", color="red")
    ax6_twin.tick_params(axis="y", labelcolor="red")

    # Summary statistics
    fig.text(
        0.02,
        0.02,
        f"Critical Dimensions:\n"
        f'  Volume peak: {critical["volume_peak"]:.3f}\n'
        f'  Surface peak: {critical["surface_peak"]:.3f}\n'
        f'  Unity crossings: {[f"{x:.3f}" for x in critical["volume_unity_crossings"]]}\n'
        f"Constants: ϖ={VARPI:.5f}, φ={PHI:.5f}",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.suptitle(
        "Dimensional Emergence Framework: Phase-Driven Genesis",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    return framework, history, critical, fractional


if __name__ == "__main__":
    print("DIMENSIONAL EMERGENCE FRAMEWORK")
    print("=" * 50)
    print("Core Constants:")
    print(f"  ϖ = {VARPI:.10f}")
    print(f"  φ = {PHI:.10f}")
    print(f"  ψ = {PSI:.10f}")
    print()

    framework, history, critical, fractional = create_comprehensive_visualization()

    print("\nCritical Dimensions:")
    print(f"  Volume peak: d = {critical['volume_peak']:.6f}")
    print(f"  Surface peak: d = {critical['surface_peak']:.6f}")
    print(f"  Volume = 1 at: d = {critical['volume_unity_crossings']}")
    print()

    print("Fractional Dimension Analysis:")
    for name, data in fractional.items():
        print(
            f"  d={name:3} ({data['d']:6.4f}): V={data['volume']:8.4f}, "
            f"S={data['surface']:8.4f}, S/V={data['ratio']:8.4f}"
        )
    print()

    print("Framework State:")
    print(f"  Emerged dimensions: {sorted(framework.emerged)}")
    print(f"  Total phase: {np.sum(np.abs(framework.phase_density)):.4f}")
    print(
        f"  Clock rate range: [{min(framework.clock_rates):.4f}, "
        f"{max(framework.clock_rates):.4f}]"
    )
