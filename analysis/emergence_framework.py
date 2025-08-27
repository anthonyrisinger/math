#!/usr/bin/env python3
"""
Emergence Framework Module
==========================

Unified framework for dimensional emergence theory where dimension is the
fundamental parameter from which time and reality emerge. Consolidates the
theoretical frameworks from dim1-dim5 modules.

Core principles:
1. Dimension is continuous and primary
2. Time emerges from dimensional change (not the reverse)
3. Phase coherence drives dimensional emergence
4. The V×S product reveals WHERE complexity maximizes
5. Reality emerges from dimensional transitions
"""

from typing import Any

import numpy as np

from .geometric_measures import PHI, E, GeometricMeasures


class EmergenceFramework:
    """
    Complete framework where dimension generates reality.

    Time emerges from dimensional evolution, not the reverse.
    Phase coherence determines which dimensions can emerge.
    """

    def __init__(self, max_dimensions: int = 16):
        self.max_dimensions = max_dimensions

        # Dimension is THE fundamental variable
        self.dimension = 0.0  # Start at the void

        # Time emerges from dimensional evolution
        self.time = 0.0

        # Phase densities for integer dimensional levels
        self.phase_density = np.zeros(max_dimensions, dtype=complex)
        self.phase_density[0] = 1.0  # Initial unity at the void

        # Track which dimensions have emerged (achieved coherence)
        self.emerged = {0}

        # Energy flow matrix tracks phase sapping between dimensions
        self.flow_matrix = np.zeros((max_dimensions, max_dimensions))

        # Emergence thresholds
        self.coherence_threshold = 0.5
        self.emergence_energy = 1.0

        # History tracking
        self.history = {
            "time": [],
            "dimension": [],
            "phase_densities": [],
            "emerged_dims": [],
        }

    def dimensional_potential(self, d: float) -> float:
        """
        Potential energy landscape in dimensional space.

        Based on geometric complexity C(d) = V(d) × S(d).
        Peaks indicate stable dimensional states.
        """
        return GeometricMeasures.complexity_measure(d)

    def phase_evolution_rate(self, d: float) -> float:
        """
        Rate of phase evolution at dimension d.

        Higher complexity → faster phase evolution
        Critical dimensions have special behavior
        """
        complexity = self.dimensional_potential(d)

        # Add resonance effects at critical dimensions
        volume_peak = 5.257  # Approximate volume peak
        complexity_peak = 6.335  # Approximate complexity peak

        resonance = 0
        resonance += np.exp(-((d - volume_peak) ** 2) / 0.1)
        resonance += np.exp(-((d - complexity_peak) ** 2) / 0.1)
        resonance += np.exp(-((d - PHI) ** 2) / 0.1)  # Golden ratio resonance

        return complexity * (1 + 0.5 * resonance)

    def evolve_phase(self, dt: float = 0.01):
        """
        Evolve phase densities according to dimensional dynamics.

        Parameters
        ----------
        dt : float
            Time step for evolution
        """
        old_phases = self.phase_density.copy()

        for i in range(self.max_dimensions):
            d = float(i)

            # Phase evolution based on dimensional potential
            evolution_rate = self.phase_evolution_rate(d)
            phase_change = evolution_rate * dt * np.exp(1j * self.time)

            # Coupling to neighboring dimensions
            coupling = 0j
            if i > 0:
                coupling += 0.1 * old_phases[i - 1]  # From lower dimension
            if i < self.max_dimensions - 1:
                coupling += 0.1 * old_phases[i + 1]  # From higher dimension

            # Update phase density
            self.phase_density[i] += phase_change + coupling * dt

            # Check for emergence threshold
            if abs(self.phase_density[i]) > self.coherence_threshold:
                if i not in self.emerged:
                    self.emerged.add(i)
                    print(f"🌟 Dimension {i} has EMERGED at time {self.time:.3f}")

        # Normalize to conserve total phase
        total_phase = np.sum(np.abs(self.phase_density) ** 2)
        if total_phase > 0:
            self.phase_density /= np.sqrt(total_phase)

        # Update time (emerges from dimensional change)
        dimensional_change = np.sum(np.abs(self.phase_density - old_phases) ** 2)
        self.time += dt * (1 + dimensional_change)

        # Update effective dimension (weighted average)
        weights = np.abs(self.phase_density) ** 2
        if np.sum(weights) > 0:
            self.dimension = np.sum(weights * np.arange(self.max_dimensions)) / np.sum(
                weights
            )

    def run_emergence_simulation(
        self, steps: int = 1000, dt: float = 0.01
    ) -> dict[str, Any]:
        """
        Run complete emergence simulation.

        Parameters
        ----------
        steps : int
            Number of evolution steps
        dt : float
            Time step size

        Returns
        -------
        dict
            Simulation results and analysis
        """
        print("🚀 Starting dimensional emergence simulation...")

        # Reset state
        self.__init__(self.max_dimensions)

        # Add initial perturbation to seed emergence
        self.phase_density[1] = 0.1 * np.exp(1j * PHI)

        # Run simulation
        for step in range(steps):
            self.evolve_phase(dt)

            # Record history every 10 steps
            if step % 10 == 0:
                self.history["time"].append(self.time)
                self.history["dimension"].append(self.dimension)
                self.history["phase_densities"].append(self.phase_density.copy())
                self.history["emerged_dims"].append(self.emerged.copy())

        print(f"✅ Simulation complete! Final dimension: {self.dimension:.3f}")
        print(f"📊 Emerged dimensions: {sorted(self.emerged)}")

        return self._analyze_simulation()

    def _analyze_simulation(self) -> dict[str, Any]:
        """Analyze simulation results."""
        np.array(self.history["phase_densities"])

        # Find dimensional transitions
        dimension_history = np.array(self.history["dimension"])
        transitions = []

        for i in range(1, len(dimension_history)):
            change = abs(dimension_history[i] - dimension_history[i - 1])
            if change > 0.1:  # Significant dimensional change
                transitions.append(
                    {
                        "time": self.history["time"][i],
                        "from_dim": dimension_history[i - 1],
                        "to_dim": dimension_history[i],
                        "magnitude": change,
                    }
                )

        # Calculate emergence metrics
        final_phases = np.abs(self.phase_density) ** 2
        dimensional_entropy = -np.sum(final_phases * np.log(final_phases + 1e-12))

        return {
            "final_dimension": self.dimension,
            "emerged_dimensions": sorted(self.emerged),
            "total_emerged": len(self.emerged),
            "dimensional_entropy": dimensional_entropy,
            "major_transitions": transitions,
            "phase_distribution": final_phases,
            "history": self.history,
            "simulation_stats": {
                "total_time": self.time,
                "average_dimension": np.mean(dimension_history),
                "max_dimension": np.max(dimension_history),
                "dimensional_variance": np.var(dimension_history),
            },
        }

    def plot_emergence_evolution(self):
        """Plot the complete emergence evolution."""
        if not self.history["time"]:
            print("⚠️  No simulation history available. Run simulation first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Dimensional Emergence Evolution", fontsize=16)

        times = np.array(self.history["time"])
        dimensions = np.array(self.history["dimension"])
        phase_densities = np.array(self.history["phase_densities"])

        # Dimensional evolution
        axes[0, 0].plot(times, dimensions, "b-", linewidth=2)
        axes[0, 0].set_xlabel("Emergent Time")
        axes[0, 0].set_ylabel("Effective Dimension")
        axes[0, 0].set_title("Dimensional Evolution")
        axes[0, 0].grid(True, alpha=0.3)

        # Mark emergence events
        for d in self.emerged:
            if d > 0:  # Skip initial void
                axes[0, 0].axhline(d, color="red", linestyle="--", alpha=0.5)

        # Phase density evolution
        for i in range(min(8, self.max_dimensions)):
            phase_mags = np.abs(phase_densities[:, i])
            axes[0, 1].plot(times, phase_mags, label=f"d={i}", linewidth=2)

        axes[0, 1].set_xlabel("Emergent Time")
        axes[0, 1].set_ylabel("Phase Density Magnitude")
        axes[0, 1].set_title("Phase Evolution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Final phase distribution
        final_phases = np.abs(self.phase_density) ** 2
        dims = np.arange(len(final_phases))
        axes[1, 0].bar(dims, final_phases, alpha=0.7, color="purple")
        axes[1, 0].set_xlabel("Dimension")
        axes[1, 0].set_ylabel("Phase Probability")
        axes[1, 0].set_title("Final Dimensional Distribution")
        axes[1, 0].grid(True, alpha=0.3)

        # Dimensional potential landscape
        d_range = np.linspace(0, 10, 1000)
        potential = [self.dimensional_potential(d) for d in d_range]

        axes[1, 1].plot(d_range, potential, "g-", linewidth=2, label="Potential")
        axes[1, 1].axvline(
            self.dimension,
            color="red",
            linestyle="--",
            label=f"Final d={self.dimension:.3f}",
        )
        axes[1, 1].set_xlabel("Dimension")
        axes[1, 1].set_ylabel("Emergence Potential")
        axes[1, 1].set_title("Dimensional Potential Landscape")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def interactive_exploration(self):
        """Interactive exploration of dimensional emergence."""
        print("🎮 Interactive Dimensional Emergence Explorer")
        print("=" * 50)

        while True:
            print("\nOptions:")
            print("1. Run emergence simulation")
            print("2. Analyze specific dimension")
            print("3. Plot potential landscape")
            print("4. Phase coherence analysis")
            print("5. Exit")

            choice = input("\nEnter choice (1-5): ").strip()

            if choice == "1":
                steps = int(input("Evolution steps (default 1000): ") or 1000)
                self.run_emergence_simulation(steps)
                self.plot_emergence_evolution()

            elif choice == "2":
                d = float(input("Dimension to analyze: "))
                potential = self.dimensional_potential(d)
                evolution_rate = self.phase_evolution_rate(d)
                print(f"\nDimension {d}:")
                print(f"  Potential: {potential:.6f}")
                print(f"  Evolution rate: {evolution_rate:.6f}")

            elif choice == "3":
                d_range = np.linspace(0, 10, 1000)
                potential = [self.dimensional_potential(d) for d in d_range]

                plt.figure(figsize=(10, 6))
                plt.plot(d_range, potential, "b-", linewidth=2)
                plt.xlabel("Dimension")
                plt.ylabel("Emergence Potential")
                plt.title("Dimensional Potential Landscape")
                plt.grid(True, alpha=0.3)
                plt.show()

            elif choice == "4":
                print("\nCurrent phase coherence:")
                for i, phase in enumerate(self.phase_density[:8]):
                    coherence = abs(phase)
                    status = "EMERGED" if i in self.emerged else "dormant"
                    print(f"  d={i}: {coherence:.4f} ({status})")

            elif choice == "5":
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice")


# Convenience functions
def run_emergence(steps=1000):
    """Quick emergence simulation."""
    framework = EmergenceFramework()
    return framework.run_emergence_simulation(steps)


def explore_emergence():
    """Interactive emergence exploration."""
    framework = EmergenceFramework()
    framework.interactive_exploration()


def analyze_emergence(d):
    """Analyze emergence properties at dimension d."""
    framework = EmergenceFramework()
    potential = framework.dimensional_potential(d)
    evolution_rate = framework.phase_evolution_rate(d)

    return {
        "dimension": d,
        "potential": potential,
        "evolution_rate": evolution_rate,
        "is_critical": potential
        > np.mean([framework.dimensional_potential(x) for x in range(10)]),
    }


# Module test
def test_emergence_framework():
    """Test the emergence framework module."""
    print("EMERGENCE FRAMEWORK MODULE TEST")
    print("=" * 50)

    framework = EmergenceFramework()

    # Test potential calculation
    print("Dimensional potentials:")
    for d in [0, 1, 2, 3, PHI, E, 5, 7]:
        potential = framework.dimensional_potential(d)
        print(f"  d={d:.3f}: potential={potential:.6f}")

    # Test short simulation
    print("\nRunning short emergence simulation...")
    results = framework.run_emergence_simulation(steps=100)

    print("Results:")
    print(f"  Final dimension: {results['final_dimension']:.3f}")
    print(f"  Emerged dimensions: {results['emerged_dimensions']}")
    print(f"  Dimensional entropy: {results['dimensional_entropy']:.6f}")

    print("\n✅ All emergence framework tests completed!")


if __name__ == "__main__":
    test_emergence_framework()
