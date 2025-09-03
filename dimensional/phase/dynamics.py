"""
Phase Dynamics Engine
======================

Main simulation engine for phase dynamics, extracted from monolithic phase.py.
"""

import numpy as np

from ..mathematics import (
    NumericalInstabilityError,
)
from .analysis import (
    advanced_emergence_detection,
    emergence_threshold,
)
from .core import (
    phase_coherence,
    phase_evolution_step,
    total_phase_energy,
)


class PhaseDynamicsEngine:
    """
    Complete phase dynamics simulation engine with advanced emergence detection.

    Manages the evolution of phase densities across dimensions,
    tracking emergence, clock rates, energy flows, and critical transitions.
    """

    def __init__(self, max_dimensions=12, use_adaptive=True, enable_advanced_detection=True):
        """
        Initialize the phase dynamics engine.

        Parameters
        ----------
        max_dimensions : int
            Maximum number of dimensions to simulate
        use_adaptive : bool
            Whether to use adaptive timestep control
        enable_advanced_detection : bool
            Enable advanced emergence detection features
        """
        self.max_dim = max_dimensions
        self.phase_density = np.zeros(max_dimensions, dtype=complex)
        self.phase_density[0] = 1.0  # Start with unity at the void

        self.flow_matrix = np.zeros((max_dimensions, max_dimensions))
        self.emerged = {0}  # Void always exists
        self.time = 0.0
        self.history = []
        self.use_adaptive = use_adaptive
        self.dt_last = 0.1  # Initial guess for dt

        # Advanced emergence detection system
        self.enable_advanced_detection = enable_advanced_detection
        self.emergence_history = []  # Store emergence analysis results
        self.critical_events = []   # Store critical transition events
        self.phase_state_history = []  # Store phase densities for temporal analysis
        self.max_history_length = 200  # Limit memory usage

        # Control semantics integration
        self.control_state = {
            'additive': {'spatial_extent': 1.0, 'domain_scale': 1.0},
            'multiplicative': {'phase_coupling': 1.0, 'twist_factor': 1.0},
            'boundary': {'edge_phase': 0.0, 'domain_wall': False}
        }
        self.control_history = []  # Track control semantic operations

    def step(self, dt):
        """
        Advance simulation by one time step.

        Parameters
        ----------
        dt : float
            Time step size
        """
        if self.use_adaptive:
            self.step_adaptive(dt)
        else:
            self.phase_density, self.flow_matrix = phase_evolution_step(
                self.phase_density, dt, self.max_dim - 1
            )
            self._update_emergence_and_history(dt)

    def step_adaptive(self, dt_target, max_error=1e-8, max_attempts=50):
        """
        Adaptive stepping with energy conservation.

        Parameters
        ----------
        dt_target : float
            Target time step size
        max_error : float
            Maximum allowed error
        max_attempts : int
            Maximum number of attempts to find good timestep
        """
        dt = min(dt_target, 0.01)  # Start with smaller steps
        best_dt = dt
        best_error = float('inf')

        for attempt in range(max_attempts):
            # Trial step
            trial_phase, trial_flow = phase_evolution_step(
                self.phase_density, dt, self.max_dim - 1
            )

            # Energy conservation check
            initial_energy = total_phase_energy(self.phase_density)
            final_energy = total_phase_energy(trial_phase)
            energy_error = abs(final_energy - initial_energy) / (initial_energy + 1e-12)

            if energy_error < max_error:
                # Accept step
                self.phase_density = trial_phase
                self.flow_matrix = trial_flow
                self.dt_last = dt
                self._update_emergence_and_history(dt)
                return

            if energy_error < best_error:
                best_error = energy_error
                best_dt = dt

            # Adjust dt
            if energy_error > max_error * 10:
                dt *= 0.5  # Reduce step size significantly
            else:
                dt *= 0.9  # Small reduction

            if dt < 1e-10:
                break

        # Use best found if no perfect step
        if best_error < max_error * 100:  # Somewhat relaxed criterion
            trial_phase, trial_flow = phase_evolution_step(
                self.phase_density, best_dt, self.max_dim - 1
            )
            self.phase_density = trial_phase
            self.flow_matrix = trial_flow
            self.dt_last = best_dt
            self._update_emergence_and_history(best_dt)
        else:
            raise NumericalInstabilityError(
                f"Could not find stable timestep after {max_attempts} attempts"
            )

    def _update_emergence_and_history(self, dt):
        """
        Update emergence detection and maintain history.

        Parameters
        ----------
        dt : float
            Time step that was taken
        """
        self.time += dt

        # Check for newly emerged dimensions
        for dim in range(self.max_dim):
            if dim not in self.emerged:
                if emergence_threshold(dim, self.phase_density):
                    self.emerged.add(dim)
                    if self.enable_advanced_detection:
                        self.critical_events.append({
                            'time': self.time,
                            'type': 'emergence',
                            'dimension': dim,
                            'phase_magnitude': abs(self.phase_density[dim])
                        })

        # Advanced emergence detection
        if self.enable_advanced_detection:
            analysis = advanced_emergence_detection(
                self.phase_density,
                self.phase_state_history[-10:] if self.phase_state_history else None
            )
            self.emergence_history.append({
                'time': self.time,
                'analysis': analysis
            })

            # Maintain phase state history
            self.phase_state_history.append(self.phase_density.copy())
            if len(self.phase_state_history) > self.max_history_length:
                self.phase_state_history.pop(0)

        # Store snapshot in history
        self.history.append({
            'time': self.time,
            'phase_density': self.phase_density.copy(),
            'emerged': self.emerged.copy(),
            'total_energy': total_phase_energy(self.phase_density),
            'coherence': phase_coherence(self.phase_density)
        })

    def simulate(self, duration, dt=0.01):
        """
        Run simulation for specified duration.

        Parameters
        ----------
        duration : float
            Total simulation time
        dt : float
            Time step size

        Returns
        -------
        dict
            Simulation results and history
        """
        n_steps = int(duration / dt)

        for _ in range(n_steps):
            self.step(dt)

        return self.get_state()

    def get_state(self):
        """
        Get current engine state.

        Returns
        -------
        dict
            Complete state dictionary
        """
        # Calculate emergence activity
        emergence_activity = len(self.emerged) / self.max_dim if self.max_dim > 0 else 0.0

        return {
            'time': self.time,
            'phase_density': self.phase_density.copy(),
            'phase_densities': self.phase_density.copy(),  # Alias for compatibility
            'emerged': self.emerged.copy(),
            'flow_matrix': self.flow_matrix.copy(),
            'total_energy': total_phase_energy(self.phase_density),
            'coherence': phase_coherence(self.phase_density),
            'emergence_activity': emergence_activity,
            'critical_events_count': len(self.critical_events),
            'history': self.history[-100:] if len(self.history) > 100 else self.history,
            'critical_events': self.critical_events,
            'control_state': self.control_state.copy()
        }

    def reset(self):
        """Reset engine to initial state."""
        self.__init__(self.max_dim, self.use_adaptive, self.enable_advanced_detection)

    def calculate_effective_dimension(self):
        """
        Calculate effective dimension based on energy distribution.

        Returns
        -------
        float
            Effective dimension weighted by energy distribution
        """
        energies = np.abs(self.phase_density) ** 2
        total_energy = np.sum(energies)

        if total_energy == 0:
            return 0.0

        # Weighted average of dimensions
        dimensions = np.arange(len(energies))
        effective_dim = np.sum(dimensions * energies) / total_energy

        return float(effective_dim)

    def apply_control_operation(self, operation_type, parameters):
        """
        Apply control semantic operation to phase dynamics.

        Parameters
        ----------
        operation_type : str
            Type of control operation ('additive', 'multiplicative', 'boundary')
        parameters : dict
            Operation-specific parameters
        """
        if operation_type == 'additive':
            # Spatial extent modification
            if 'spatial_extent' in parameters:
                scale = parameters['spatial_extent']
                self.phase_density *= scale
                self.control_state['additive']['spatial_extent'] *= scale

        elif operation_type == 'multiplicative':
            # Phase coupling modification
            if 'twist' in parameters:
                twist = parameters['twist']
                phases = np.angle(self.phase_density)
                magnitudes = np.abs(self.phase_density)
                phases += twist * np.arange(self.max_dim) / self.max_dim
                self.phase_density = magnitudes * np.exp(1j * phases)
                self.control_state['multiplicative']['twist_factor'] = twist

        elif operation_type == 'boundary':
            # Boundary condition modification
            if 'edge_phase' in parameters:
                edge_phase = parameters['edge_phase']
                self.phase_density[-1] *= np.exp(1j * edge_phase)
                self.control_state['boundary']['edge_phase'] = edge_phase

        # Record operation
        self.control_history.append({
            'time': self.time,
            'operation': operation_type,
            'parameters': parameters.copy()
        })

    def inject(self, dimension, energy):
        """
        Inject energy into a specific dimension.

        Parameters
        ----------
        dimension : int
            Target dimension index
        energy : float
            Amount of energy to inject
        """
        if dimension < len(self.phase_density):
            self.phase_density[dimension] += energy

    def inject_energy(self, amount, target_dimension):
        """
        Inject energy into a specific dimensional level.

        Parameters
        ----------
        amount : float
            Amount of energy to inject
        target_dimension : float
            Target dimension level for energy injection
        """
        # Convert fractional dimension to integer index
        dim_index = int(np.clip(np.round(target_dimension), 0, self.max_dim - 1))

        # Inject energy as phase magnitude increase
        current_phase = self.phase_density[dim_index]
        current_magnitude = np.abs(current_phase)
        current_phase_angle = np.angle(current_phase)

        # Increase magnitude while preserving phase
        new_magnitude = current_magnitude + amount
        self.phase_density[dim_index] = new_magnitude * np.exp(1j * current_phase_angle)

    def evolve(self, n_steps, dt=0.01):
        """
        Evolve the system for n_steps with given time step.

        Parameters
        ----------
        n_steps : int
            Number of evolution steps to take
        dt : float, optional
            Time step size (default: 0.01)

        Returns
        -------
        dict
            Evolution results with current_emerged and final state
        """
        for _ in range(n_steps):
            self.step(dt)

        return {
            'current_emerged': self.emerged.copy(),
            'final_state': self.get_state()
        }
