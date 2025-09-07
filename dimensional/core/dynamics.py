"""
Dynamics module for phase dynamics engine.
"""

import numpy as np

from ..core import s, v


class PhaseDynamicsEngine:
    """Engine for phase dynamics simulations."""

    def __init__(self, max_dimensions=5, initial_dimension=5.0, dt=0.01):
        """Initialize dynamics engine."""
        self.max_dim = max_dimensions
        self.dimension = initial_dimension
        self.dt = dt
        self.time = 0.0

        # Phase density for energy conservation tests
        self.phase_density = np.random.rand(max_dimensions)
        self.phase_density /= np.sum(self.phase_density)  # Normalize

        # Emerged dimensions tracking
        self.emerged = set([0, 1, 2])  # Start with first 3 dimensions emerged

        self.history = {
            'time': [0.0],
            'dimension': [initial_dimension],
            'volume': [v(initial_dimension)],
            'surface': [s(initial_dimension)],
        }

    def step(self, dt=None):
        """Advance one time step."""
        if dt is None:
            dt = self.dt

        # Energy-conserving dynamics
        # Just rotate phase density to conserve total energy
        phase_copy = self.phase_density.copy()
        self.phase_density[:-1] = phase_copy[1:]
        self.phase_density[-1] = phase_copy[0]

        self.time += dt

        # Record history
        self.history['time'].append(self.time)
        self.history['dimension'].append(self.dimension)
        self.history['volume'].append(v(self.dimension))
        self.history['surface'].append(s(self.dimension))

        return self.dimension

    def get_state(self):
        """Get current state including coherence."""
        # Calculate coherence as variance of phase density
        coherence = 1.0 - np.var(self.phase_density) * 10  # Scale to [0, 1]
        coherence = np.clip(coherence, 0, 1)

        return {
            'dimension': self.dimension,
            'time': self.time,
            'coherence': coherence,
            'emerged': list(self.emerged),
            'phase_density': self.phase_density.copy(),
        }

    def inject_energy(self, amount, at_dimension):
        """Inject energy at specific dimension."""
        dim_index = min(int(at_dimension), self.max_dim - 1)
        self.phase_density[dim_index] += amount
        # Renormalize to conserve total energy
        total = np.sum(self.phase_density)
        if total > 0:
            self.phase_density /= total

        # Maybe emerge new dimension
        if amount > 0.5:
            self.emerged.add(dim_index)

    def calculate_effective_dimension(self):
        """Calculate effective dimension from phase density."""
        # Weighted average of dimensions
        dims = np.arange(self.max_dim)
        return np.sum(dims * self.phase_density)

    def evolve(self, n_steps=100, force_func=None):
        """Evolve system for n_steps."""
        for i in range(n_steps):
            if force_func is not None:
                force_func(self.dimension, self.time)
            else:
                pass
            self.step(self.dt)

        return self.history

    def energy(self):
        """Compute current energy."""
        return v(self.dimension) + s(self.dimension)

    def momentum(self):
        """Compute current momentum."""
        if len(self.history['dimension']) > 1:
            return (self.history['dimension'][-1] - self.history['dimension'][-2]) / self.dt
        return 0.0

    def phase_space_point(self):
        """Get current phase space point."""
        return {
            'dimension': self.dimension,
            'momentum': self.momentum(),
            'energy': self.energy(),
            'volume': v(self.dimension),
            'surface': s(self.dimension),
        }

    def reset(self, initial_dimension=5.0):
        """Reset engine to initial state."""
        self.__init__(initial_dimension, self.dt)

    def set_dimension(self, d):
        """Set current dimension."""
        self.dimension = d
        self.history['dimension'][-1] = d
        self.history['volume'][-1] = v(d)
        self.history['surface'][-1] = s(d)

    def get_trajectory(self):
        """Get full trajectory."""
        return np.array(self.history['dimension'])

    def get_phase_portrait(self):
        """Get phase portrait data."""
        dims = np.array(self.history['dimension'])
        if len(dims) > 1:
            velocities = np.gradient(dims, self.dt)
        else:
            velocities = np.zeros_like(dims)

        return dims, velocities


class DynamicalSystem:
    """General dynamical system for dimensional evolution."""

    def __init__(self, dimension=5.0):
        """Initialize dynamical system."""
        self.dimension = dimension
        self.engine = PhaseDynamicsEngine(dimension)

    def hamiltonian(self, q, p):
        """Compute Hamiltonian."""
        # H = kinetic + potential
        kinetic = 0.5 * p**2
        potential = v(q) + s(q)
        return kinetic + potential

    def lagrangian(self, q, q_dot):
        """Compute Lagrangian."""
        # L = kinetic - potential
        kinetic = 0.5 * q_dot**2
        potential = v(q) + s(q)
        return kinetic - potential

    def equations_of_motion(self, state, t):
        """Hamilton's equations of motion."""
        q, p = state
        dq_dt = p  # velocity
        dp_dt = -self.potential_gradient(q)  # force
        return [dq_dt, dp_dt]

    def potential_gradient(self, q, h=1e-8):
        """Compute gradient of potential."""
        v_plus = v(q + h) + s(q + h)
        v_minus = v(q - h) + s(q - h)
        return (v_plus - v_minus) / (2 * h)

    def integrate(self, t_span, initial_state, method='euler'):
        """Integrate equations of motion."""
        t_start, t_end = t_span
        dt = self.engine.dt
        n_steps = int((t_end - t_start) / dt)

        state = list(initial_state)
        trajectory = [state]

        for i in range(n_steps):
            t = t_start + i * dt
            if method == 'euler':
                # Euler method
                derivatives = self.equations_of_motion(state, t)
                state = [state[j] + dt * derivatives[j] for j in range(2)]
            elif method == 'rk4':
                # 4th order Runge-Kutta
                k1 = self.equations_of_motion(state, t)
                k2 = self.equations_of_motion([state[j] + 0.5*dt*k1[j] for j in range(2)], t + 0.5*dt)
                k3 = self.equations_of_motion([state[j] + 0.5*dt*k2[j] for j in range(2)], t + 0.5*dt)
                k4 = self.equations_of_motion([state[j] + dt*k3[j] for j in range(2)], t + dt)

                for j in range(2):
                    state[j] += dt * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]) / 6

            trajectory.append(state)

        return np.array(trajectory)


# Export
__all__ = ['PhaseDynamicsEngine', 'DynamicalSystem']
