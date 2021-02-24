"""

Contains baseline wrappers for generating phase-space trajectories,
most importantly a wrapper for performing stochastic integration using the
Euler Maruyama scheme.

Author: Jesse Hoogland
Year: 2020

"""
from typing import Optional, Any

import numpy as np
from nptyping import NDArray
from scipy.integrate import RK45
from tqdm import tqdm

from .integrate import EulerMaruyama
from .lyapunov import get_lyapunov_spectrum
from .utils import np_cache

# Meant to be a series of positions in some phase space.
# The first dimension is over time; the second over phase space.
TimeSeries = NDArray[(Any, Any), float]

# A position in phase space.
# I should probably call this State or Configuration instead.
Position = NDArray[(Any), float]


class Trajectory:
    """

    A wrapper for an ODESolver to integrate formulas specified in children.

    """
    def __init__(self,
                 timestep: float =1e-3,
                 init_state: Optional[Position] = None,
                 n_dofs: Optional[int] = 100,
                 vectorized: bool=True):
        """
        :param timestep: The timestep to take curing an evolution.
            This is 1 for discrete trajectories, otherwise a value
            likely much smaller than 1.
        :param init_state: The state to initialize the neurons with.
            defaults to a state of `n_dofs` neurons drawn randomly
            from the uniform distribution.  if int, then this
            multiplies the above, if left blank, then `n_dofs` must be
            specified.
        :param n_dofs: The number of dofs.  if `init_state` is of type
            `int`, this must be specified, else `n_dofs` is
            overwritten by the size of `init_state`
        :param vectorized: This is required by ODESolver in some way that I
            have yet to figure out.  TODO: Is this even necessary?
        """
        self.timestep = timestep

        if init_state is None:
            assert not n_dofs is None
            init_state = np.random.uniform(size=n_dofs)

        elif isinstance(init_state, float):
            assert not n_dofs is None
            init_state = np.random.uniform(size=n_dofs) * init_state

        else:
            n_dofs = init_state.size

        self.init_state = init_state
        self.n_dofs = n_dofs
        self.vectorized = vectorized

    def __str__(self):
        return f"<Trajectory dof={self.n_dofs}>"

    def get_integrator(self, init_dofs, n_steps):
        raise NotImplementedError

    def take_step(self, t: float, state: Position) -> Position:
        raise NotImplementedError

    def jacobian(self, state: Position) -> Position:
        raise NotImplementedError

    @np_cache(dir_path="./saves/lyapunov/", file_prefix="spectrum-", ignore=[0])
    def get_lyapunov_spectrum(self,
                              trajectory: np.ndarray,
                              n_burn_in: int = 0,
                              n_exponents: Optional[int] = None,
                              t_ons: int = 10,
                              **kwargs) -> np.ndarray:
        """
        See `.lyapunov`
        """
        return get_lyapunov_spectrum(
            self.jacobian,
            trajectory,
            n_burn_in=n_burn_in,
            n_exponents=n_exponents,
            t_ons=t_ons,
            n_dofs=self.n_dofs,
            timestep=self.timestep,
            **kwargs
        )

    @np_cache(dir_path="./saves/trajectories/", file_prefix="trajectory-")
    def run(self, n_burn_in: int = 500, n_steps: int = 10000):

        integrator = self.get_integrator(self.init_state, n_steps)
        state = np.zeros([n_steps, self.n_dofs])

        for _ in tqdm(range(n_burn_in), desc="Burning in"):
            integrator.step()

        for t in tqdm(range(n_steps), desc="Generating samples: "):
            state[t, :] = np.array(integrator.y)
            integrator.step()

        return state


class DeterministicTrajectory(Trajectory):
    def get_integrator(self, init_dofs, n_steps):
        return RK45(self.take_step,
                    0,
                    init_dofs,
                    n_steps,
                    max_step=self.timestep,
                    vectorized=self.vectorized)

    def take_step(self, t: float, state: Position) -> Position:
        raise NotImplementedError

    def jacobian(self, state: Position) -> Position:
        raise NotImplementedError


class StochasticTrajectory(Trajectory):
    def get_integrator(self, init_dofs, n_steps):
        return EulerMaruyama(self.take_step,
                             self.get_random_step,
                             0,
                             init_dofs,
                             n_steps,
                             timestep=self.timestep,
                             vectorized=self.vectorized)

    def take_step(self, t: float, state: Position) -> Position:
        raise NotImplementedError

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError
