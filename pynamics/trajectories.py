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
        :param vectorized: This is by ODESolver in some way that I
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
        return "trajectory-dof{}".format(self.n_dofs)

    def get_integrator(self, init_dofs, n_steps):
        raise NotImplementedError

    def take_step(self, t: float, state: Position) -> Position:
        raise NotImplementedError

    def jacobian(self, state: Position) -> Position:
        raise NotImplementedError

    @np_cache(dir_path="./saves/lyapunov/",
              file_prefix="spectrum-",
              ignore=[0])
    def get_lyapunov_spectrum(self,
                              trajectory: np.ndarray,
                              n_burn_in: int = 0,
                              n_exponents: Optional[int] = None,
                              t_ons: int = 10) -> np.ndarray:
        """
        See `./lyapunov` > `get_lyapunov_spectrum`
        """
        return get_lyapunov_spectrum(
            self.jacobian, trajectory, n_burn_in=n_burn_in, n_exponents=n_exponents, t_ons=t_ons, n_dofs=self.n_dofs
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

    def jacobian(self, state: Position) -> Position:
        raise NotImplementedError

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def downsample(rate: int = 10):
    """
    A function decorator that downsamples the result returned by that function.
    Assumes the function returns a np.ndarray of the shape (n_samples, n_dofs).

    :returns: a function which returns a downsampled array of shape (n_samples // rate, n_dofs)
    """
    def inner(func):
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            if rate == 0:
                return samples
            return samples[::rate]

        return wrapper

    return inner


def downsample_split(rate: int = 10):
    """
    A function decorator that downsamples the result returned by that function.
    Assumes the function returns a np.ndarray of the shape (n_samples, n_dofs).

    Instead of throwing away the intermediate entries, this creates a separate
    downsampled chain for each possible starting point.

    :returns: a function which returns a downsampled array of shape (rate, n_samples // rate, n_dofs)
    """
    def inner(func):
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)

            if rate == 0:
                return np.array([samples])

            n_downsamples = len(samples) // rate
            downsamples = np.zeros((rate, n_downsamples, samples.shape[1]))

            for i in range(rate):
                downsamples[i, :, :] = samples[i::rate, :]

            return downsamples

        return wrapper

    return inner


def avg_over(key: str, axis: int = 0, include_std: bool = False):
    """
    A function decorator which averages a function over one of its given parameters.

    To be used in conjunction with the above.

    :param axis: the axis of the result to average over, defaults to 0, the first axis.
    :param kwargs: should contain one keyword argument

    e.g. A function is given an array [n_samples, n_timesteps, n_dofs].
    This decorator performs that function n_samples times for the values [i, :, :] where i in n_samples.
    """
    def inner(func):
        def wrapper(*args, **kwargs):
            value = kwargs.pop(key)

            # Make sure kwargs contains a suitable value for this key
            if value is None:
                raise ValueError(f"key {key} not in kwargs.")
            elif not isinstance(value, np.ndarray):
                raise TypeError(f"key {key} not a numpy array.")

            if axis != 0:
                np.swapaxes(value, 0, axis)

            responses = []

            for i in range(value.shape[0]):
                avg_kwarg = {}
                avg_kwarg[key] = value[i, :, :]
                responses.append(func(*args, **avg_kwarg, **kwargs))

            responses = np.array(responses)

            if include_std:
                return np.mean(responses, axis=0), np.std(responses, axis=axis)

            return np.mean(responses, axis=0)

        return wrapper

    return inner
