"""

Author: Jesse Hoogland
Year: 2020

"""
from typing import Union, Any, Callable

from scipy.integrate import quad, dblquad
from nptyping import NDArray
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..trajectories import StochasticTrajectory, TimeSeries, Position
from ..integrate import Position


class BrownianMotion(StochasticTrajectory):
    """
    (1d) Brownian motion.

    The equations of motion are given by: $$m\ddot x = -\gamma \dot x
    - V'(x) + \sqrt{2\gamma \beta^{-1}} dW_t,$$ where

    :param x: is the position of the particle
    :param beta: The thermodynamic beta $\beta^{-1} = k_B T$
    :param gamma: is the damping constant
    :param is_overdamped: is a boolean tag equivalent to taking the
        limit $m \to 0$.

        - V is the potential (V' denotes its first spatial
          derivative).  By default this is 0, though this is
          overridden in child classes.

        - dW_t is a white noise Wiener process with standard deviation
          dt.

    At a quick glance, one sees that this is overparametrized.  We
    subsume $m$ into a redefinition of the other parameters.  We set
    $m:=1$.  We can't change this value but we can access it.

    In the over-damped limit, $m\ddot x \ll 1 $, so the E.O.M reduces
    to: $$\gamma \dot x = - V'(x) + \sqrt{2\gamma k_B T} dW_t,$$
    """

    def __init__(self,
                 beta: float = 1.,
                 gamma: float = 1.,
                 is_overdamped: bool = False,
                 **kwargs):
        self.beta = beta
        self.gamma = gamma
        self.is_overdamped = is_overdamped

        vectorized = not is_overdamped
        n_dofs = 1 if is_overdamped else 2
        super().__init__(vectorized=vectorized, n_dofs=n_dofs, **kwargs)

        # We run the conditional here so we don't have to go through it every time we integrate a step forward
        if self.is_overdamped:
            self.m = 0.
            self._take_step = lambda t, x: - self._grad_potential(x) / self.gamma
            self._get_random_step = lambda t, x:  np.sqrt(2. * self.gamma / self.beta) / self.gamma
        else:
            self.m = 1.
            self._take_step = lambda t, x: np.array([x[1], -self.gamma * x[1] - self._grad_potential(x[0])])
            self._get_random_step = lambda t, x: np.array([0, np.sqrt(2. * self.gamma / self.beta)])


    def __repr__(self):
        return f"<BrownianMotion beta:{self.beta} gamma:{self.gamma} is_overdamped:{self.is_overdamped} timestep={self.timestep}>"

    # Helper methods for computing various macroscopic observables
    # By default we assume a free particle

    @staticmethod
    def _laplacian_potential(x: float) -> float:
        return 0.

    @staticmethod
    def _grad_potential(x: Union[NDArray[Any], float]) -> Union[NDArray[Any, float], float]:
        return 0.

    @staticmethod
    def _potential_energy(x: Union[NDArray[Any], float]) -> Union[NDArray[Any, float], float]:
        return 0.

    def _kinetic_energy(self, v: Union[NDArray[Any], float]) -> Union[NDArray[Any, float], float]:
        return (self.m * v ** 2) / 2.

    def _energy(self, x: float, v: float=0) -> float:
        if self.is_overdamped:
            return self._potential_energy(x)

        return self._potential_energy(x) + self._kinetic_energy(v)

    def _boltzmann_weight(self, x:float, v: float=0) -> float:
        return np.exp(-self._energy(x, v) * self.beta)

    # Wrappers for computing macroscopic observables of either
    # positions or whole timeseries

    def grad_potential(self, state: Union[TimeSeries, Position]) -> Union[NDArray[Any, float], float]:
        if isinstance(state, TimeSeries):
            # i.e. it's a time_series
            return self._grad_potential(state[:, 0])

        return self._grad_potential(state[0])

    def potential_energy(self, state: Union[TimeSeries, Position]) -> Union[NDArray[Any, float], float]:
        if isinstance(state, TimeSeries):
            # i.e. it's a time_series
            return self._potential_energy(state[:, 0])

        return self._potential_energy(state[0])

    def kinetic_energy(self, state: Union[TimeSeries, Position]) -> Union[NDArray[Any, float], float]:
        if isinstance(state, TimeSeries):
            # i.e. it's a time_series

            # we make sure we're in the not overdamped case
            if state.shape[1] > 1:
                return self._kinetic_energy(state[:, 1])

            # otherwise there is no notion of instantaneous velocity and kinetic energy
            return 0.

        return self._kinetic_energy(state[1])

    def energy(self, state: Union[TimeSeries, Position]) -> Union[NDArray[Any, float], float]:
        return self.potential_energy(state) + self.kinetic_energy(state)

    def boltzmann_weights(self, state: Union[TimeSeries, Position]) -> Union[NDArray[Any, float], float]:
        energies = self.energy(state)
        return np.exp(-energies * (self.beta))

    def boltzmann_probs(self, state: Union[TimeSeries, Position], dX: float=1) -> Union[NDArray[Any, float], float]:
        # TODO: Do something with dX
        weights = self.boltzmann_weights(state)
        return weights / np.sum(weights)

    def take_step(self, t: float, x: Position) -> Position:
        return self._take_step(t, x)

    def get_random_step(self, t: float, x: Position) -> Position:
        return self._get_random_step(t, x)

    @staticmethod
    def running_avg(time_series: TimeSeries, fn: Callable[[TimeSeries], NDArray[Any, float]], window_size: int=1000, step_size: int=100, verbose: bool=False) -> NDArray[Any, float]:
        n_windows = int(np.floor((time_series.shape[0] - window_size + 1) / step_size))
        avgs = np.zeros(n_windows)

        range_ = range(n_windows)

        if verbose:
            range_ = tqdm(range_, desc="Calculating running average")

        for i in range_:
            avgs[i] = np.mean(fn(time_series[i * step_size: i * step_size+ window_size]))

        return avgs

    @property
    def partition_fn(self):
        assert self.is_overdamped, "Must be overdamped to compute partition function"

        return quad(self._boltzmann_weight, -np.inf, np.inf)[0] # Returns (result, error)

    @property
    def avg_energy(self):
        assert self.is_overdamped, "Must be overdamped to compute average energy"

        if self.is_overdamped:
            energy_weighted = lambda x: self._energy(x) * self._boltzmann_weight(x) / self.partition_fn
            return quad(energy_weighted, -np.inf, np.inf)[0] # Returns (result, error)

        else:
            energy_weighted = lambda x, v: self._energy(x, v)* self._boltzmann_weight(x, v) / self.partition_fn
            return dblquad(energy_weighted)

    def first_passage_time(self, x: float, left_endpoint: float= -np.inf, right_endpoint: float=0):
        inner_fn = lambda y: quad(self._boltzmann_weight, left_endpoint, y)[0]

        # Note the next doesn't have the minus sign
        outer_fn = lambda y: np.exp(self._potential_energy(y) * self.beta) * inner_fn(y)
        outer = quad(outer_fn, x, right_endpoint)[0]
        return self.gamma * self.beta * outer

    def avg_first_passage_time(self, left_endpoint: float= -np.inf, right_endpoint: float=0):
        total_weight = quad(self._boltzmann_weight, left_endpoint, right_endpoint)[0]

        inner_fn = lambda y: quad(self._boltzmann_weight, left_endpoint, y)[0]
        # Note the next doesn't have the minus sign
        middle_fn = lambda y, x: np.exp((self._potential_energy(y) - self._potential_energy(x)) * self.beta) * inner_fn(y)
        outer_fn = lambda x: quad(middle_fn, x, right_endpoint, args=(x, ))[0]
        outer = quad(outer_fn, left_endpoint, right_endpoint)[0]

        return self.gamma * self.beta * outer / total_weight

    def count_passages(self, trajectory: TimeSeries, left_endpoint: float= -np.inf, right_endpoint: float=0):
        # to_the_left = np.where(trajectory > left_endpoint, 1, 0)
        # to_the_right = np.where( trajectory < right_endpoint, 1, 0)
        # binary_series = np.where(to_the_left + to_the_right == 2, 1, 0) # 1 is inside the region; 0 outside
        # transitions = np.where(binary_series[:-1] != binary_series[1:], 1, 0) # we care about transitions 1 -> 0
        # return np.sum(transitions)

        n_passages = 0

        in_range = lambda x: x > left_endpoint and x < right_endpoint

        for i in range(trajectory.shape[0] - 1):
            if in_range(trajectory[i]) and not in_range(trajectory[i + 1]):
                n_passages += 1

        return n_passages

    def measure_first_passage_time(self, trajectory: TimeSeries, left_endpoint: float= -np.inf, right_endpoint: float=0):
        return trajectory.shape[0] * self.timestep / self.count_passages(trajectory, left_endpoint, right_endpoint)


class DoubleWell(BrownianMotion):
    """
    :param state: refers to a full state in phase-space.  This is a pair
              [x, v] for the not overdamped case.
    :param x: is the spatial component of the state.  This is the full
              state in the overdamped case.
    :param v: is the velocity component of the state.
    """

    # Helper methods for computing various macroscopic observables

    @staticmethod
    def _laplacian_potential(x: float) -> float:
        return 4 * ( 3 * x ** 2 - 1)

    @staticmethod
    def _grad_potential(x: Union[NDArray[Any], float]) -> float:
        return 4. * x * (x ** 2 - 1)

    @staticmethod
    def _potential_energy(x: Union[NDArray[Any], float]) -> float:
        return (x ** 2 - 1.) ** 2

    @property
    def barrier_freq(self) -> float:
        return np.sqrt(np.abs(self._laplacian_potential(0)))

    @property
    def well_freq(self) -> float:
        return np.sqrt(self._laplacian_potential(-1))

    @property
    def thermal_lengthscale(self) -> float:
        return np.sqrt(2. / (self.beta)) / self.barrier_freq

    @property
    def barrier_energy(self) -> float:
        return self._potential_energy(0) - self._potential_energy(-1)

    @property
    def energyscale(self) -> float:
        return self.barrier_energy * self.beta

    @property
    def barrier_height(self):
        return 2

    @property
    def rel_barrier_height(self):
        return 2 * self.beta

    def get_timescale(self, eigval: float, multiplier: float=1.) -> float:
        return -float(self.timestep * multiplier/ np.log(eigval))

    @staticmethod
    def count_bw_wells(time_series: TimeSeries):
        n_timesteps= time_series.shape[0]
        times = 0
        n_passes = 0

        def get_region(pos):
            if pos[0] <= -1:
                return -1
            elif pos[0] > 1:
                return 1
            else:
                return 0

        time_passed = 0
        prev_region = None
        prev_well = None

        for i in range(n_timesteps):
            pos = time_series[i]
            curr_region = get_region(pos)

            if curr_region == 1 and prev_well == -1:
                # If we pass the right well after coming from the left
                #   \     __      /
                #    \___/  \_->_/
                # We stop the clock
                times += time_passed
                n_passes += 1
                prev_well = 1

            elif curr_region == -1 and prev_well == 1:
                # If we pass the left well after coming from the right
                #   \      __     /
                #    \_<-_/  \___/
                # We stop the clock
                times += time_passed
                n_passes += 1
                prev_well = -1

            elif curr_region == 0:
                if prev_region == -1:
                    # If we pass the left well headed for the right
                    #   \      __     /
                    #    \_->_/  \___/
                    # We start the clock
                    time_passed = 1
                elif prev_region == 1:
                    # If we pass the right wall headed for the left
                    #   \     __      /
                    #    \___/  \_<-_/
                    # We start the clock
                    time_passed = 1
                else:
                    # We are loitering in the interim
                    #   \      __      /
                    #    \_|>_/  \_<|_/
                    time_passed += 1

            elif curr_region != 0:
                # We have not yet initialized a previous well
                prev_well = curr_region

            prev_region = curr_region

        return n_passes * n_timesteps / times

    def transition_time(self, trajectory: TimeSeries, min_staying_timesteps: int=1) -> float:
        """
        The amount of time it takes for a particle to pass from one well to the other,
        excluding the waiting time.
        AKA Transition-event duration or translocation time

        This is the interval bw the last time the particle passes the start point
        and the first time it reaches the endpoint. Where the start/endpoints are either left/right wells.
        """
        trajectory_duration = trajectory.shape[0] * self.timestep
        n_crossovers = self.count_crossovers(trajectory, 0., min_staying_timesteps)

        time_1 = trajectory_duration / n_crossovers
        time_2 = trajectory_duration / self.count_bw_wells(trajectory)
        return time_1, time_2

    def dwell_time(self, trajectory: TimeSeries) -> float:
        """
        The average time spent in either well. AKA Waiting time, first passage time.

        Given a trajectory that is T long and N crossings over the origin,
        the dwell time is T/N
        """
        return self.measure_first_passage_time(trajectory, -np.inf, 0)

def test_count_passages_1():
    dw = DoubleWell(is_overdamped=True, timestep=0.001)
    a = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2])
    b = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2, -1, -2])
    c = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2, -1, -2, 5, 3])
    d = np.array([-1, -1, 2, -3, 2, -0.5, 1, 2, 1, 5, 2, -1, -2, 5, 3])
    assert dw.count_passages(a) == 1
    assert dw.count_passages(b) == 1
    assert dw.count_passages(c) == 2
    assert dw.count_passages(d) == 4


def test_count_passages_2():
    dw = DoubleWell(is_overdamped=True, timestep=0.001)
    a = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2])
    b = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2, -1, -2])
    c = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2, -1, -2, 5, 3])
    d = np.array([-1, -1, 2, -3, 2, -0.5, 1, 2, 1, 5, 2, -5, -2, 5, 3])
    assert dw.count_passages(a, -np.inf, -2.5) == 1
    assert dw.count_passages(b, -np.inf, -2.5) == 1
    assert dw.count_passages(c, -np.inf, -2.5) == 1
    assert dw.count_passages(d, -np.inf, -2.5) == 2


def test_count_passages_3():
    dw = DoubleWell(is_overdamped=True, timestep=0.001)
    a = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2])
    b = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2, -1, -2])
    c = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2, -1, -2, 5, 3])
    d = np.array([-1.1, -1.1, 2, -3, 2, -0.5, 1, 2, 1, 5, 2, -5, -2, 5, 3])
    assert dw.count_passages(a, -np.inf, -1) == 1
    assert dw.count_passages(b, -np.inf, -1) == 1
    assert dw.count_passages(c, -np.inf, -1) == 2
    assert dw.count_passages(d, -np.inf, -1) == 3

def test_count_passages_4():
    dw = DoubleWell(is_overdamped=True, timestep=0.001)
    a = np.array([-1, -1, -2, -3, -2, -0.5, 1, 1.9, 1, 5, 2])
    b = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2.1, 1, 5, 2.1, -1, -2])
    c = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2.1, 1, 5, 2, -1, -2, 5, 3])
    d = np.array([-1, -1, 2.1, -3, 3, -0.5, 1, 1.9, 1, 5, 2, -5, -2, 5, 3])
    assert dw.count_passages(a, -np.inf, 2) == 1
    assert dw.count_passages(b, -np.inf, 2) == 2
    assert dw.count_passages(c, -np.inf, 2) == 3
    assert dw.count_passages(d, -np.inf, 2) == 4

def test_measure_passages_1():
    timestep = 0.001
    dw = DoubleWell(is_overdamped=True, timestep=timestep)
    a = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2])
    b = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2, -1, -2])
    c = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2, -1, -2, 5, 3])
    d = np.array([-1, -1, 2, -3, 2, -0.5, 1, 2, 1, 5, 2, -1, -2, 5, 3])
    assert dw.measure_first_passage_time(a) == 11 * timestep / 1
    assert dw.measure_first_passage_time(b) == 13 * timestep / 1
    assert dw.measure_first_passage_time(c) == 15 * timestep / 2
    assert dw.measure_first_passage_time(d) == 15 * timestep / 4

def test_measure_passages_2():
    timestep = 0.001
    dw = DoubleWell(is_overdamped=True, timestep=timestep)
    a = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2])
    b = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2, -1, -2])
    c = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2, -1, -2, 5, 3])
    d = np.array([-1, -1, 2, -3, 2, -0.5, 1, 2, 1, 5, 2, -5, -2, 5, 3])
    assert dw.measure_first_passage_time(a, -np.inf, -2.5) == 11 * timestep / 1
    assert dw.measure_first_passage_time(b, -np.inf, -2.5) == 13 * timestep / 1
    assert dw.measure_first_passage_time(c, -np.inf, -2.5) == 15 * timestep / 1
    assert dw.measure_first_passage_time(d, -np.inf, -2.5) == 15 * timestep / 2


def test_measure_passages_3():
    timestep = 0.001
    dw = DoubleWell(is_overdamped=True, timestep=timestep)
    a = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2])
    b = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2, -1, -2])
    c = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2, 1, 5, 2, -1, -2, 5, 3])
    d = np.array([-1.1, -1.1, 2, -3, 2, -0.5, 1, 2, 1, 5, 2, -5, -2, 5, 3])
    assert dw.measure_first_passage_time(a, -np.inf, -1) == 11 * timestep / 1
    assert dw.measure_first_passage_time(b, -np.inf, -1) == 13 * timestep / 1
    assert dw.measure_first_passage_time(c, -np.inf, -1) == 15 * timestep / 2
    assert dw.measure_first_passage_time(d, -np.inf, -1) == 15 * timestep / 3

def test_measure_passages_4():
    timestep = 0.001
    dw = DoubleWell(is_overdamped=True, timestep=timestep)
    a = np.array([-1, -1, -2, -3, -2, -0.5, 1, 1.9, 1, 5, 2])
    b = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2.1, 1, 5, 2.1, -1, -2])
    c = np.array([-1, -1, -2, -3, -2, -0.5, 1, 2.1, 1, 5, 2, -1, -2, 5, 3])
    d = np.array([-1, -1, 2.1, -3, 3, -0.5, 1, 1.9, 1, 5, 2, -5, -2, 5, 3])
    assert dw.measure_first_passage_time(a, -np.inf, 2) == 11 * timestep / 1
    assert dw.measure_first_passage_time(b, -np.inf, 2) == 13 * timestep / 2
    assert dw.measure_first_passage_time(c, -np.inf, 2) == 15 * timestep / 3
    assert dw.measure_first_passage_time(d, -np.inf, 2) == 15 * timestep / 4
