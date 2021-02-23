"""

Author: Jesse Hoogland
Year: 2020

Parameter values:
- 0 < r < 1: Origin stable
- 1 < r 1.346: Two stable points
- 1.346 < r < 13.926: Two stable spirals
- 13.926 < r < 24.06: Transient chaos
- 24.06 < r < 24.74: Coexistence of fixed point and chaos
- 24.74 < r < 28: Chaos
- 28 < r < 313: Periodic windows


"""
import matplotlib
import numpy as np

matplotlib.rcParams['text.usetex'] = True

from ..trajectories import DeterministicTrajectory


class Lorenz(DeterministicTrajectory):
    def __init__(self, sigma=10., rho=28, beta=8. / 3, **kwargs):
        super(Lorenz, self).__init__(n_dofs=3, **kwargs)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def take_step(self, t, pos):
        x, y, z = pos
        return np.array([self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z])

    def jacobian(self, state):
        x, y, z = state
        return np.array([[-self.sigma, self.sigma, 0],
                         [(self.rho - z), -1., -x],
                         [y, x, -self.beta]])

    def __repr__(self):
        return f"<Lorenz sigma={self.sigma} rho={self.rho} beta={self.beta} timestep={self.timestep}>"
