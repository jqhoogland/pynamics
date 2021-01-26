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
import numpy as np
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    def __repr__(self):
        return f"<Lorenz sigma={self.sigma} rho={self.rho} beta={self.beta} timestep={self.timestep}>"
