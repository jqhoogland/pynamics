"""

Author: Jesse Hoogland
Year: 2020

"""
from typing import Union

import numpy as np
from scipy.integrate import OdeSolver, DenseOutput
from scipy.integrate._ivp.common import (validate_max_step, validate_tol,
                                         select_initial_step, norm,
                                         warn_extraneous, validate_first_step)

Position = Union[np.ndarray, float]

def em_step(f, g, t, y, timestep):
    return y + f(t, y) * timestep + g(
        t, y) * np.random.normal(scale=np.sqrt(timestep))


class EmDenseOutput(DenseOutput):
    """
    It interpolates between ``t_min`` and ``t_max``. Evaluation outside this interval is not forbidden, but the accuracy is not guaranteed.

    Currently, this is just a linear interpolation.
    TODO: Something fancier (at least allow for multiple components)

    """
    def __init__(self, t_old, t, y_old, y):
        super(EmDenseOutput, self).__init__(t_old, t)
        self.h = t - t_old
        self.y_old = y_old
        self.y = y
        self.m = y - y_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h

        y = self.m * x

        if y.ndim == 2:
            y += self.y_old[:, None]
        else:
            y += self.y_old

        return y


class EulerMaruyama(OdeSolver):
    """

    Integrates a function of the kind
    $$ dX_t = f(t, X) dt + g(t, X) dW_t, $$
    where dW_t is Weiner noise with standard deviation equal to dt

    """
    def __init__(self,
                 f,
                 g,
                 t0,
                 y0,
                 t_bound,
                 timestep=0.001,
                 vectorized=False,
                 **extraneous):
        warn_extraneous(extraneous)
        super(EulerMaruyama, self).__init__(f,
                                            t0,
                                            y0,
                                            t_bound,
                                            vectorized,
                                            support_complex=True)
        self.g = g
        self.y_old = None
        self.timestep = timestep

    def _step_impl(self):
        """
        Propagates the integrator one step further

        :returns (success, message): (Bool True or False, None or String, correspondingly )
        """
        t = self.t
        y = self.y

        self.y_old = self.y
        self.y = em_step(self.fun, self.g, t, y, self.timestep)
        self.t = t + self.timestep

        return True, None

    def _dense_output_impl(self):
        """
        Returns a DenseOutput object covering the last successful step.
        """
        return EmDenseOutput(self.t_old, self.t, self.y_old, self.y)
