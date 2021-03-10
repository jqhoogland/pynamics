"""

Contains code to determine the Lyapunov spectrum via the reorthonormalization procedure

Author: Jesse Hoogland
Year: 2020

See [@wolf1985]


"""
from typing import Optional, Callable

import numpy as np
from tqdm import tqdm

from .utils import random_orthonormal, qr_positive


def qr_evolve(trajectory, jacobian, q, desc, w_ons, return_spectrum: bool = False):
    lyapunov_spectrum = np.zeros(q.shape[1])

    for t, state in enumerate(tqdm(trajectory,
                                   desc=desc)):
        q = jacobian(state) @ q

        if t % w_ons == 0:
            q, r = qr_positive(q)

            r_diagonal = np.copy(np.diag(r))
            r_diagonal[r_diagonal == 0] = 1

            lyapunov_spectrum += np.log(r_diagonal)

    if return_spectrum:
        return q, lyapunov_spectrum

    return q


def get_lyapunov_spectrum(
        jacobian: Callable[[np.ndarray], np.ndarray],
        trajectory: np.ndarray,
        n_burn_in: int = 0,
        n_exponents: Optional[int] = None,
        t_ons: int = 1,
        timestep: int = 1,
        n_dofs: int = 100,
        **kwargs
) -> np.ndarray:
    """
    :param jacobian:
    :param trajectory: The discretized samples, with shape
        (n_timesteps, n_dofs),
    :param n_burn_in: The number of initial transients to discard
    :param n_exponents: The number of lyapunov exponents to
        calculate (in decreasing order).  Leave this blank to
        compute the full spectrum.
    :param t_ons: To lower computational burden, we do not perform
        the full reorthonormalization step with each step in the
        trajectory.  Instead, we reorthonormalize every `t_ons`
        steps.
        TODO: Iteratively compute the optimal `t_ons`
    :param n_dofs:
    """

    # The reorthonormalization interval
    w_ons = kwargs.get("w_ons", round(t_ons / timestep))

    # There are a total of `n_dofs` lyapunov exponents (1 for every degree of freedom)
    # However, we only return the largest `n_exponents` of these
    if n_exponents is None:
        n_exponents = n_dofs

    assert 0 < n_exponents <= n_dofs, f"The # of exponents, {n_exponents}, must be positive & less than {n_dofs}."

    # We renormalize (/sample) only once every `t_ons` steps
    n_samples = (trajectory.shape[0] - n_burn_in) // w_ons

    # q will update at each timestep
    # r will update only every `t_ons` steps
    q = random_orthonormal([n_dofs, n_exponents])
    _ = np.zeros([n_exponents, n_exponents])

    # Burn in so Q can relax to the Osedelets matrix
    # for t, state in enumerate(tqdm(trajectory[:n_burn_in],
    #                                desc="Burning-in Osedelets matrix")):
    #     q = jacobian(state) @ q
    #
    #     if t % w_ons == 0:
    #         q, _ = qr_positive(q)

    q = qr_evolve(trajectory, jacobian, q, "Burning-in Osedelets matrix", w_ons, False)

    # Run the actual decomposition on the remaining steps
    q, lyapunov_spectrum = qr_evolve(trajectory, jacobian, q, "Reorthnormalizing", w_ons, True)

    # The Lyapunov exponents are the time-averaged logarithms of the
    # on-diagonal (i.e scaling) elements of R
    lyapunov_spectrum /= n_samples * t_ons

    return lyapunov_spectrum

def get_attractor_dimension(lyapunov_spectrum: np.ndarray) -> float:
    """
    The attractor dimensionality is given by the interpolated number of Lyapunov exopnents that sum to $0$

    This assumes that lyapunov_spectrum is already in decreasing order.
    """
    k = 0
    _sum = 0

    while _sum >= 0 and k < lyapunov_spectrum.size - 1:
        _sum += lyapunov_spectrum[k]
        k += 1

    _sum -= lyapunov_spectrum[k - 1]

    return k + _sum / np.abs(lyapunov_spectrum[k])  # Python counts from 0!
