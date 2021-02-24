from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def count_trivial_fixed_pts(trajectory: np.ndarray, atol: float = 1e-3) -> int:
    """
    :param trajectory: the trajectory of shape [n_dofs, n_timesteps]
    :param atol: This is the absolute numerical tolerance that
        determines whether a subtrajectory has reached 0.
    """

    # 1. We transform the trajectory into a binary array according to
    #    whether a point is within (=> 0)the given threshold of zeros or not (=> 1)
    trajectory_bin = np.where(np.isclose(trajectory.T, 0., atol=atol), 0, 1)

    # 2. We count the number of columns with only zeros.
    trajectory_collapsed = np.sum(trajectory_bin, axis=1)
    fixed_pt_neurons = np.where(trajectory_collapsed == 0, 1, 0)

    return np.sum(fixed_pt_neurons)


def count_fixed_pts(trajectory: np.ndarray, atol: float = 1e-3) -> int:
    """
    :param trajectory: the trajectory of shape [n_dofs, n_timesteps]
    :param atol: This is the absolute numerical tolerance that
        determines whether a subtrajectory has reached 0.
    """
    # 1. We transform the trajectory into a binary array according to
    #    whether a point is within (=> 0)the given threshold of its final value or not (=> 1)

    initial_state = np.array([trajectory.T[:, -1]]).T * np.ones(trajectory.T.shape)
    trajectory_bin = np.where(np.isclose(trajectory.T, initial_state, atol=atol),
                              0, 1)

    # 2. We count the number of columns with only zeros.
    trajectory_collapsed = np.sum(trajectory_bin, axis=1)
    fixed_pt_neurons = np.where(trajectory_collapsed == 0, 1, 0)

    return np.sum(fixed_pt_neurons)


def count_cycles(trajectory: np.ndarray, atol: float = 1e-1, max_n_steps: Optional[int] = None,
                 verbose: bool = False) -> int:
    """
    :param trajectory: the trajectory of shape [n_dofs, n_timesteps]

    inspiration from https://stackoverflow.com/a/17090200/1701415
    """
    n_dofs, n_timesteps = trajectory.T.shape

    if max_n_steps and max_n_steps < n_timesteps:
        trajectory = trajectory.T[:, (n_timesteps - max_n_steps):]
    else:
        trajectory = trajectory.T

    cycles = np.zeros(n_dofs)

    # TODO: See if you can do this without the explicit for loop
    for i in tqdm(range(trajectory.shape[0]), desc="Counting cycles..."):
        path = trajectory[i, :]

        # 1. We compute the trajectories' autocorrelations (individually per neuron)
        path_normalized = path - np.mean(path)
        path_norm = np.sum(path_normalized ** 2)
        acor = np.correlate(path_normalized, path_normalized, "full") / path_norm
        acor = acor[len(acor) // 2:]  # Autocorrelation is symmetrical about the half-way point
        # TODO: Use a more efficient way
        # TODO: figure out the effects of the boundary

        # 2. Figure out where the autocorrelation peaks are
        acor_peaks = np.where(np.logical_and(acor > np.roll(acor, 1),
                                             acor > np.roll(acor, -1)), acor, 0)

        # 3. Figure out whether these peaks are within our tolerance of 1.
        #    We subtract one because the first entry will always have perfect autocorrelation.
        close_peaks = np.where(acor_peaks > 1. - atol, 1., 0.)
        is_cycle = np.sum(close_peaks) - 1. > 0

        cycles[i] = is_cycle

        if verbose:
            # While debugging
            plt.plot(path[:10000])
            plt.show()

            plt.plot(acor)
            plt.plot(close_peaks)
            plt.show()

    print(np.sum(cycles))

    return np.sum(cycles)


def participation_ratio(trajectory, max_n_steps: Optional[int] = None):
    """
    TODO: consistency in shape of trajectory
    :param trajectory: the trajectory of shape [n_timesteps, n_dofs]
    """
    n_timesteps = trajectory.shape[1]

    if max_n_steps and max_n_steps < n_timesteps:
        trajectory = trajectory[:, (n_timesteps - max_n_steps):]

    covariance = np.cov(trajectory)
    eigvals, _ = np.linalg.eig(covariance)

    return np.power(np.sum(eigvals), 2) / np.sum(np.power(eigvals, 2))


def test_count_trivial_fixed_pts():
    trajectory = np.array([[0., 1., 2.], [0., 0., 0.], [0., 0., 0.],
                           [1., 0., 0.]])
    assert count_trivial_fixed_pts(trajectory.T) == 2
    assert count_fixed_pts(trajectory.T) == count_trivial_fixed_pts(trajectory.T)


def test_count_fixed_pts():
    trajectory = np.array([[0., 1., 2.], [0., 0., 0.], [2., 2., 2.],
                           [1., 1., 1.]])
    assert count_fixed_pts(trajectory.T) == 3
    assert count_fixed_pts(trajectory.T) >= count_trivial_fixed_pts(trajectory.T)


def test_count_cycles():
    x = np.arange(0, 100 * np.pi, 0.01)
    signal_1 = np.sin(x)
    signal_2 = np.random.uniform(size=len(x))
    signal_3 = np.sin(2 * x) + np.cos(3 * x)
    signal_4 = np.ones(len(x))
    signal_5 = np.random.uniform(size=len(x))

    trajectory = np.array([signal_1, signal_2, signal_3, signal_4, signal_5])

    assert count_cycles(trajectory.T, 0.1, verbose=False) == 2


def test_participation_ratio():
    # Fully dependent components -> D = 1
    signal_1 = np.arange(5)
    trajectory_1 = np.array([signal_1, 2 * signal_1, 3 * signal_1]).T
    assert np.isclose(participation_ratio(trajectory_1), 1.)

    # Fully independent components -> D = N (number of components)
    trajectory_2 = np.eye(100, 5).T

    assert np.isclose(participation_ratio(trajectory_2), 5., atol=0.1)
