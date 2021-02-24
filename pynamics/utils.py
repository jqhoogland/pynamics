"""

Contains helper functions for other modules.

Author: Jesse Hoogland
Year: 2020

"""

import hashlib
import logging
import os
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp


def np_cache(dir_path: str = "./saves/", file_prefix: Optional[str] = None, ignore: Optional[list] = []):
    """
    A wrapper to load a previous response to a function (or to run the function and save the result otherwise).
    Assuming the function returns a np.ndarray as its response
    """

    def inner(func):
        def wrapper(*args, save=True, load=True, **kwargs):

            relevant_args = [*args]
            relevant_kwargs = {**kwargs}

            for ignore_arg in ignore:
                if isinstance(ignore_arg, int):
                    relevant_args.pop(ignore_arg)
                else:
                    del relevant_kwargs[ignore_arg]

            relevant_args = tuple(relevant_args)
            params = (str(relevant_args) + str(relevant_kwargs)).encode('utf-8')

            file_name = file_prefix + hashlib.md5(params).hexdigest() + ".npy"
            file_path = os.path.join(dir_path, file_name)

            if not os.path.isdir(dir_path):
                logging.info("Creating directory %s", dir_path)
                os.mkdir(dir_path)

            if os.path.isfile(file_path) and load:
                logging.info("Loading from save %s", file_path)
                return np.load(file_path)

            response = func(*args, **kwargs)

            if save:
                logging.info("Saving to %s", file_path)
                np.save(file_path, response)

            return response

        return wrapper

    return inner


def qr_positive(a: np.ndarray, *args,
                **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    q, r = np.linalg.qr(a, *args, **kwargs)
    diagonal_signs = np.sign(np.diagonal(r))
    return q @ np.diag(diagonal_signs), np.diag(
        diagonal_signs) @ r  # TODO: make sure these are aligned correctly


def random_orthonormal(shape: Tuple[int, int]):
    # Source: https://stackoverflow.com/a/38430739/1701415
    a = np.random.randn(*shape)
    q, r = qr_positive(a)
    return q


def eigsort(A, k: Optional[int]=None, which="LM", eig_method="sp"):
    if k is None:
        k = A.shape[0]

    if (k >= A.shape[0] - 1):
        eig_method = "np"

    eig_vals, eig_vecs = None, None

    if (eig_method == "sp"):
        eig_vals, eig_vecs = sp.linalg.eigs(A, k, which=which)
    elif (eig_method == "np"):
        eig_vals, eig_vecs = np.linalg.eig(A)

    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    return (eig_vals, eig_vecs)


def normalize_rows(w):
    # Source: https://stackoverflow.com/a/59365444/1701415
    # Find the row scalars as a Matrix_(n,1)
    row_sum_w = np.abs(w).sum(axis=1)

    row_sum_w[row_sum_w == 0] = 1. # If a row has 0 weight, it stays 0

    scaling_matrix = np.eye(w.shape[0]) / row_sum_w

    return scaling_matrix.dot(w)


def svd_whiten(X):
    # Source: https://stackoverflow.com/a/11336203/1701415

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white


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


# ------------------------------------------------------------
# TESTING

def test_qr_positive():
    a = np.random.uniform(size=(100, 50))
    q, r = qr_positive(a)

    logging.debug(a.shape, q.shape, r.shape)
    logging.debug(a, q @ r)

    assert np.allclose(a, q @ r)
    assert q.shape == (100, 50)
    assert np.allclose(q.T @ q, np.eye(50))
    assert r.shape == (50, 50)
    assert np.allclose(a, q @ r)
    assert np.all(np.diagonal(r) >= 0)


def test_random_orthonormal():
    q = random_orthonormal((100, 50))

    assert q.shape == (100, 50)
    assert np.allclose(q.T @ q, np.eye(50))


def test_normalize_rows():
    assert np.allclose(
        normalize_rows(
                np.array([[5, 1, 4], [0, 1, 1], [1, 1, 2]],
                         dtype="float64")),
            np.array([[0.5, 0.1, 0.4], [0, 0.5, 0.5], [0.25, 0.25, 0.5]],
                     dtype="float64"),
    )

def test_normalize_rows_1():
    assert np.allclose(
        normalize_rows(
                np.array([[5, 1, 4], [0, 0, 0], [1, 1, 2]],
                         dtype="float64")),
            np.array([[0.5, 0.1, 0.4], [0, 0, 0], [0.25, 0.25, 0.5]],
                     dtype="float64"),
    )
def test_eigsort():

    assert np.allclose(eigsort(np.array([[2, 0, 0], [0, 4, 0], [0, 0,3]]), k=10, eig_method="np")[0],
                       np.array([4, 3, 2]))
