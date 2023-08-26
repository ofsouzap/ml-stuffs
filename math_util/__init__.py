from typing import Optional
import numpy as np
from numpy import typing as npt
from numpy import linalg
from check import *
from progress import Progress


class DimensionTooHighError(Exception):
    pass


def covariance_matrix(xs: npt.NDArray, progress: Optional[Progress] = None) -> npt.NDArray:
    """Returns a scaled covariance matrix created from the input data

Parameters:

    xs - a NxM array of input data

Returns:

    cov_mat - an MxM real-symmetric scaled covariance matrix from the input data. Note that the result will not be normalised so only the proportions are valid
"""

    check_is_mat(xs, "Input must be a matrix")

    N = xs.shape[0]
    M = xs.shape[1]

    cov_mat = np.zeros(shape=(M,M), dtype=np.float32)

    if progress:
        progress.max = N

    # Using Python iteration due to memory limitations for larger data sets

    for i in range(N):

        if progress:
            progress.next()

        cov_mat += np.matmul(
            xs[i,:,np.newaxis],
            xs[i,:,np.newaxis].T
        )

    if progress:
        progress.finish()

    check_mat_dim(cov_mat, M)

    return cov_mat


def principal_components(m: npt.NDArray) -> npt.NDArray:
    """Returns an ordered list of the unit-vector principal components of a covariance matrix using Principal Component Analysis

Parameters:

    m - an NxN covariance matrix of numeric values

Returns:

    axes - an NxN array of principal components where each value is a unit N-vector
"""

    check_is_mat(m, "m must be a matrix (ie an array with ndim=2)")
    check_mat_square(m, "m must be a square matrix")

    evals, evecs = linalg.eig(m)
    evecs = evecs.T

    sort_order = np.argsort(evals)[::-1]

    ordered_evecs = evecs[sort_order,:]

    return ordered_evecs


def project_to_components(xs: npt.NDArray, comps: npt.NDArray) -> npt.NDArray:
    return xs @ comps.T


def pca(xs: npt.NDArray, N: int, covarince_matrix_progress: Optional[Progress] = None) -> npt.NDArray:
    """Uses Principal Component Analysis (PCA) to reduce the dimensionality of some data points

Parameters:

    xs - a PxM array with the data the reduce the dimensionality of

    N - the number of dimensions to reduce the data to. If this is greater than the current dimensionality of the data then an exception is thrown

    covariance_matrix_progress (optional) - a progress tracker to use for showing the progress of the covariance matrix creation

Returns:

    ys - a PxN array of the data after having its dimensionality reduce through PCA
"""

    check_is_mat(xs, "xs should be a matrix")

    if N > xs.shape[1]:
        raise DimensionTooHighError()

    cov_mat = covariance_matrix(xs, progress=covarince_matrix_progress)
    comps = principal_components(cov_mat)[:N]

    # comps is NxM

    ys = project_to_components(xs, comps)

    return ys
