import numpy as np
import numpy.typing as npt
from check import *


_DEFAULT_RTOL: float = 1e-05
_DEFAULT_ATOL: float = 1e-08


def arrays_close(xs: npt.NDArray,
                 ys: npt.NDArray,
                 rtol: float = _DEFAULT_RTOL,
                 atol: float = _DEFAULT_ATOL) -> bool:
    return bool(np.all(np.isclose(xs, ys, rtol=rtol, atol=atol), axis=None))


def assert_arrays_close(xs: npt.NDArray,
                        ys: npt.NDArray,
                        rtol: float = _DEFAULT_RTOL,
                        atol: float = _DEFAULT_ATOL) -> None:
    assert arrays_close(xs, ys, rtol=rtol, atol=atol)


def assert_vectors_colinear(_xs: npt.NDArray,
                            _ys: npt.NDArray,
                            allow_zero: bool = False,
                            rtol: float = _DEFAULT_RTOL,
                            atol: float = _DEFAULT_ATOL) -> None:

    assert _xs.ndim == _ys.ndim, "Arrays must have same dimensionality"
    assert _xs.ndim in [1, 2], "Assertion function only valid for vectors or arrays of vectors (ie where ndim==1 or ndim==2)"

    xs: npt.NDArray
    ys: npt.NDArray

    if _xs.ndim == 1:
        xs = _xs[:, np.newaxis]
        ys = _ys[:, np.newaxis]
    else:
        xs = _xs
        ys = _ys

    # Check if zero

    assert (
        allow_zero or (
            np.all(np.any(np.logical_not(np.isclose(xs, 0, rtol=rtol, atol=atol)), axis=1), axis=0) and
            np.all(np.any(np.logical_not(np.isclose(ys, 0, rtol=rtol, atol=atol)), axis=1), axis=0)
        )
    ), "A vector is zero and isn't allowed to be"

    # Check that colinear

    facs = ys[:,0] / xs[:,0]
    scaled_xs = xs * (facs[:,np.newaxis])

    assert_arrays_close(scaled_xs, ys, rtol=rtol, atol=atol)


def assert_matrices_proportional(x: npt.NDArray,
                                 y: npt.NDArray,
                                 allow_zero: bool = False,
                                 rtol: float = _DEFAULT_RTOL,
                                 atol: float = _DEFAULT_ATOL) -> None:

    assert x.ndim == 2, "Assertion function only valid for matrices"

    # Check if zero

    assert (
        allow_zero or (
            np.logical_not(np.all(np.isclose(x, 0, rtol=rtol, atol=atol), axis=None)) and
            np.logical_not(np.all(np.isclose(y, 0, rtol=rtol, atol=atol), axis=None))
        )
    ), "A matrix is zero and isn't allowed to be"

    # Check that proportional

    fac = y[0,0] / x[0,0]
    scaled_xs = x * fac

    assert_arrays_close(scaled_xs, y, rtol=rtol, atol=atol)


def assert_matrix_symmetric(m: npt.NDArray) -> None:

    check_is_mat(m)

    for i in range(m.shape[0]):
        for j in range(i):
            assert np.isclose(m[i,j], m[j,i])
