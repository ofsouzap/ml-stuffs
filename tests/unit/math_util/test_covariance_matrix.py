import pytest
import numpy as np
import numpy.typing as npt
from tests.test_util import *
from math_util import covariance_matrix


_CASES = [
    (
        np.array([
            [2],
            [4]
        ], dtype=np.float32),
        np.array([
            [1]
        ], dtype=np.float32)
    ),
    (
        np.array([
            [1, 2],
            [4, 5],
            [1, 4]
        ], dtype=np.float32),
        np.array([
            [18, 26],
            [26, 45]
        ], dtype=np.float32)
    ),
    (
        np.array([
            [1, 2, 3],
            [4, 5, 6],
            [1, 4, 6]
        ], dtype=np.float32),
        np.array([
            [ 18, 26, 33],
            [ 26, 45, 60],
            [ 33, 60, 81]
        ], dtype=np.float32)
    ),
    (
        np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6]
        ], dtype=np.float32),
        np.array([
            [ 30, 40, 50 ],
            [ 40, 54, 68 ],
            [ 50, 68, 86 ],
        ], dtype=np.float32)
    ),
    (
        np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6]
        ], dtype=np.int32),
        np.array([
            [ 30, 40, 50 ],
            [ 40, 54, 68 ],
            [ 50, 68, 86 ],
        ], dtype=np.float32)
    ),
    (
        np.array([
            [ -1.4, -6.3,  3.3 ],
            [ -4.2,  5.0,  6.0 ],
            [  0.0, 10.0,  0   ],
            [  1.1,  4.2,  6.6 ]
        ], dtype=np.float32),
        np.array([
            [ 20.81, -7.56, -22.56 ],
            [ -7.56, 182.33, 36.93 ],
            [ -22.56, 36.93, 90.45 ],
        ], dtype=np.float32)
    ),
    (
        np.array([
            [10, -1, 3],
            [5, 5, 4],
        ], dtype=np.float32),
        np.array([
            [125, 15, 50],
            [15, 26, 17],
            [50, 17, 25],
        ], dtype=np.float32)
    ),
]


def test_empty():

    xs = np.zeros(shape=(0,0), dtype=np.float32)

    out = covariance_matrix(xs)

    assert out.ndim == 2
    assert out.shape[0] == 0
    assert out.shape[1] == 0


@pytest.mark.parametrize("xs", [inp for inp, _ in _CASES])
def test_symmetry(xs: npt.NDArray):
    out = covariance_matrix(xs)
    assert_matrix_symmetric(out)


@pytest.mark.parametrize("xs", [inp for inp, _ in _CASES])
def test_dimensions(xs: npt.NDArray):
    out = covariance_matrix(xs)
    assert out.ndim == 2
    assert out.shape[0] == xs.shape[1]
    assert out.shape[1] == xs.shape[1]


@pytest.mark.parametrize(("xs", "ys"), _CASES)
def test_values(xs: npt.NDArray, ys: npt.NDArray):
    out = covariance_matrix(xs)
    assert_matrices_proportional(out, ys)
