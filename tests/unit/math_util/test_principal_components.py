import pytest
import numpy as np
from tests.test_util import *
from math_util.pca import principal_components


_CASES = [
    (
        np.array([
            [ 1 ]
        ], dtype=np.float32),
        np.array([
            [ 1 ],
        ], dtype=np.float32)
    ),
    (
        np.array([
            [18, 26],
            [26, 45]
        ], dtype=np.float32),
        np.array([
            [ 0.60753478, 1 ],
            [ -1.64599631, 1 ],
        ], dtype=np.float32)
    ),
    (
        np.array([
            [ 18, 26, 33],
            [ 26, 45, 60],
            [ 33, 60, 81]
        ], dtype=np.float32),
        np.array([
            [ 0.42969178, 0.74848035, 1 ],
            [ -2.11556720, -0.12152379, 1 ],
            [ 0.56816867, -1.66221786, 1 ],
        ], dtype=np.float32)
    ),
    (
        np.array([
            [ 30, 40, 50 ],
            [ 40, 54, 68 ],
            [ 50, 68, 86 ],
        ], dtype=np.float32),
        np.array([
            [ 0.58679955, 0.79339977, 1 ],
            [ -1.42013288, -0.21006644, 1 ],
            [ 1, -2, 1 ],
        ], dtype=np.float32)
    ),
    (
        np.array([
            [ 20.81, -7.56, -22.56 ],
            [ -7.56, 182.33, 36.93 ],
            [ -22.56, 36.93, 90.45 ],
        ], dtype=np.float32),
        np.array([
            [ -0.24550606, 2.72358946, 1 ],
            [ -0.31494145, -0.39555155, 1 ],
            [ 3.26652151, -0.07271623, 1 ],
        ], dtype=np.float32)
    ),
    (
        np.array([
            [125, 15, 50],
            [15, 26, 17],
            [50, 17, 25],
        ], dtype=np.float32),
        np.array([
            [2.33814136, 0.423012569, 1],
            [-0.957343082, 2.92758074, 1],
            [-19, -25, 55],
        ], dtype=np.float32)
    ),
]


@pytest.mark.parametrize("xs", [inp for inp, _ in _CASES])
def test_dimensions(xs: npt.NDArray):
    out = principal_components(xs)
    assert out.ndim == 2
    assert out.shape[0] == xs.shape[0]
    assert out.shape[1] == xs.shape[0]


@pytest.mark.parametrize(("xs", "ys"), _CASES)
def test_values(xs: npt.NDArray, ys: npt.NDArray):
    out = principal_components(xs)
    assert_vectors_colinear(out, ys)
