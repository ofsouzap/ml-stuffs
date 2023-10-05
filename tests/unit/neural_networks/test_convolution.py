from typing import List, Tuple
import pytest
import numpy as np
import numpy.typing as npt
from tests.test_util import assert_allclose
from neural_networks.convolution import *


_CONVOLVE_SINGLE_CASES: List[Tuple[str, Tuple[npt.NDArray, npt.NDArray, bool, npt.NDArray]]] = [
    (
        "Identity 1x1",
        (
            np.array([
                [ 1, 2, 3 ],
                [ 4, 5, 6 ],
                [ 7, 8, 9 ],
            ], dtype=np.float64),
            np.array([
                [ 1 ],
            ], dtype=np.float64),
            False,
            np.array([
                [ 1, 2, 3 ],
                [ 4, 5, 6 ],
                [ 7, 8, 9 ],
            ], dtype=np.float64),
        ),
    ),
    (
        "Multiplier 1x1",
        (
            np.array([
                [ 1, 2, 3 ],
                [ 4, 5, 6 ],
                [ 7, 8, 9 ],
            ], dtype=np.float64),
            np.array([
                [ 3 ],
            ], dtype=np.float64),
            False,
            np.array([
                [ 3, 6, 9 ],
                [ 12, 15, 18 ],
                [ 21, 24, 27 ],
            ], dtype=np.float64),
        ),
    ),
    (
        "Identity 3x3",
        (
            np.array([
                [ 1, 2, 3, 4, 5 ],
                [ 2, 3, 4, 5, 6 ],
                [ 3, 4, 5, 6, 7 ],
                [ 4, 5, 6, 7, 8 ],
                [ 5, 6, 7, 8, 9 ],
            ], dtype=np.float64),
            np.array([
                [ 0, 0, 0 ],
                [ 0, 1, 0 ],
                [ 0, 0, 0 ],
            ], dtype=np.float64),
            False,
            np.array([
                [ 3, 4, 5 ],
                [ 4, 5, 6 ],
                [ 5, 6, 7 ],
            ], dtype=np.float64),
        ),
    ),
    (
        "Rectangular Input Identity",
        (
            np.array([
                [ 1, 2, 3, 4, 5 ],
                [ 2, 3, 4, 5, 6 ],
                [ 3, 4, 5, 6, 7 ],
            ], dtype=np.float64),
            np.array([
                [ 0, 0, 0 ],
                [ 0, 1, 0 ],
                [ 0, 0, 0 ],
            ], dtype=np.float64),
            False,
            np.array([
                [ 3, 4, 5 ],
            ], dtype=np.float64),
        ),
    ),
    (
        "Gaussian Blur 3x3",
        (
            np.array([
                [ 1, 2, 3, 4, 5 ],
                [ 2, 3, 4, 5, 6 ],
                [ 3, 4, 5, 6, 7 ],
                [ 4, 5, 6, 7, 8 ],
                [ 5, 6, 7, 8, 9 ],
                [ 6, 7, 8, 9, 10 ],
            ], dtype=np.float64),
            np.array([
                [ 1, 3, 1 ],
                [ 3, 5, 3 ],
                [ 1, 3, 1 ],
            ], dtype=np.float64),
            False,
            np.array([
                [ 55, 84, 75 ],
                [ 84, 75, 126 ],
                [ 75, 126, 147 ],
                [ 126, 147, 168 ],
            ], dtype=np.float64),
        ),
    ),
]


@pytest.mark.parametrize(["img", "kernel_arr", "normalise", "exp"], [case for _,case in _CONVOLVE_SINGLE_CASES], ids=[label for label,_ in _CONVOLVE_SINGLE_CASES])
def test_convolve_single(img: npt.NDArray, kernel_arr: npt.NDArray, normalise: bool, exp: npt.NDArray):

    # Arrange
    kernel = Kernel(kernel_arr, normalise)

    # Act
    out = kernel.convolve_single(img)

    # Assert
    assert_allclose(out, exp)
