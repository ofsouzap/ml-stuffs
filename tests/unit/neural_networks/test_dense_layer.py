from typing import Iterable, Tuple
import pytest
import numpy as np
import numpy.typing as npt
from tests.test_util import *
from neural_networks.layers import DenseLayer


_IDENTITIY_CASES: Iterable[Tuple[int, npt.NDArray]] = [
    (
        2,
        np.array([ 5, 3 ], dtype=np.float64)
    ),
    (
        3,
        np.array([ -4, 3, 6.4 ], dtype=np.float64)
    ),
    (
        4,
        np.array([ 5.3, 6.0, -1.0, 0.0 ], dtype=np.float64)
    ),
    (
        4,
        np.array([ -1.0, -2.1, 9.9, 2.4 ], dtype=np.float64)
    ),
    (
        6,
        np.array([ 0.0, 0.3, 0.0, 0.0, 5.6, 2.3 ], dtype=np.float64)
    ),
    (
        6,
        np.array([ 10.4, -1742, 94, -0.5316, 256.3, 0.1 ], dtype=np.float64)
    ),
]


_FORWARDS_CASES_AUTO_CALC: Iterable[Tuple[npt.NDArray, npt.NDArray, npt.NDArray]] = [
    (
        np.array([
            [ 1, 2 ],
            [ 3, 4 ],
            [ 5, 6 ],
        ], dtype=np.float64),
        np.zeros(shape=(2,), dtype=np.float64),
        np.array([ 1, 1, 1 ], dtype=np.float64),
    ),
    # TODO - more test cases
]


def _create(w: npt.NDArray, b: npt.NDArray, learning_rate: float = 1e-3) -> DenseLayer:
    return DenseLayer(
        n=w.shape[0],
        m=w.shape[1],
        learning_rate=learning_rate,
        weights=w,
        bias=b
    )


def _calc_auto_exp_forwards(w: npt.NDArray, b: npt.NDArray, inp: npt.NDArray) -> npt.NDArray:
    assert inp.ndim == b.ndim == 1
    assert w.ndim == 2
    assert w.shape[0] == inp.shape[0]
    assert w.shape[1] == b.shape[0]

    N, M = w.shape[0], w.shape[1]

    out = np.zeros(shape=(w.shape[1]))

    for j in range(M):

        out[j] = 0  # Just being safe, however unnecessary

        for i in range(N):

            add_val = (inp[i] * w[i,j]) + b[j]
            out[j] += add_val

    return out


def _run_test_forwards(w: npt.NDArray, b: npt.NDArray, inp: npt.NDArray, exp: npt.NDArray):
    assert w.ndim == 2
    assert b.ndim == inp.ndim == exp.ndim == 1
    assert w.shape[0] == inp.shape[0]
    assert w.shape[1] == b.shape[0] == exp.shape[0]

    # Arrange
    layer = _create(w, b)

    # Act
    out = layer.forwards(inp)

    # Assert
    assert_allclose(out, exp)


@pytest.mark.parametrize(["N", "inp"], _IDENTITIY_CASES)
def test_forwards_identity(N: int, inp: npt.NDArray):
    assert inp.shape == (N,)
    w = np.identity(N)
    b = np.zeros(shape=N)
    return _run_test_forwards(w, b, inp, inp)


@pytest.mark.parametrize(["w", "b", "inp"], _FORWARDS_CASES_AUTO_CALC)
def test_forwards_cases_auto_calc(w, b, inp):
    exp = _calc_auto_exp_forwards(w, b, inp)
    return _run_test_forwards(w, b, inp, exp)


# TODO - backwards propagation tests
