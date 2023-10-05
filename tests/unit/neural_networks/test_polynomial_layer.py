from typing import Iterable, Tuple, List
import pytest
import numpy as np
import numpy.typing as npt
from tests.test_util import *
from neural_networks.layers import PolynomialLayer


DEFAULT_LEARNING_RATE = 1e-3


_IDENTITIY_CASES_SINGLE: Iterable[Tuple[npt.NDArray]] = [
    (
        np.array([ 5, 3 ], dtype=np.float64),
    ),
    (
        np.array([ -4, 3, 6.4 ], dtype=np.float64),
    ),
    (
        np.array([ 5.3, 6.0, -1.0, 0.0 ], dtype=np.float64),
    ),
    (
        np.array([ -1.0, -2.1, 9.9, 2.4 ], dtype=np.float64),
    ),
    (
        np.array([ 0.0, 0.3, 0.0, 0.0, 5.6, 2.3 ], dtype=np.float64),
    ),
    (
        np.array([ 10.4, -1742, 94, -0.5316, 256.3, 0.1 ], dtype=np.float64),
    ),
]


_IDENTITIY_CASES_MULTI: Iterable[Tuple[npt.NDArray]] = [
    (
        np.array([
            [ 2, 4.0, 0.0 ],
            [ -2, 6.0, 3.0 ],
            [ -2, 1.3, 0.3 ],
        ], dtype=np.float64),
    ),
    (
        np.array([
            [ 2, 4.0, 0.0, 0.0 ],
            [ -2, 6.0, 3.0, 1.2 ],
            [ -2, 1.3, 0.3, 1.2 ],
        ], dtype=np.float64),
    ),
]


_FORWARDS_CASES_AUTO_CALC_SINGLE: Iterable[Tuple[List[npt.NDArray], npt.NDArray, npt.NDArray]] = [
    (
        [
            np.array([
                [ 1, 2 ],
                [ 3, 4 ],
                [ 5, 6 ],
            ], dtype=np.float64),
            np.array([
                [ 1, 0 ],
                [ 3, 0 ],
                [ 5, 0 ],
            ], dtype=np.float64),
            np.array([
                [ 0, 0 ],
                [ 0, 0 ],
                [ 0, 0 ],
            ], dtype=np.float64),
            np.array([
                [ 2, 2 ],
                [ 2, 2 ],
                [ 2, 2 ],
            ], dtype=np.float64),
        ],
        np.zeros(shape=(2,), dtype=np.float64),
        np.array([ 1, 1, 1 ], dtype=np.float64),
    ),
    (
        [
            np.array([
                [ 1, 2 ],
                [ 3, 4 ],
                [ 5, 6 ],
            ], dtype=np.float64),
            np.array([
                [ 0, 0 ],
                [ 0, 0 ],
                [ 0, 0 ],
            ], dtype=np.float64),
        ],
        np.zeros(shape=(2,), dtype=np.float64),
        np.array([ 1, -6, 3.2 ], dtype=np.float64),
    ),
    (
        [
            np.array([
                [ 1, 2 ],
                [ 3, 4 ],
                [ 5, 6 ],
            ], dtype=np.float64),
            np.array([
                [ 2, 5 ],
                [ 5, 5 ],
                [ 5, 5 ],
            ], dtype=np.float64),
        ],
        np.array([ 0.1, -0.1 ], dtype=np.float64),
        np.array([ 1, -6, 3.2 ], dtype=np.float64),
    ),
    (
        [
            np.array([
                [ 1, 2, -2.4, 0.5 ],
                [ 3, 4.1, 1, 2 ],
                [ 5, 6, -6, 3 ],
            ], dtype=np.float64),
            np.array([
                [ 1, 2, -4, 0.5 ],
                [ 0, 1, 0, 2.3 ],
                [ 0, 0, 0, 3 ],
            ], dtype=np.float64),
        ],
        np.array([ 0.1, 0.2, -0.3, 10.2 ], dtype=np.float64),
        np.array([ -1.0, 1.24, 0.4 ], dtype=np.float64),
    ),
    (
        [
            np.array([
                [ 0, 0 ],
                [ 0, 0 ],
                [ 0, 0 ],
            ], dtype=np.float64),
            np.array([
                [ 0, 0 ],
                [ 0, 0 ],
                [ 0, 0 ],
            ], dtype=np.float64),
        ],
        np.zeros(shape=(2,), dtype=np.float64),
        np.array([ 0, -0.5, 52 ], dtype=np.float64),
    ),
    (
        [
            np.array([
                [ 0, 0, 0, 0 ],
                [ 0, 0, 0, 0 ],
                [ 0, 0, 0, 0 ],
            ], dtype=np.float64),
            np.array([
                [ 0, 0, 0, 1 ],
                [ 0, 0, 0, 1 ],
                [ 0, 0, 1, 1 ],
            ], dtype=np.float64),
            np.array([
                [ 0, 0, 0, 0 ],
                [ 0, 0, 0, 2 ],
                [ 0, 0, 0, 0 ],
            ], dtype=np.float64),
        ],
        np.ones(shape=(4,), dtype=np.float64),
        np.array([ 0, -0.5, 52 ], dtype=np.float64),
    ),
]


_FORWARDS_CASES_AUTO_CALC_MULTI: Iterable[Tuple[List[npt.NDArray], npt.NDArray, npt.NDArray]] = [
    (
        [
            np.array([
                [ 1, 2 ],
                [ 3, 4 ],
                [ 5, 6 ],
            ], dtype=np.float64),
            np.array([
                [ 1, 0 ],
                [ 3, 0 ],
                [ 5, 0 ],
            ], dtype=np.float64),
            np.array([
                [ 0, 0 ],
                [ 0, 0 ],
                [ 0, 0 ],
            ], dtype=np.float64),
            np.array([
                [ 2, 2 ],
                [ 2, 2 ],
                [ 2, 2 ],
            ], dtype=np.float64),
        ],
        np.zeros(shape=(2,), dtype=np.float64),
        np.array([
            [ 1, 1, 1 ],
            [ 2, 1, 1 ],
            [ 1, 2, 1 ],
            [ 1, 1, 2 ],
        ], dtype=np.float64),
    ),
    (
        [
            np.array([
                [ 1, 2 ],
                [ 3, 4 ],
                [ 5, 6 ],
            ], dtype=np.float64),
            np.array([
                [ 0, 0 ],
                [ 0, 0 ],
                [ 0, 0 ],
            ], dtype=np.float64),
        ],
        np.zeros(shape=(2,), dtype=np.float64),
        np.array([
            [ 1, -6, 3.2 ],
            [ 2, 0, 2 ],
            [ 5, -6, 3.2 ],
        ], dtype=np.float64),
    ),
    (
        [
            np.array([
                [ 1, 2 ],
                [ 3, 4 ],
                [ 5, 6 ],
            ], dtype=np.float64),
            np.array([
                [ 2, 5 ],
                [ 5, 5 ],
                [ 5, 5 ],
            ], dtype=np.float64),
        ],
        np.array([ 0.1, -0.1 ], dtype=np.float64),
        np.array([
            [ 1, -6, 3.2 ],
            [ 1, -6, 3.2 ],
            [ 1, -6, 3.2 ],
            [ 1, -6, 3.2 ],
        ], dtype=np.float64),
    ),
    (
        [
            np.array([
                [ 1, 2, -2.4, 0.5 ],
                [ 3, 4.1, 1, 2 ],
                [ 5, 6, -6, 3 ],
            ], dtype=np.float64),
            np.array([
                [ 1, 2, -4, 0.5 ],
                [ 0, 1, 0, 2.3 ],
                [ 0, 0, 0, 3 ],
            ], dtype=np.float64),
        ],
        np.array([ 0.1, 0.2, -0.3, 10.2 ], dtype=np.float64),
        np.array([
            [ -1.0, 1.24, 0.4 ],
        ], dtype=np.float64),
    ),
    (
        [
            np.array([
                [ 0, 0 ],
                [ 0, 0 ],
                [ 0, 0 ],
            ], dtype=np.float64),
            np.array([
                [ 0, 0 ],
                [ 0, 0 ],
                [ 0, 0 ],
            ], dtype=np.float64),
        ],
        np.zeros(shape=(2,), dtype=np.float64),
        np.array([
            [ 0, -0.5, 52 ],
        ], dtype=np.float64),
    ),
    (
        [
            np.array([
                [ 0, 0, 0, 0 ],
                [ 0, 0, 0, 0 ],
                [ 0, 0, 0, 0 ],
            ], dtype=np.float64),
            np.array([
                [ 0, 0, 0, 1 ],
                [ 0, 0, 0, 1 ],
                [ 0, 0, 1, 1 ],
            ], dtype=np.float64),
            np.array([
                [ 0, 0, 0, 0 ],
                [ 0, 0, 0, 2 ],
                [ 0, 0, 0, 0 ],
            ], dtype=np.float64),
        ],
        np.ones(shape=(4,), dtype=np.float64),
        np.array([
            [ 0, -0.5, 52 ],
            [ 0, 0, 0 ],
        ], dtype=np.float64),
    ),
]


def _create(ws: List[npt.NDArray], b: npt.NDArray, learning_rate: float = DEFAULT_LEARNING_RATE) -> PolynomialLayer:
    return PolynomialLayer(
        n=ws[0].shape[0],
        m=ws[0].shape[1],
        order=len(ws),
        learning_rate=learning_rate,
        order_weights=ws,
        bias=b
    )


def _calc_auto_exp_forwards_single(ws: List[npt.NDArray], b: npt.NDArray, inp: npt.NDArray) -> npt.NDArray:
    assert inp.ndim == b.ndim == 1
    assert all(map(lambda w: w.ndim == 2, ws))
    assert all(map(lambda w: w.shape[0] == inp.shape[0], ws))
    assert all(map(lambda w: w.shape[1] == b.shape[0], ws))

    N, M = ws[0].shape[0], ws[0].shape[1]

    out = np.zeros(shape=(ws[0].shape[1]))

    # Weights

    for j in range(M):

        out[j] = 0  # Just being safe, however unnecessary

        for i in range(N):

            add_val = 0

            for n in range(1,len(ws)+1):
                ws_index = n-1
                w = ws[ws_index]
                add_val += (inp[i]**n) * w[i,j]

            out[j] += add_val

    # Biases

    for j in range(M):
        out[j] += b[j]

    return out


def _calc_auto_exp_forwards_multi(ws: List[npt.NDArray], b: npt.NDArray, inps: npt.NDArray) -> npt.NDArray:
    return np.array([_calc_auto_exp_forwards_single(ws, b, inp) for inp in inps])


def _run_test_forwards_single(ws: List[npt.NDArray], b: npt.NDArray, inp: npt.NDArray, exp: npt.NDArray):
    assert all(map(lambda w: w.ndim == 2, ws))
    assert b.ndim == inp.ndim == exp.ndim == 1
    assert all(map(lambda w: w.shape[0] == inp.shape[0], ws))
    assert all(map(lambda w: w.shape[1] == b.shape[0] == exp.shape[0], ws))

    # Arrange
    layer = _create(ws, b)

    # Act
    out = layer.forwards_single(inp)

    # Assert
    assert_allclose(out, exp)


def _run_test_forwards_multi(ws: List[npt.NDArray], b: npt.NDArray, inps: npt.NDArray, exps: npt.NDArray):
    assert all(map(lambda w: w.ndim == 2, ws))
    assert b.ndim == 1
    assert inps.ndim == exps.ndim == 2
    assert all(map(lambda w: w.shape[0] == inps.shape[1], ws))
    assert all(map(lambda w: w.shape[1] == b.shape[0] == exps.shape[1], ws))

    # Arrange
    layer = _create(ws, b)

    # Act
    outs = layer.forwards_multi(inps)

    # Assert
    assert_allclose(outs, exps)


@pytest.mark.parametrize(["inp"], _IDENTITIY_CASES_SINGLE)
def test_forwards_identity_single(inp: npt.NDArray):
    assert inp.ndim == 1
    N = inp.shape[0]
    ws = [np.identity(N)]
    b = np.zeros(shape=N)
    return _run_test_forwards_single(ws, b, inp, inp)


@pytest.mark.parametrize(["inps"], _IDENTITIY_CASES_MULTI)
def test_forwards_identity_multi(inps: npt.NDArray):
    assert inps.ndim == 2
    N = inps.shape[1]
    ws = [np.identity(N)]
    b = np.zeros(shape=N)
    return _run_test_forwards_multi(ws, b, inps, inps)


@pytest.mark.parametrize(["ws", "b", "inp"], _FORWARDS_CASES_AUTO_CALC_SINGLE)
def test_forwards_cases_auto_calc_single(ws, b, inp):
    exp = _calc_auto_exp_forwards_single(ws, b, inp)
    return _run_test_forwards_single(ws, b, inp, exp)


@pytest.mark.parametrize(["ws", "b", "inps"], _FORWARDS_CASES_AUTO_CALC_MULTI)
def test_forwards_cases_auto_calc_multi(ws, b, inps):
    exps = _calc_auto_exp_forwards_multi(ws, b, inps)
    return _run_test_forwards_multi(ws, b, inps, exps)
