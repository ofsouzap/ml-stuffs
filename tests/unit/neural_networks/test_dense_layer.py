from typing import Iterable, Tuple
import pytest
import numpy as np
import numpy.typing as npt
from tests.test_util import *
from neural_networks.layers import DenseLayer


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


_FORWARDS_CASES_AUTO_CALC_SINGLE: Iterable[Tuple[npt.NDArray, npt.NDArray, npt.NDArray]] = [
    (
        np.array([
            [ 1, 2 ],
            [ 3, 4 ],
            [ 5, 6 ],
        ], dtype=np.float64),
        np.zeros(shape=(2,), dtype=np.float64),
        np.array([ 1, 1, 1 ], dtype=np.float64),
    ),
    (
        np.array([
            [ 1, 2 ],
            [ 3, 4 ],
            [ 5, 6 ],
        ], dtype=np.float64),
        np.zeros(shape=(2,), dtype=np.float64),
        np.array([ 1, -6, 3.2 ], dtype=np.float64),
    ),
    (
        np.array([
            [ 1, 2 ],
            [ 3, 4 ],
            [ 5, 6 ],
        ], dtype=np.float64),
        np.array([ 0.1, -0.1 ], dtype=np.float64),
        np.array([ 1, -6, 3.2 ], dtype=np.float64),
    ),
    (
        np.array([
            [ 1, 2, -2.4, 0.5 ],
            [ 3, 4.1, 1, 2 ],
            [ 5, 6, -6, 3 ],
        ], dtype=np.float64),
        np.array([ 0.1, 0.2, -0.3, 10.2 ], dtype=np.float64),
        np.array([ -1.0, 1.24, 0.4 ], dtype=np.float64),
    ),
    (
        np.array([
            [ 0, 0 ],
            [ 0, 0 ],
            [ 0, 0 ],
        ], dtype=np.float64),
        np.zeros(shape=(2,), dtype=np.float64),
        np.array([ 0, -0.5, 52 ], dtype=np.float64),
    ),
    (
        np.array([
            [ 0, 0, 0, 0 ],
            [ 0, 0, 0, 0 ],
            [ 0, 0, 0, 0 ],
        ], dtype=np.float64),
        np.ones(shape=(4,), dtype=np.float64),
        np.array([ 0, -0.5, 52 ], dtype=np.float64),
    ),
]


_FORWARDS_CASES_AUTO_CALC_MULTI: Iterable[Tuple[npt.NDArray, npt.NDArray, npt.NDArray]] = [
    (
        np.array([
            [ 1, 2 ],
            [ 3, 4 ],
            [ 5, 6 ],
        ], dtype=np.float64),
        np.zeros(shape=(2,), dtype=np.float64),
        np.array([
            [ 1, 1, 1 ],
            [ 0, 2, 1.2 ],
            [ 0, 4, 3 ],
        ], dtype=np.float64),
    ),
    (
        np.array([
            [ 1, 2 ],
            [ 3, 4 ],
            [ 5, 6 ],
        ], dtype=np.float64),
        np.zeros(shape=(2,), dtype=np.float64),
        np.array([
            [ 1, -6, 3.2 ],
        ], dtype=np.float64),
    ),
    (
        np.array([
            [ 1, 2 ],
            [ 3, 4 ],
            [ 5, 6 ],
        ], dtype=np.float64),
        np.array([ 0.1, -0.1 ], dtype=np.float64),
        np.array([
            [ 1, -6, 3.2 ],
            [ 0, 0, 0 ],
            [ 7.5, 3.1, -100 ],
            [ 1, 1, 1 ],
        ], dtype=np.float64),
    ),
    (
        np.array([
            [ 1, 2, -2.4, 0.5 ],
            [ 3, 4.1, 1, 2 ],
            [ 5, 6, -6, 3 ],
        ], dtype=np.float64),
        np.array([ 0.1, 0.2, -0.3, 10.2 ], dtype=np.float64),
        np.array([
            [ -1.0, 1.24, 0.4 ],
            [ 0, 0, 0 ],
            [ 1, 1, 1 ],
            [ 2, 2, 2 ],
            [ -100, 100, 100 ],
            [ 200, 200, 5 ],
            [ 3.14, 1.59, 2.63 ],
        ], dtype=np.float64),
    ),
]


_BACKWARDS_CASES_CHANGE_DIRS_SINGLE: Iterable[Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]] = [
    (
        np.array([
            [ 1 ],
            [ 1 ],
        ], dtype=np.float64),
        np.zeros(shape=(1,), dtype=np.float64),
        np.ones(shape=(2,), dtype=np.float64),
        np.ones(shape=(1,), dtype=np.float64),
        np.zeros(shape=(2,1), dtype=np.bool_),
        np.zeros(shape=(1,), dtype=np.bool_),
    ),
    (
        np.array([
            [ 1 ],
            [ 1 ],
        ], dtype=np.float64),
        np.zeros(shape=(1,), dtype=np.float64),
        np.ones(shape=(2,), dtype=np.float64),
        -np.ones(shape=(1,), dtype=np.float64),
        np.ones(shape=(2,1), dtype=np.bool_),
        np.ones(shape=(1,), dtype=np.bool_),
    ),
    (
        np.array([
            [ 3, -3 ],
            [ 4, 0 ],
            [ 1, 1 ],
        ], dtype=np.float64),
        np.array([ 0, 2 ], dtype=np.float64),
        np.array([ 1, -2, 10 ], dtype=np.float64),
        np.array([ 1, -3.2 ], np.float64),
        np.array([
            [ False, True ],
            [ True, False ],
            [ False, True ],
        ], dtype=np.bool_),
        np.array([ False, True ], dtype=np.bool_),
    ),
]


_BACKWARDS_CASES_CHANGE_DIRS_MULTI: Iterable[Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]] = [
    (
        np.array([
            [ 1 ],
            [ 1 ],
        ], dtype=np.float64),
        np.zeros(shape=(1,), dtype=np.float64),
        np.array([
            [ 1, 1 ],
            [ 2, 1 ],
        ], dtype=np.float64),
        np.array([
            [ 1 ],
            [ 1 ],
        ], dtype=np.float64),
        np.zeros(shape=(2,1), dtype=np.bool_),
        np.zeros(shape=(1,), dtype=np.bool_),
    ),
    (
        np.array([
            [ 1 ],
            [ 1 ],
        ], dtype=np.float64),
        np.zeros(shape=(1,), dtype=np.float64),
        np.array([
            [ 1, 1 ],
            [ 3, 5 ],
        ], dtype=np.float64),
        np.array([
            [ -1 ],
            [ -1 ],
        ], dtype=np.float64),
        np.ones(shape=(2,1), dtype=np.bool_),
        np.ones(shape=(1,), dtype=np.bool_),
    ),
    (
        np.array([
            [ 3, -3 ],
            [ 4, 0 ],
            [ 1, 1 ],
        ], dtype=np.float64),
        np.array([ 0, 2 ], dtype=np.float64),
        np.array([
            [ 1, -2, 10 ],
            [ 0, 0, 10 ],
            [ 1, -5, 10 ],
        ], dtype=np.float64),
        np.array([
            [ 1, -3.2 ],
            [ 1, -3.2 ],
            [ 1, -3.2 ],
        ], np.float64),
        np.array([
            [ False, True ],
            [ True, False ],
            [ False, True ],
        ], dtype=np.bool_),
        np.array([ False, True ], dtype=np.bool_),
    ),
]


_BACKWARDS_CASES_OUTPUT_GRADS_SINGLE: Iterable[Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]] = [
    (
        np.array([
            [ 1 ],
            [ 1 ],
        ], dtype=np.float64),
        np.zeros(shape=(1,), dtype=np.float64),
        np.ones(shape=(2,), dtype=np.float64),
        np.ones(shape=(1,), dtype=np.float64),
        np.array([ 1, 1 ], np.float64),
    ),
    (
        np.array([
            [ 1 ],
            [ 1 ],
        ], dtype=np.float64),
        np.zeros(shape=(1,), dtype=np.float64),
        np.ones(shape=(2,), dtype=np.float64),
        -np.ones(shape=(1,), dtype=np.float64),
        np.array([ -1, -1 ], np.float64),
    ),
    (
        np.array([
            [ 3, -3 ],
            [ 4, 0 ],
            [ 1, 1 ],
        ], dtype=np.float64),
        np.array([ 0, 2 ], dtype=np.float64),
        np.array([ 1, -2, 10 ], dtype=np.float64),
        np.array([ 1, -3.2 ], np.float64),
        np.array([ 12.6, 4, -2.2 ], np.float64),
    ),
    (
        np.array([
            [ 0 ],
            [ 0 ],
        ], dtype=np.float64),
        np.zeros(shape=(1,), dtype=np.float64),
        np.zeros(shape=(2,), dtype=np.float64),
        np.ones(shape=(1,), dtype=np.float64),
        np.zeros(shape=(2,), dtype=np.float64),
    ),
]


_BACKWARDS_CASES_OUTPUT_GRADS_MULTI: Iterable[Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]] = [
    (
        np.array([
            [ 1 ],
            [ 1 ],
        ], dtype=np.float64),
        np.zeros(shape=(1,), dtype=np.float64),
        np.array([
            [ 1, 1 ],
            [ 0, 0 ],
            [ 1, -1 ],
        ], dtype=np.float64),
        np.array([
            [ 1 ],
            [ 1 ],
            [ 1 ],
        ], dtype=np.float64),
        np.array([
            [ 1, 1 ],
            [ 1, 1 ],
            [ 1, 1 ],
        ], np.float64),
    ),
    (
        np.array([
            [ 3, -3 ],
            [ 4, 0 ],
            [ 1, 1 ],
        ], dtype=np.float64),
        np.array([ 0, 2 ], dtype=np.float64),
        np.array([
            [ 1, -2, 10 ],
            [ 0, 3, 14 ],
        ], np.float64),
        np.array([
            [ 1, -3.2 ],
            [ 1, -3.2 ],
        ], np.float64),
        np.array([
            [ 12.6, 4, -2.2 ],
            [ 12.6, 4, -2.2 ],
        ], np.float64),
    ),
]


def _create(w: npt.NDArray, b: npt.NDArray, learning_rate: float = DEFAULT_LEARNING_RATE) -> DenseLayer:
    return DenseLayer(
        n=w.shape[0],
        m=w.shape[1],
        learning_rate=learning_rate,
        weights=w,
        bias=b
    )


def _calc_auto_exp_forwards_single(w: npt.NDArray, b: npt.NDArray, inp: npt.NDArray) -> npt.NDArray:
    assert inp.ndim == b.ndim == 1
    assert w.ndim == 2
    assert w.shape[0] == inp.shape[0]
    assert w.shape[1] == b.shape[0]

    N, M = w.shape[0], w.shape[1]

    out = np.zeros(shape=(w.shape[1]))

    # Weights

    for j in range(M):

        out[j] = 0  # Just being safe, however unnecessary

        for i in range(N):

            add_val = (inp[i] * w[i,j])
            out[j] += add_val

    # Biases

    for j in range(M):
        out[j] += b[j]

    return out


def _calc_auto_exp_forwards_multi(w: npt.NDArray, b: npt.NDArray, inps: npt.NDArray) -> npt.NDArray:
    assert inps.ndim == 2
    assert w.ndim == 2
    assert b.ndim == 1
    assert w.shape[0] == inps.shape[1]
    assert w.shape[1] == b.shape[0]

    N, M = w.shape[0], w.shape[1]

    out = np.zeros(shape=(inps.shape[0],w.shape[1],))

    # Weights

    for j in range(M):

        out[:,j] = 0  # Just being safe, however unnecessary

        for i in range(N):

            add_vals = (inps[:,i] * w[i,j])
            out[:,j] += add_vals

    # Biases

    for j in range(M):
        out[:,j] += b[j]

    return out


def _run_test_forwards_single(w: npt.NDArray, b: npt.NDArray, inp: npt.NDArray, exp: npt.NDArray):
    assert w.ndim == 2
    assert b.ndim == inp.ndim == exp.ndim == 1
    assert w.shape[0] == inp.shape[0]
    assert w.shape[1] == b.shape[0] == exp.shape[0]

    # Arrange
    layer = _create(w, b)

    # Act
    out = layer.forwards_single(inp)

    # Assert
    assert_allclose(out, exp)


def _run_test_forwards_multi(w: npt.NDArray, b: npt.NDArray, inps: npt.NDArray, exps: npt.NDArray):
    assert w.ndim == 2
    assert b.ndim == 1
    assert inps.ndim == exps.ndim == 2
    assert w.shape[0] == inps.shape[1]
    assert w.shape[1] == b.shape[0] == exps.shape[1]

    # Arrange
    layer = _create(w, b)

    # Act
    outs = layer.forwards_multi(inps)

    # Assert
    assert_allclose(outs, exps)


@pytest.mark.parametrize(["inp"], _IDENTITIY_CASES_SINGLE)
def test_forwards_identity_single(inp: npt.NDArray):
    N = inp.shape[0]
    w = np.identity(N)
    b = np.zeros(shape=N)
    return _run_test_forwards_single(w, b, inp, inp)


@pytest.mark.parametrize(["inps"], _IDENTITIY_CASES_MULTI)
def test_forwards_identity_multi(inps: npt.NDArray):
    assert inps.ndim == 2
    N = inps.shape[1]
    w = np.identity(N)
    b = np.zeros(shape=N)
    return _run_test_forwards_multi(w, b, inps, inps)


@pytest.mark.parametrize(["w", "b", "inp"], _FORWARDS_CASES_AUTO_CALC_SINGLE)
def test_forwards_cases_auto_calc_single(w, b, inp):
    exp = _calc_auto_exp_forwards_single(w, b, inp)
    return _run_test_forwards_single(w, b, inp, exp)


@pytest.mark.parametrize(["w", "b", "inps"], _FORWARDS_CASES_AUTO_CALC_MULTI)
def test_forwards_cases_auto_calc_multi(w, b, inps):
    exps = _calc_auto_exp_forwards_multi(w, b, inps)
    return _run_test_forwards_multi(w, b, inps, exps)


@pytest.mark.parametrize(["w", "b", "xs", "grad_wrt_ys", "exp_ws_inc", "exp_bs_inc"], _BACKWARDS_CASES_CHANGE_DIRS_SINGLE)
def test_backwards_cases_change_directions_single(
    w: npt.NDArray,
    b: npt.NDArray,
    xs: npt.NDArray,
    grad_wrt_ys: npt.NDArray,
    exp_ws_inc: npt.NDArray[np.bool_],
    exp_bs_inc: npt.NDArray[np.bool_]):

    assert w.ndim == exp_ws_inc.ndim == 2
    assert b.ndim == xs.ndim == grad_wrt_ys.ndim == exp_bs_inc.ndim == 1
    assert w.shape == exp_ws_inc.shape
    assert b.shape == exp_bs_inc.shape
    assert w.shape[0] == xs.shape[0]
    assert w.shape[1] == b.shape[0]

    # Arrange

    layer = _create(w, b)

    prev_ws = layer.weights.copy()
    prev_bs = layer.bias.copy()

    # Act
    layer.backwards_single(xs, grad_wrt_ys)

    # Assert

    new_ws = layer.weights.copy()
    new_bs = layer.bias.copy()

    ws_inc = new_ws > prev_ws
    bs_inc = new_bs > prev_bs

    assert_allclose(ws_inc, exp_ws_inc)
    assert_allclose(bs_inc, exp_bs_inc)


@pytest.mark.parametrize(["w", "b", "xs", "grads_wrt_ys", "exp_ws_inc", "exp_bs_inc"], _BACKWARDS_CASES_CHANGE_DIRS_MULTI)
def test_backwards_cases_change_directions_multi(
    w: npt.NDArray,
    b: npt.NDArray,
    xs: npt.NDArray,
    grads_wrt_ys: npt.NDArray,
    exp_ws_inc: npt.NDArray[np.bool_],
    exp_bs_inc: npt.NDArray[np.bool_]):

    assert w.ndim == exp_ws_inc.ndim == 2
    assert b.ndim == exp_bs_inc.ndim == 1
    assert xs.ndim == grads_wrt_ys.ndim == 2
    assert w.shape == exp_ws_inc.shape
    assert b.shape == exp_bs_inc.shape
    assert w.shape[0] == xs.shape[1]
    assert w.shape[1] == b.shape[0] == grads_wrt_ys.shape[1]
    assert xs.shape[0] == grads_wrt_ys.shape[0]

    # Arrange

    layer = _create(w, b)

    prev_ws = layer.weights.copy()
    prev_bs = layer.bias.copy()

    # Act
    layer.backwards_multi(xs, grads_wrt_ys)

    # Assert

    new_ws = layer.weights.copy()
    new_bs = layer.bias.copy()

    ws_inc = new_ws > prev_ws
    bs_inc = new_bs > prev_bs

    assert_allclose(ws_inc, exp_ws_inc)
    assert_allclose(bs_inc, exp_bs_inc)


@pytest.mark.parametrize(["w", "b", "xs", "grad_wrt_ys", "exp"], _BACKWARDS_CASES_OUTPUT_GRADS_SINGLE)
def test_backwards_cases_output_grads_single(
    w: npt.NDArray,
    b: npt.NDArray,
    xs: npt.NDArray,
    grad_wrt_ys: npt.NDArray,
    exp: npt.NDArray):

    assert w.ndim == 2
    assert b.ndim == xs.ndim == grad_wrt_ys.ndim == exp.ndim == 1
    assert w.shape[0] == xs.shape[0] == exp.shape[0]
    assert w.shape[1] == b.shape[0]

    # Arrange
    layer = _create(w, b)

    # Act
    out = layer.backwards_single(xs, grad_wrt_ys)

    # Assert
    assert_allclose(out, exp)


@pytest.mark.parametrize(["w", "b", "xs", "grad_wrt_ys", "exp"], _BACKWARDS_CASES_OUTPUT_GRADS_MULTI)
def test_backwards_cases_output_grads_multi(
    w: npt.NDArray,
    b: npt.NDArray,
    xs: npt.NDArray,
    grad_wrt_ys: npt.NDArray,
    exp: npt.NDArray):

    assert w.ndim == 2
    assert b.ndim == 1
    assert xs.ndim == grad_wrt_ys.ndim == exp.ndim == 2
    assert w.shape[0] == xs.shape[1] == exp.shape[1]
    assert w.shape[1] == b.shape[0]
    assert xs.shape[0] == grad_wrt_ys.shape[0] == exp.shape[0]

    # Arrange
    layer = _create(w, b)

    # Act
    out = layer.backwards_multi(xs, grad_wrt_ys)

    # Assert
    assert_allclose(out, exp)
