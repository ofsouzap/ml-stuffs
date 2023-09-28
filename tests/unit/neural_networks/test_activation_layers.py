from typing import Iterable, Tuple, Callable
import pytest
import numpy as np
import numpy.typing as npt
from tests.test_util import *
from math_util.vector_functions import DiffVectorVectorFunction
from neural_networks.layers import ActivationLayer, ReluActivationLayer, SigmoidActivationLayer


DEFAULT_LEARNING_RATE = 1e-3


_LAYER_GENS: Iterable[Callable[[int, float], ActivationLayer]] = [
    lambda n, lr: ReluActivationLayer(n, lr),
    lambda n, lr: SigmoidActivationLayer(n, lr),
]


_FORWARDS_CASES_INPS_SINGLE: Iterable[Tuple[npt.NDArray]] = [
    (
        np.array([ 1, 1, 1 ], dtype=np.float64),
    ),
    (
        np.array([ 1, -6 ], dtype=np.float64),
    ),
    (
        np.array([ -1.0, 1.24, 0.4 ], dtype=np.float64),
    ),
    (
        np.zeros(shape=(3), dtype=np.float64),
    ),
]


_FORWARDS_CASES_INPS_MULTI: Iterable[Tuple[npt.NDArray]] = [
    (
        np.array([
            [ 1, 1, 1 ],
            [ 10, -1, 3 ],
        ], dtype=np.float64),
    ),
    (
        np.array([
            [ 1, -6 ],
        ], dtype=np.float64),
    ),
    (
        np.array([
            [ -1.0, 1.24, 0.4 ],
            [ 0, 0, 1 ],
            [ 12, 124, -9.04 ],
        ], dtype=np.float64),
    ),
    (
        np.zeros(shape=(10,3), dtype=np.float64),
    ),
]


def _cross_forward_cases_with_layers(layer_gens: Iterable[Callable[[int, float], ActivationLayer]], cases: Iterable[Tuple[npt.NDArray]]) -> Iterable[Tuple[ActivationLayer, npt.NDArray]]:
    out = []
    for case in cases:
        for layer_gen in layer_gens:
            inp = case[0]
            out.append((layer_gen(inp.shape[0], DEFAULT_LEARNING_RATE), inp))
    return out


_FORWARDS_CASES_AUTO_CALC_SINGLE = _cross_forward_cases_with_layers(_LAYER_GENS, _FORWARDS_CASES_INPS_SINGLE)
_FORWARDS_CASES_AUTO_CALC_MULTI = _cross_forward_cases_with_layers(_LAYER_GENS, _FORWARDS_CASES_INPS_MULTI)


_BACKWARDS_CASES_SINGLE: Iterable[Tuple[ActivationLayer, npt.NDArray, npt.NDArray, npt.NDArray]] = [
    (
        ReluActivationLayer(4, DEFAULT_LEARNING_RATE),
        np.array([ -1.2, 0.0, 5.3, 502 ], np.float64),
        np.array([ 0, 0, 0, 0 ], np.float64),
        np.array([ 0, 0, 0, 0 ], np.float64),
    ),
    (
        ReluActivationLayer(6, DEFAULT_LEARNING_RATE),
        np.array([ 4.3,  7,   4,   -2,   1,     1 ], np.float64),
        np.array([ 1.0, -1.5, 0.0, -5.3, 0.1, -10 ], np.float64),
        np.array([ 1.0, -1.5, 0.0,  0,   0.1, -10 ], np.float64),
    ),
    (
        SigmoidActivationLayer(4, DEFAULT_LEARNING_RATE),
        np.array([ -1.2, 0.0, 5.3, 502 ], np.float64),
        np.array([ 0, 0, 0, 0 ], np.float64),
        np.array([ 0, 0, 0, 0 ], np.float64),
    ),
    (
        SigmoidActivationLayer(6, DEFAULT_LEARNING_RATE),
        np.array([ 4.3,  7,   4,   -2,   1,     1 ], np.float64),
        np.array([ 1.0, -1.5, 0.0, -5.3, 0.1, -10 ], np.float64),
        np.array([
            0.01320770826,
            -1.36533177e-3,
            0,
            -0.5564660026,
            0.019661196332,
            -1.966119332
        ], np.float64),
    ),
]


_BACKWARDS_CASES_MULTI: Iterable[Tuple[ActivationLayer, npt.NDArray, npt.NDArray, npt.NDArray]] = [
    (
        ReluActivationLayer(4, DEFAULT_LEARNING_RATE),
        np.array([
            [ -1.2, 0.0, 5.3, 502 ],
            [ 0, 1, 1, -1 ],
        ], dtype=np.float64),
        np.array([
            [ 0, 0, 0, 0 ],
            [ 0, 0, 0, 0 ],
        ], dtype=np.float64),
        np.array([
            [ 0, 0, 0, 0 ],
            [ 0, 0, 0, 0 ],
        ], dtype=np.float64),
    ),
    (
        ReluActivationLayer(6, DEFAULT_LEARNING_RATE),
        np.array([
            [ 4.3,  7,   4,   -2,   1,     1 ],
            [ -2, -1, 1, 1, 1, 5 ],
        ], dtype=np.float64),
        np.array([
            [ 1.0, -1.5,  0.0, -5.3,  0.1, -10 ],
            [ 1.0,  5.0, -2.0,  3.3, -2.3, -50 ],
        ], dtype=np.float64),
        np.array([
            [ 1.0, -1.5,  0,    0,    0.1, -10 ],
            [ 0,    0.0, -2.0,  3.3, -2.3, -50 ],
        ], dtype=np.float64),
    ),
    (
        SigmoidActivationLayer(4, DEFAULT_LEARNING_RATE),
        np.array([
            [ -1.2, 0.0, 5.3, 502 ],
            [ 1, 0, 4, 2 ],
        ], dtype=np.float64),
        np.array([
            [ 0, 0, 0, 0 ],
            [ 0, 0, 0, 0 ],
        ], dtype=np.float64),
        np.array([
            [ 0, 0, 0, 0 ],
            [ 0, 0, 0, 0 ],
        ], dtype=np.float64),
    ),
    (
        SigmoidActivationLayer(6, DEFAULT_LEARNING_RATE),
        np.array([
            [ 4.3,  7,   4,   -2,   1,     1 ],
            [ 4.3,  7,   4,   -2,   1,     1 ],
        ], dtype=np.float64),
        np.array([
            [ 1.0, -1.5, 0.0, -5.3, 0.1, -10 ],
            [ 1.0, -1.5, 0.0, -5.3, 0.1, -10 ],
        ], dtype=np.float64),
        np.array([
            [
                0.01320770826,
                -1.36533177e-3,
                0,
                -0.5564660026,
                0.019661196332,
                -1.966119332
            ],
            [
                0.01320770826,
                -1.36533177e-3,
                0,
                -0.5564660026,
                0.019661196332,
                -1.966119332
            ],
        ], dtype=np.float64),
    ),
]


def _auto_calc_exp_forwards_single(func: DiffVectorVectorFunction, inp: npt.NDArray) -> npt.NDArray:

    assert inp.ndim == 1

    out = func.f_single(inp)

    return out


def _auto_calc_exp_forwards_multi(func: DiffVectorVectorFunction, inps: npt.NDArray) -> npt.NDArray:

    assert inps.ndim == 2

    outs = func.f_multi(inps)

    return outs


@pytest.mark.parametrize(["layer", "inp"], _FORWARDS_CASES_AUTO_CALC_SINGLE)
def test_forwards_cases_auto_calc_single(layer: ActivationLayer, inp: npt.NDArray):

    # Arrange
    exp = _auto_calc_exp_forwards_single(layer.func, inp)

    # Act
    out = layer.forwards_single(inp)

    # Assert
    assert_allclose(out, exp)


@pytest.mark.parametrize(["layer", "inps"], _FORWARDS_CASES_AUTO_CALC_MULTI)
def test_forwards_cases_auto_calc_multi(layer: ActivationLayer, inps: npt.NDArray):

    # Arrange
    exps = _auto_calc_exp_forwards_multi(layer.func, inps)

    # Act
    outs = layer.forwards_multi(inps)

    # Assert
    assert_allclose(outs, exps)


@pytest.mark.parametrize(["layer", "x", "grad_wrt_y", "exp"], _BACKWARDS_CASES_SINGLE)
def test_backwards_cases_single(layer: ActivationLayer, x: npt.NDArray, grad_wrt_y: npt.NDArray, exp: npt.NDArray):
    assert x.ndim == grad_wrt_y.ndim == exp.ndim == 1
    assert x.shape == grad_wrt_y.shape == exp.shape

    # Act
    out = layer.backwards_single(x, grad_wrt_y)

    # Assert
    assert_allclose(out, exp, atol=1e-4)


@pytest.mark.parametrize(["layer", "xs", "grads_wrt_ys", "exps"], _BACKWARDS_CASES_MULTI)
def test_backwards_cases_multi(layer: ActivationLayer, xs: npt.NDArray, grads_wrt_ys: npt.NDArray, exps: npt.NDArray):
    assert xs.ndim == grads_wrt_ys.ndim == exps.ndim == 2
    assert xs.shape == grads_wrt_ys.shape == exps.shape

    # Act
    outs = layer.backwards_multi(xs, grads_wrt_ys)

    # Assert
    assert_allclose(outs, exps, atol=1e-4)
