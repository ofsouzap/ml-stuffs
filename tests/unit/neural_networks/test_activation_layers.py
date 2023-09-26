from typing import Iterable, Tuple, Callable
import pytest
import numpy as np
import numpy.typing as npt
from tests.test_util import *
from neural_networks.layers import ActivationLayer, ReluActivationLayer, SigmoidActivationLayer


DEFAULT_LEARNING_RATE = 1e-3


_LAYER_GENS: Iterable[Callable[[int, float], ActivationLayer]] = [
    lambda n, lr: ReluActivationLayer(n, lr),
    lambda n, lr: SigmoidActivationLayer(n, lr),
]


_FORWARDS_CASES_INPS: Iterable[Tuple[npt.NDArray]] = [
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


def _cross_forward_cases_with_layers(layer_gens: Iterable[Callable[[int, float], ActivationLayer]], cases: Iterable[Tuple[npt.NDArray]]) -> Iterable[Tuple[ActivationLayer, npt.NDArray]]:
    for case in cases:
        for layer_gen in layer_gens:
            inp = case[0]
            yield (layer_gen(inp.shape[0], DEFAULT_LEARNING_RATE), inp)


_FORWARDS_CASES_AUTO_CALC = _cross_forward_cases_with_layers(_LAYER_GENS, _FORWARDS_CASES_INPS)


_BACKWARDS_CASES: Iterable[Tuple[ActivationLayer, npt.NDArray, npt.NDArray, npt.NDArray]] = [
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


def _calc_auto_exp_forwards(func: Callable[[npt.NDArray], npt.NDArray], inp: npt.NDArray) -> npt.NDArray:

    assert inp.ndim == 1

    out = func(inp)

    return out


@pytest.mark.parametrize(["layer", "inp"], _FORWARDS_CASES_AUTO_CALC)
def test_forwards_cases_auto_calc(layer: ActivationLayer, inp: npt.NDArray):

    # Arrange
    exp = _calc_auto_exp_forwards(layer.func.f, inp)

    # Act
    out = layer.forwards(inp)

    # Assert
    assert_allclose(out, exp)


@pytest.mark.parametrize(["layer", "x", "grad_wrt_y", "exp"], _BACKWARDS_CASES)
def test_backwards_cases(layer: ActivationLayer, x: npt.NDArray, grad_wrt_y: npt.NDArray, exp: npt.NDArray):
    assert x.ndim == grad_wrt_y.ndim == exp.ndim == 1
    assert x.shape == grad_wrt_y.shape == exp.shape

    # Act
    out = layer.backwards(x, grad_wrt_y)

    # Assert
    assert_allclose(out, exp, atol=1e-4)
