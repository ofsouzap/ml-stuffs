from typing import Iterable, Iterator, Tuple, Callable
import pytest
import numpy as np
import numpy.typing as npt
from tests.test_util import *
from math_util.vector_functions import NNCostFunction, sum_of_squares_cost
from neural_networks.networks import Network
from neural_networks.layers import DenseLayer, ReluActivationLayer, SigmoidActivationLayer


_DEFAULT_LEARNING_RATE: float = 1e-6


_SAMPLE_NETWORK_GENS: Iterable[Callable[[], Network]] = [
    lambda: Network([  # Identity dense layer
        DenseLayer(
            3, 3, _DEFAULT_LEARNING_RATE,
            np.array([
                [ 1, 0, 0 ],
                [ 0, 1, 0 ],
                [ 0, 0, 1 ],
            ], dtype=np.float64),
            np.array([ 0, 0, 0 ], np.float64),
        ),
    ]),
    lambda: Network([  # Identity dense layer and then ReLU
        DenseLayer(
            3, 3, _DEFAULT_LEARNING_RATE,
            np.array([
                [ 1, 0, 0 ],
                [ 0, 1, 0 ],
                [ 0, 0, 1 ],
            ], dtype=np.float64),
            np.array([ 0, 0, 0 ], np.float64),
        ),
        ReluActivationLayer(3, _DEFAULT_LEARNING_RATE),
    ]),
    lambda: Network([  # Identity dense layer and then Sigmoid
        DenseLayer(
            3, 3, _DEFAULT_LEARNING_RATE,
            np.array([
                [ 1, 0, 0 ],
                [ 0, 1, 0 ],
                [ 0, 0, 1 ],
            ], dtype=np.float64),
            np.array([ 0, 0, 0 ], np.float64),
        ),
        SigmoidActivationLayer(3, _DEFAULT_LEARNING_RATE),
    ]),
    lambda: Network([  # Some dense layer, ReLU, another dense layer, Sigmoid
        DenseLayer(
            3, 2, _DEFAULT_LEARNING_RATE,
            np.array([
                [ 1, 3 ],
                [ -3, 10 ],
                [ -2, 0 ],
            ], dtype=np.float64),
            np.array([ 5, -1 ], np.float64),
        ),
        ReluActivationLayer(2, _DEFAULT_LEARNING_RATE),
        DenseLayer(
            2, 1, _DEFAULT_LEARNING_RATE,
            np.array([
                [ 2 ],
                [ 1 ],
            ], dtype=np.float64),
            np.array([ 1 ], np.float64),
        ),
        SigmoidActivationLayer(1, _DEFAULT_LEARNING_RATE),
    ]),
]


def _cross_case_gens_with_networks(cases: Iterable[Callable[[int, int], Tuple]], networks: Iterable[Callable[[], Network]]) -> Iterator[Tuple]:
    for case_gen in cases:
        for network_gen in networks:
            network = network_gen()
            case_ = case_gen(network.input_n, network.output_n)
            yield tuple([network] + list(case_))


_FORWARDS_AUTO_CALC_CASE_GENS: Iterable[Callable[[int, int], Tuple[npt.NDArray]]] = [
    lambda n, m: (np.zeros(shape=(n,)),),
    lambda n, m: (np.ones(shape=(n,)),),
    lambda n, m: (np.arange(n),),
    lambda n, m: (np.linspace(-50, 50, n),),
]


_FORWARDS_AUTO_CALC_CASES = _cross_case_gens_with_networks(_FORWARDS_AUTO_CALC_CASE_GENS, _SAMPLE_NETWORK_GENS)


_FORWARDS_MANUAL_CALC_CASES: Iterable[Tuple[Network, npt.NDArray, npt.NDArray]] = [
    (
        Network([
            DenseLayer(
                3, 1, _DEFAULT_LEARNING_RATE,
                np.array([
                    [ 1.0 ],
                    [ 2.0 ],
                    [ 3.0 ],
                ], dtype=np.float64)
            ),
            ReluActivationLayer(1, _DEFAULT_LEARNING_RATE),
        ]),
        np.array([ 0.0, -2.0, 3.5 ], dtype=np.float64),
        np.array([ 6.5 ], dtype=np.float64),
    ),
    (
        Network([
            DenseLayer(
                3, 1, _DEFAULT_LEARNING_RATE,
                np.array([
                    [ 1.0 ],
                    [ 2.0 ],
                    [ 3.0 ],
                ], dtype=np.float64)
            ),
            ReluActivationLayer(1, _DEFAULT_LEARNING_RATE),
        ]),
        np.array([ 0.0, -2.0, -3.5 ], dtype=np.float64),
        np.array([ 0.0 ], dtype=np.float64),
    ),
]


_LEARN_PROGRESS_CASE_GENS: Iterable[Callable[[int, int], Tuple[npt.NDArray, npt.NDArray, NNCostFunction, int]]] = [
    lambda n, m: (  # Zero
        np.ones(shape=(n,), dtype=np.float64),
        np.zeros(m, dtype=np.float64),
        sum_of_squares_cost,
        10
    ),
    lambda n, m: (  # Scaling factor
        np.arange(n, dtype=np.float64),
        2*np.arange(m, dtype=np.float64),
        sum_of_squares_cost,
        10
    ),
    lambda n, m: (  # Linear re-ordering
        np.arange(n, dtype=np.float64),
        5*np.arange(m, dtype=np.float64)[::-1],
        sum_of_squares_cost,
        10
    ),
    lambda n, m: (  # Linear combination
        np.arange(n, dtype=np.float64),
        10*np.arange(m, dtype=np.float64)[::-1] + 3*np.arange(m, dtype=np.float64),
        sum_of_squares_cost,
        10
    ),
]


_LEARN_PROGRESS_SINGLE_CASES = _cross_case_gens_with_networks(_LEARN_PROGRESS_CASE_GENS, _SAMPLE_NETWORK_GENS)


def _auto_calc_exp_forwards(network: Network, inp: npt.NDArray) -> npt.NDArray:
    assert inp.ndim == 1

    curr = inp.copy()
    for layer in network.layers:
        curr = layer.forwards_single(curr)

    return curr


@pytest.mark.parametrize(["network", "inp"], _FORWARDS_AUTO_CALC_CASES)
def test_forwards_auto_calc(network: Network, inp: npt.NDArray):

    # Arrange
    exp = _auto_calc_exp_forwards(network, inp)

    # Act
    out = network.forwards(inp)

    # Assert
    assert_allclose(out, exp)


@pytest.mark.parametrize(["network", "inp", "exp"], _FORWARDS_MANUAL_CALC_CASES)
def test_forwards_manual_calc(network: Network, inp: npt.NDArray, exp: npt.NDArray):

    # Act
    out = network.forwards(inp)

    # Assert
    assert_allclose(out, exp)


@pytest.mark.parametrize(["network", "inp", "net_exp", "cost_func", "iteration_count"], _LEARN_PROGRESS_SINGLE_CASES)
def test_learn_progress_single(network: Network, inp: npt.NDArray, net_exp: npt.NDArray, cost_func: NNCostFunction, iteration_count: int):

    # Arrange
    orig_cost = network.calculate_cost(inp, net_exp, cost_func)

    # Act
    for _ in range(iteration_count):
        network.learn_step_single(inp, net_exp, cost_func)
    new_cost = network.calculate_cost(inp, net_exp, cost_func)

    # Assert
    assert new_cost < orig_cost, "Cost hasn't been reduced"
