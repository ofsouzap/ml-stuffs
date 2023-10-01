from typing import Iterable, Tuple, Callable, List
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


def _cross_case_gens_with_networks(cases: Iterable[Callable[[int, int], Tuple]], networks: Iterable[Callable[[], Network]]) -> Iterable[Tuple]:

    outs = []

    for case_gen in cases:
        for network_gen in networks:
            network = network_gen()
            case_ = case_gen(network.input_n, network.output_n)
            outs.append(tuple([network] + list(case_)))

    return outs


_FORWARDS_AUTO_CALC_SINGLE_CASE_GENS: Iterable[Callable[[int, int], Tuple[npt.NDArray]]] = [
    lambda n, m: (np.zeros(shape=(n,)),),
    lambda n, m: (np.ones(shape=(n,)),),
    lambda n, m: (np.arange(n),),
    lambda n, m: (np.linspace(-50, 50, n),),
]


_FORWARDS_AUTO_CALC_MULTI_CASE_GENS: Iterable[Callable[[int, int], Tuple[npt.NDArray]]] = [
    lambda n, m: (np.zeros(shape=(2,n,)),),
    lambda n, m: (np.ones(shape=(3,n,)),),
    lambda n, m: (np.array([list(range(n)) for _ in range(m)]),),
    lambda n, m: (np.array([np.linspace(10*i, -10*i, n) for i in range(n+m)]),),
]


_FORWARDS_AUTO_CALC_SINGLE_CASES = _cross_case_gens_with_networks(_FORWARDS_AUTO_CALC_SINGLE_CASE_GENS, _SAMPLE_NETWORK_GENS)
_FORWARDS_AUTO_CALC_MULTI_CASES = _cross_case_gens_with_networks(_FORWARDS_AUTO_CALC_MULTI_CASE_GENS, _SAMPLE_NETWORK_GENS)


_FORWARDS_MANUAL_CALC_SINGLE_CASES: Iterable[Tuple[Network, npt.NDArray, npt.NDArray]] = [
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


_FORWARDS_MANUAL_CALC_MULTI_CASES: Iterable[Tuple[Network, npt.NDArray, npt.NDArray]] = [
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
        np.array([
            [  0.0, -2.0,  3.5 ],
            [  0.0, -2.0, -3.5 ],
            [  1.0,  1.0,  1.0 ],
        ], dtype=np.float64),
        np.array([
            [ 6.5 ],
            [ 0.0 ],
            [ 6.0 ],
        ], dtype=np.float64),
    ),
]


_FULL_FORWARDS_MANUAL_CALC_MULTI_CASES: Iterable[Tuple[Network, npt.NDArray, List[npt.NDArray]]] = [
    (
        Network([
            DenseLayer(
                2, 4, _DEFAULT_LEARNING_RATE,
                np.array([
                    [ 1, 0, 1, -1 ],
                    [ 0, 1, -1, 1 ],
                ], dtype=np.float64),
                np.array([ 0, 0, 0, 0 ], dtype=np.float64),
            ),
            DenseLayer(
                4, 1, _DEFAULT_LEARNING_RATE,
                np.array([
                    [ 1 ],
                    [ -2 ],
                    [ -2 ],
                    [ 1 ],
                ], dtype=np.float64),
                np.array([ 2 ], dtype=np.float64),
            ),
        ]),
        np.array([
            [ 1, -0.5 ],
            [ 0, 0 ],
        ], dtype=np.float64),
        [
            np.array([
                [ 1, -0.5 ],
                [ 0, 0 ],
            ], dtype=np.float64),
            np.array([
                [ 1, -0.5, 1.5, -1.5 ],
                [ 0, 0, 0, 0 ],
            ], dtype=np.float64),
            np.array([
                [ -0.5 ],
                [ 2 ],
            ], dtype=np.float64),
        ]
    ),
]


_LEARN_PROGRESS_SINGLE_CASE_GENS: Iterable[Callable[[int, int], Tuple[npt.NDArray, npt.NDArray, NNCostFunction, int]]] = [
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


_LEARN_PROGRESS_MULTI_CASE_GENS: Iterable[Callable[[int, int], Tuple[npt.NDArray, npt.NDArray, NNCostFunction, int]]] = [
    lambda n, m: (  # Zero
        np.ones(shape=(10,n), dtype=np.float64),
        np.zeros(shape=(10,m), dtype=np.float64),
        sum_of_squares_cost,
        10,
    ),
    lambda n, m: (  # Scaling factor
        np.array([list(range(n)) for _ in range(5)], dtype=np.float64),
        2*np.array([list(range(m)) for _ in range(5)], dtype=np.float64),
        sum_of_squares_cost,
        10,
    ),
    lambda n, m: (  # Linearly-scaled re-ordering
        np.array([list(range(n)) for _ in range(5)], dtype=np.float64),
        5*np.array([list(range(m)) for _ in range(5)], dtype=np.float64)[::-1],
        sum_of_squares_cost,
        10,
    ),
    lambda n, m: (  # Linear combination
        np.array([list(range(n)) for _ in range(5)], dtype=np.float64),
        10*np.array([list(range(m)) for _ in range(5)], dtype=np.float64)[::-1] + 3*np.array([list(range(m)) for _ in range(5)], dtype=np.float64),
        sum_of_squares_cost,
        10,
    ),
    lambda n, m: (  # 2*x
        np.array([
            list(range(n)),
            [1 for _ in range(n)],
            [0 for _ in range(n)],
        ], dtype=np.float64),
        np.array([
            [2*x for x in range(m)],
            [2 for _ in range(m)],
            [0 for _ in range(m)],
        ], dtype=np.float64),
        sum_of_squares_cost,
        10,
    )
]


_LEARN_PROGRESS_SINGLE_CASES = _cross_case_gens_with_networks(_LEARN_PROGRESS_SINGLE_CASE_GENS, _SAMPLE_NETWORK_GENS)
_LEARN_PROGRESS_MULTI_CASES = _cross_case_gens_with_networks(_LEARN_PROGRESS_MULTI_CASE_GENS, _SAMPLE_NETWORK_GENS)


_LEARN_STEP_OUTPUT_SINGLE_CASES: Iterable[Tuple[str, Tuple[Network, npt.NDArray, npt.NDArray, NNCostFunction, npt.NDArray]]] = [
    (
        "W002",
        (
            Network([
                DenseLayer(
                    2, 4, _DEFAULT_LEARNING_RATE,
                    np.array([
                        [ 1, 0, 1, -1 ],
                        [ 0, 1, -1, 1 ],
                    ], dtype=np.float64),
                    np.array([ 0, 0, 0, 0 ], dtype=np.float64),
                ),
                DenseLayer(
                    4, 1, _DEFAULT_LEARNING_RATE,
                    np.array([
                        [ 1 ],
                        [ -2 ],
                        [ -2 ],
                        [ 1 ],
                    ], dtype=np.float64),
                    np.array([ 2 ], dtype=np.float64),
                ),
            ]),
            np.array([ 1, -0.5 ], dtype=np.float64),
            np.array([ 2.5 ], dtype=np.float64),
            sum_of_squares_cost,
            np.array([ 12, -6 ], dtype=np.float64),
        ),
    ),
]


_LEARN_STEP_OUTPUT_MULTI_CASES: Iterable[Tuple[str, Tuple[Network, npt.NDArray, npt.NDArray, NNCostFunction, npt.NDArray]]] = [
    (
        "W002",
        (
            Network([
                DenseLayer(
                    2, 4, _DEFAULT_LEARNING_RATE,
                    np.array([
                        [ 1, 0, 1, -1 ],
                        [ 0, 1, -1, 1 ],
                    ], dtype=np.float64),
                    np.array([ 0, 0, 0, 0 ], dtype=np.float64),
                ),
                DenseLayer(
                    4, 1, _DEFAULT_LEARNING_RATE,
                    np.array([
                        [ 1 ],
                        [ -2 ],
                        [ -2 ],
                        [ 1 ],
                    ], dtype=np.float64),
                    np.array([ 2 ], dtype=np.float64),
                ),
            ]),
            np.array([
                [ 1, -0.5 ],
                [ 0, 0 ],
            ], dtype=np.float64),
            np.array([
                [ 2.5 ],
                [ -2.5 ],
            ], dtype=np.float64),
            sum_of_squares_cost,
            np.array([
                [ 12, -6 ],
                [ -18, 9 ],
            ], dtype=np.float64),
        ),
    ),
]


_LEARN_STOCHASTIC_PROGRESS_MULTI_CASE_GENS: Iterable[Callable[[int, int], Tuple[npt.NDArray, npt.NDArray, NNCostFunction, int, int]]] = [
    lambda n, m: (  # Zero
        np.ones(shape=(10,n), dtype=np.float64),
        np.zeros(shape=(10,m), dtype=np.float64),
        sum_of_squares_cost,
        5,
        10,
    ),
    lambda n, m: (  # Scaling factor
        np.array([list(range(n)) for _ in range(5)], dtype=np.float64),
        2*np.array([list(range(m)) for _ in range(5)], dtype=np.float64),
        sum_of_squares_cost,
        3,
        10,
    ),
    lambda n, m: (  # Linearly-scaled re-ordering
        np.array([list(range(n)) for _ in range(5)], dtype=np.float64),
        5*np.array([list(range(m)) for _ in range(5)], dtype=np.float64)[::-1],
        sum_of_squares_cost,
        3,
        10,
    ),
    lambda n, m: (  # Linear combination
        np.array([list(range(n)) for _ in range(5)], dtype=np.float64),
        10*np.array([list(range(m)) for _ in range(5)], dtype=np.float64)[::-1] + 3*np.array([list(range(m)) for _ in range(5)], dtype=np.float64),
        sum_of_squares_cost,
        3,
        10,
    ),
    lambda n, m: (  # 2*x
        np.array([
            list(range(n)),
            [1 for _ in range(n)],
            [0 for _ in range(n)],
        ], dtype=np.float64),
        np.array([
            [2*x for x in range(m)],
            [2 for _ in range(m)],
            [0 for _ in range(m)],
        ], dtype=np.float64),
        sum_of_squares_cost,
        2,
        10,
    )
]


_LEARN_STOCHASTIC_PROGRESS_MULTI_CASES = _cross_case_gens_with_networks(_LEARN_STOCHASTIC_PROGRESS_MULTI_CASE_GENS, _SAMPLE_NETWORK_GENS)


def _auto_calc_exp_forwards_single(network: Network, inp: npt.NDArray) -> npt.NDArray:
    assert inp.ndim == 1

    curr = inp.copy()
    for layer in network.layers:
        curr = layer.forwards_single(curr)

    return curr


def _auto_calc_exp_forwards_multi(network: Network, inps: npt.NDArray) -> npt.NDArray:
    assert inps.ndim == 2

    currs = inps.copy()
    for layer in network.layers:
        currs = layer.forwards_multi(currs)

    return currs


@pytest.mark.parametrize(["network", "inp"], _FORWARDS_AUTO_CALC_SINGLE_CASES)
def test_forwards_auto_calc_single(network: Network, inp: npt.NDArray):

    # Arrange
    exp = _auto_calc_exp_forwards_single(network, inp)

    # Act
    out = network.forwards_single(inp)

    # Assert
    assert_allclose(out, exp)


@pytest.mark.parametrize(["network", "inps"], _FORWARDS_AUTO_CALC_MULTI_CASES)
def test_forwards_auto_calc_multi(network: Network, inps: npt.NDArray):

    # Arrange
    exps = _auto_calc_exp_forwards_multi(network, inps)

    # Act
    outs = network.forwards_multi(inps)

    # Assert
    assert_allclose(outs, exps)


@pytest.mark.parametrize(["network", "inp", "exp"], _FORWARDS_MANUAL_CALC_SINGLE_CASES)
def test_forwards_manual_calc_single(network: Network, inp: npt.NDArray, exp: npt.NDArray):

    # Act
    out = network.forwards_single(inp)

    # Assert
    assert_allclose(out, exp)


@pytest.mark.parametrize(["network", "inps", "exps"], _FORWARDS_MANUAL_CALC_MULTI_CASES)
def test_forwards_manual_calc_multi(network: Network, inps: npt.NDArray, exps: npt.NDArray):

    # Act
    outs = network.forwards_multi(inps)

    # Assert
    assert_allclose(outs, exps)


@pytest.mark.parametrize(["network", "inps", "exp_seqs"], _FULL_FORWARDS_MANUAL_CALC_MULTI_CASES)
def test_full_forwards_manual_calc_multi(network: Network, inps: npt.NDArray, exp_seqs: List[npt.NDArray]):
    assert inps.ndim == 2
    assert all(map(lambda x: x.ndim == 2, exp_seqs))

    # Act
    out_seqs = list(network.full_forwards_multi(inps))

    # Assert
    assert len(out_seqs) == len(exp_seqs)
    for i in range(len(out_seqs)):
        assert_allclose(out_seqs[i], exp_seqs[i])


@pytest.mark.parametrize(["network", "inp", "net_exp", "cost_func", "iteration_count"], _LEARN_PROGRESS_SINGLE_CASES)
def test_learn_progress_single(network: Network, inp: npt.NDArray, net_exp: npt.NDArray, cost_func: NNCostFunction, iteration_count: int):

    # Arrange
    orig_cost = network.calculate_cost_single(inp, net_exp, cost_func)

    # Act
    for _ in range(iteration_count):
        network.learn_step_single(inp, net_exp, cost_func)
    new_cost = network.calculate_cost_single(inp, net_exp, cost_func)

    # Assert
    assert new_cost < orig_cost, "Cost hasn't been reduced"


@pytest.mark.parametrize(["network", "inps", "net_exps", "cost_func", "iteration_count"], _LEARN_PROGRESS_MULTI_CASES)
def test_learn_progress_multi(network: Network, inps: npt.NDArray, net_exps: npt.NDArray, cost_func: NNCostFunction, iteration_count: int):
    assert inps.ndim == net_exps.ndim == 2
    assert inps.shape[0] == net_exps.shape[0]
    assert inps.shape[1] == network.input_n
    assert net_exps.shape[1] == network.output_n

    # Arrange
    orig_costs = network.calculate_cost_multi(inps, net_exps, cost_func)
    avg_orig_cost = sum(orig_costs) / len(orig_costs)

    # Act

    for _ in range(iteration_count):
        network.learn_step_multi(inps, net_exps, cost_func)

    new_costs = network.calculate_cost_multi(inps, net_exps, cost_func)
    assert new_costs.ndim == 1
    avg_new_cost = np.mean(new_costs)

    # Assert
    assert avg_new_cost < avg_orig_cost, "Average cost hasn't been reduced"


@pytest.mark.parametrize(
    ["network", "inp", "net_exp", "cost_func", "exp"],
    [x for _,x in _LEARN_STEP_OUTPUT_SINGLE_CASES],
    ids=[name for name,_ in _LEARN_STEP_OUTPUT_SINGLE_CASES],
)
def test_learn_step_output_single(network: Network, inp: npt.NDArray, net_exp: npt.NDArray, cost_func: NNCostFunction, exp: npt.NDArray):
    assert inp.ndim == net_exp.ndim == exp.ndim == 1

    # Act
    grad_wrt_x = network.learn_step_single(inp, net_exp, cost_func)

    # Assert
    assert_allclose(grad_wrt_x, exp)


@pytest.mark.parametrize(
    ["network", "inps", "net_exps", "cost_func", "exps"],
    [x for _,x in _LEARN_STEP_OUTPUT_MULTI_CASES],
    ids=[name for name,_ in _LEARN_STEP_OUTPUT_MULTI_CASES],
)
def test_learn_step_output_multi(network: Network, inps: npt.NDArray, net_exps: npt.NDArray, cost_func: NNCostFunction, exps: npt.NDArray):
    assert inps.ndim == net_exps.ndim == exps.ndim == 2
    assert inps.shape[0] == net_exps.shape[0] == exps.shape[0]

    # Act
    grads_wrt_xs = network.learn_step_multi(inps, net_exps, cost_func)

    # Assert
    assert_allclose(grads_wrt_xs, exps)


@pytest.mark.parametrize(["network", "inps", "net_exps", "cost_func", "sample_size", "iteration_count"], _LEARN_STOCHASTIC_PROGRESS_MULTI_CASES)
def test_learn_stochastic_progress(network: Network, inps: npt.NDArray, net_exps: npt.NDArray, cost_func: NNCostFunction, sample_size: int, iteration_count: int):
    assert inps.ndim == net_exps.ndim == 2
    assert inps.shape[0] == net_exps.shape[0]
    assert inps.shape[1] == network.input_n
    assert net_exps.shape[1] == network.output_n

    # Arrange
    orig_costs = network.calculate_cost_multi(inps, net_exps, cost_func)
    avg_orig_cost = sum(orig_costs) / len(orig_costs)

    # Act

    network.learn_stochastic(
        xs=inps,
        exps=net_exps,
        cost_func=cost_func,
        sample_size=sample_size,
        iteration_count=iteration_count,
    )

    new_costs = network.calculate_cost_multi(inps, net_exps, cost_func)
    assert new_costs.ndim == 1
    avg_new_cost = np.mean(new_costs)

    # Assert
    assert avg_new_cost < avg_orig_cost, "Average cost hasn't been reduced"
