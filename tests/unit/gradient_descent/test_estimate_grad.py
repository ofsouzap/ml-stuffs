from typing import List, Tuple, Callable
import pytest
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose
from gradient_descent import estimate_grads


_CASES: List[Tuple[Callable, npt.NDArray, npt.NDArray]] = [
    (
        lambda poss: np.sum(np.square(poss), axis=1),
        np.array([
            [0, 0, 0],
            [0, 3, 2],
            [2.3, 4, 1],
            [-5.3, 0, 0.5],
            [7, -10, 1.5],
        ]),
        np.array([
            [0,0,0],
            [0,6,4],
            [4.6,8,2],
            [-10.6,0,1],
            [14,-20,3]
        ])
    ),
    (
        lambda poss: poss[:,0],
        np.array([
            [0, 0, 0],
            [0, 3, 2],
            [2.3, 4, 1],
            [-5.3, 0, 0.5],
            [7, -10, 1.5],
        ]),
        np.array([
            [1,0,0],
            [1,0,0],
            [1,0,0],
            [1,0,0],
            [1,0,0],
        ])
    ),
    (
        lambda poss: poss[:,1]**3,
        np.array([
            [0, 0, 0],
            [0, 3, 2],
            [2.3, 4, 1],
            [-5.3, 0, 0.5],
            [7, -10, 1.5],
        ]),
        np.array([
            [0,0,0],
            [0,27,0],
            [0,48,0],
            [0,0,0],
            [0,300,0],
        ])
    ),
]


@pytest.mark.parametrize(("vec_field_func", "inps", "exps"), _CASES)
def test_dimensions(vec_field_func, inps: npt.NDArray[np.float64], exps: npt.NDArray[np.float64]):
    result = estimate_grads(vec_field_func, inps)
    assert result.ndim == 2


@pytest.mark.parametrize(("vec_field_func", "inps", "exps"), _CASES)
def test_values(vec_field_func, inps: npt.NDArray[np.float64], exps: npt.NDArray[np.float64]):
    result = estimate_grads(vec_field_func, inps)
    assert_allclose(result, exps, atol=1e-07)
