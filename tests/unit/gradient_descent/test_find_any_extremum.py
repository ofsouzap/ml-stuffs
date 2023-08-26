from typing import List, Dict, Tuple, Iterable
import pytest
import numpy as np
import numpy.typing as npt
from gradient_descent import find_any_extremum


_CASES: List[Tuple[Dict, Iterable[npt.NDArray[np.float64]]]] = [
    (
        {
            "vec_field_func": lambda poss: np.sum(np.square(poss), axis=1),
            "start": np.array([0,0]),
        },
        [np.array([0,0])]
    ),
    (
        {
            "vec_field_func": lambda poss: np.sum(np.square(poss), axis=1),
            "start": np.array([-2.0,4.0]),
        },
        [np.array([0,0])]
    ),
]


@pytest.mark.parametrize(("params", "exps"), _CASES)
def test_dimensions(params: Dict, exps: Iterable[npt.NDArray[np.float64]]):
    result = find_any_extremum(**params)
    assert result.ndim == 1


@pytest.mark.parametrize(("params", "exps"), _CASES)
def test_results(params: Dict, exps: Iterable[npt.NDArray[np.float64]]):
    result = find_any_extremum(**params)
    assert any(map(lambda exp: np.allclose(result, exp, atol=1e-04), exps))
