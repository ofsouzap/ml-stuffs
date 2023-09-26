from typing import NamedTuple, Callable
import numpy as np
import numpy.typing as npt


class DiffVectorVectorFunction(NamedTuple):
    """A function operating on and returning a vector that is differentiable"""
    f: Callable[[npt.NDArray], npt.NDArray]
    grad_f: Callable[[npt.NDArray], npt.NDArray]


class NNCostFunction(NamedTuple):
    """A differentiable cost function for a neural network"""

    f: Callable[[npt.NDArray, npt.NDArray], float]
    """The cost function.

Parameters:

    observed - the observed output vector

    expected - the expected output vector

Returns:

    cost - the calculated cost of the observation
"""

    grad_f: Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
    """The derivative function of the cost function w.r.t. the observed output vector (i.e. ∂ cost / ∂ observed_value).

Parameters:

    observed - the observed output vector

    expected - the expected output vector

Returns:

    grad - the calculated gradient of the cost of the observation with respect to the observation
"""


relu = DiffVectorVectorFunction(
    lambda x: np.maximum(np.zeros_like(x), x),
    lambda x: (x > 0).astype(x.dtype)
)


sigmoid = DiffVectorVectorFunction(
    lambda x: 1 / (1 + np.exp(-x)),  # σ = 1 / (1 + e^-x)
    lambda x: np.exp(-x) / np.square(1 + np.exp(-x))  # dσ/dx = e^-x / (1 + e^-x)^2
)


sum_of_squares_cost = NNCostFunction(
    lambda obs, exp: np.sum(np.square(obs - exp), axis=0),
    lambda obs, exp: 2 * (obs - exp),
)
