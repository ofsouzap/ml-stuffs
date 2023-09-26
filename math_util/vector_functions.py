from typing import NamedTuple, Callable
import numpy as np
import numpy.typing as npt


class DifferentiableVectorFunction(NamedTuple):
    """A function operating on a vector that is differentiable"""
    f: Callable[[npt.NDArray], npt.NDArray]
    grad_f: Callable[[npt.NDArray], npt.NDArray]


relu = DifferentiableVectorFunction(
    lambda x: np.maximum(np.zeros_like(x), x),
    lambda x: (x > 0).astype(x.dtype)
)


sigmoid = DifferentiableVectorFunction(
    lambda x: 1 / (1 + np.exp(-x)),  # σ = 1 / (1 + e^-x)
    lambda x: np.exp(-x) / np.square(1 + np.exp(-x))  # dσ/dx = e^-x / (1 + e^-x)^2
)
