from typing import NamedTuple, Callable
import numpy as np
import numpy.typing as npt


class DifferentiableVectorFunction(NamedTuple):
    """A function operating on a vector that is differentiable"""
    f: Callable[[npt.NDArray], npt.NDArray]
    grad_f: Callable[[npt.NDArray], npt.NDArray]


relu = DifferentiableVectorFunction(
    lambda x: np.max(0, x),
    lambda x: (x > 0).astype(x.dtype)
)


sigmoid = DifferentiableVectorFunction(
    lambda x: 1 / (1 + np.exp(-x)),
    lambda x: np.exp(-x) / np.square(1 + np.exp(-x))
)
