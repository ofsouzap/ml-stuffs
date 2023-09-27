from typing import Callable
import numpy as np
import numpy.typing as npt


class DiffVectorVectorFunction:
    """An immutable differentiable function operating on and returning a vector"""

    def __init__(self,
                 f_multi: Callable[[npt.NDArray], npt.NDArray],
                 grad_f_multi: Callable[[npt.NDArray], npt.NDArray]):
        self.__f_multi = f_multi
        self.__grad_f_multi = grad_f_multi

    def f(self, x: npt.NDArray) -> npt.NDArray:
        """Apply the function to a single input.

Parameters:

    x - the input vector

Returns:

    y - the output vector
"""
        assert x.ndim == 1, "Input should be one-dimensional"

        return self.f_multi(x[np.newaxis,:])[0]

    def f_multi(self, xs: npt.NDArray) -> npt.NDArray:
        """Apply the function to multiple inputs.

Parameters:

    xs - the input vectors

Returns:

    ys - the output vectors
"""
        assert xs.ndim == 2, "Input should be two-dimensional"

        return self.__f_multi(xs)

    def grad_f(self, x: npt.NDArray) -> npt.NDArray:
        """Apply the gradient function to a single input.

Parameters:

    x - the input vector

Returns:

    y - the output vector
"""
        assert x.ndim == 1, "Input should be one-dimensional"

        return self.grad_f_multi(x[np.newaxis,:])[0]

    def grad_f_multi(self, xs: npt.NDArray) -> npt.NDArray:
        """Apply the gradient function to multiple inputs.

Parameters:

    xs - the input vectors

Returns:

    ys - the output vectors
"""
        assert xs.ndim == 2, "Input should be two-dimensional"

        return self.__grad_f_multi(xs)


class NNCostFunction:
    """An immutable differentiable cost function for a neural network"""

    def __init__(self,
                 f_multi: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
                 grad_f_multi: Callable[[npt.NDArray, npt.NDArray], npt.NDArray]):
        self.__f_multi = f_multi
        self.__grad_f_multi = grad_f_multi

    def f(self, obs: npt.NDArray, exp: npt.NDArray) -> float:
        """Apply the cost function.

Parameters:

    obs - the observed output vector

    exp - the expected output vector

Returns:

    cost - the calculated cost of the observation
"""
        assert obs.ndim == exp.ndim == 1, "Inputs must be one-dimensional"

        return self.__f_multi(
            obs[np.newaxis,:],
            exp[np.newaxis,:],
        )[0]

    def f_multi(self, obss: npt.NDArray, exps: npt.NDArray) -> npt.NDArray:
        """Apply the cost function to multiple inputs.

Parameters:

    obss - the observed output vectors

    exps - the expected output vectors

Returns:

    costs - a vector of the calculated costs of the observation
"""
        assert obss.ndim == exps.ndim == 2, "Inputs must be two-dimensional"
        assert obss.shape[0] == exps.shape[0], "Inputs must have the same number of values"

        return self.__f_multi(obss, exps)

    def grad_f(self, obs: npt.NDArray, exp: npt.NDArray) -> npt.NDArray:
        """Apply the derivative function of the cost function w.r.t. the observed output vector (i.e. ∂ cost / ∂ observed_value).

Parameters:

    obs - the observed output vector

    exp - the expected output vector

Returns:

    grad - the calculated gradient of the cost of the observation with respect to the observation
"""
        assert obs.ndim == exp.ndim == 1, "Inputs must be one-dimensional"

        return self.grad_f_multi(
            obs[np.newaxis,:],
            exp[np.newaxis,:],
        )[0]

    def grad_f_multi(self, obss: npt.NDArray, exps: npt.NDArray) -> npt.NDArray:
        """Apply the derivative function of the cost function w.r.t. the observed output vector (i.e. ∂ cost / ∂ observed_value) to apply to multiple inputs.

Parameters:

    observed - the observed output vectors

    expected - the expected output vectors

Returns:

    grads - the calculated gradients of the cost of the observation with respect to the observations
"""
        assert obss.ndim == exps.ndim == 2, "Inputs must be two-dimensional"
        assert obss.shape[0] == exps.shape[0], "Inputs must have same number of values"

        return self.__grad_f_multi(obss, exps)


relu = DiffVectorVectorFunction(
    lambda x: np.maximum(np.zeros_like(x), x),
    lambda x: (x > 0).astype(x.dtype)
)


sigmoid = DiffVectorVectorFunction(
    lambda x: 1 / (1 + np.exp(-x)),  # σ = 1 / (1 + e^-x)
    lambda x: np.exp(-x) / np.square(1 + np.exp(-x))  # dσ/dx = e^-x / (1 + e^-x)^2
)


sum_of_squares_cost = NNCostFunction(
    f_multi=lambda obss, exps: np.sum(np.square(obss - exps), axis=1),
    grad_f_multi=lambda obss, exps: 2 * (obss - exps),
)
