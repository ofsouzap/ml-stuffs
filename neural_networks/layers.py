from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from math_util.vector_functions import DifferentiableVectorFunction, relu, sigmoid


class LayerBase(ABC):

    def __init__(self,
                 input_n: int,
                 output_n: int,
                 learning_rate: float):
        self.__input_n = input_n
        self.__output_n = output_n
        self.__learning_rate = learning_rate

    @property
    def input_n(self) -> int:
        """Number of input variables to the layer"""
        return self.__input_n

    @property
    def output_n(self) -> int:
        """Number of output variables to the layer"""
        return self.__output_n

    @property
    def learning_rate(self) -> float:
        return self.__learning_rate

    @abstractmethod
    def forwards(self, x: npt.NDArray) -> npt.NDArray:
        """Performs forwards-propagation"""
        pass

    @abstractmethod
    def backwards(self, x: npt.NDArray, grad_wrt_y: npt.NDArray) -> npt.NDArray:
        """Performs backwards-propagation, altering the parameters of the layer according to the learning rate.

Parameters:

    x - the inputs that were passed to the layer

    grad_wrt_y - the gradient vector of the cost function with respect to the outputs of the function.

Returns:

    grad_wrt_x - the gradient vector of the cost function with respect to the inputs of the function.
"""
        pass


class ActivationLayer(LayerBase):

    def __init__(self, n: int, learning_rate: float, func: DifferentiableVectorFunction):
        super().__init__(n, n, learning_rate)
        self._n = n
        self._func = func

    @property
    def n(self) -> int:
        return self._n

    @property
    def func(self) -> DifferentiableVectorFunction:
        return self._func

    def forwards(self, x: npt.NDArray) -> npt.NDArray:
        return self.func.f(x)

    def backwards(self, x: npt.NDArray, grad_wrt_y: npt.NDArray) -> npt.NDArray:
        return np.multiply(grad_wrt_y, self.func.grad_f(x))


class ReluActivationLayer(ActivationLayer):
    def __init__(self, n: int, learning_rate: float):
        super().__init__(n, learning_rate, relu)


class SigmoidActivationLayer(ActivationLayer):
    def __init__(self, n: int, learning_rate: float):
        super().__init__(n, learning_rate, sigmoid)


class DenseLayer(LayerBase):

    def __init__(self, n: int, m: int, learning_rate: float, weights: Optional[npt.NDArray] = None, bias: Optional[npt.NDArray] = None):
        super().__init__(n, m, learning_rate)

        self._weights: npt.NDArray
        self._bias: npt.NDArray

        if weights is not None:
            assert weights.ndim == 2, "Weights must be two-dimensional array"
            assert weights.shape[0] == self.input_n, "Weights has incorrect number of input values"
            assert weights.shape[1] == self.output_n, "Weights has incorrect number of output values"
            self._weights = weights
        else:
            self._weights = np.ones(shape=(self.input_n,self.output_n))

        if bias is not None:
            assert bias.ndim == 1, "Bias must be one-dimensional array"
            assert bias.shape[0] == self.output_n, "Bias has incorrect number of values"
            self._bias = bias
        else:
            self._bias = np.zeros(shape=(self.output_n,))

    @property
    def n(self) -> int:
        return self.output_n

    @property
    def weights(self) -> npt.NDArray:
        """Weight matrix for the layer"""
        return self._weights

    @property
    def bias(self) -> npt.NDArray:
        """Bias values for the layer"""
        return self._bias

    def forwards(self, x: npt.NDArray) -> npt.NDArray:
        assert x.shape == (self.input_n,), "Input shape invalid"
        return (x @ self.weights) + self.bias

    def backwards(self, x: npt.NDArray, grad_wrt_y: npt.NDArray) -> npt.NDArray:

        assert x.shape == (self.input_n,), "Invalid x shape"
        assert grad_wrt_y.shape == (self.output_n,), "Invalid grad_wrt_y shape"

        # Calculate gradients w.r.t. parameters

        grad_wrt_weights = x[:,np.newaxis] * grad_wrt_y[np.newaxis,:]
        """∂ cost / ∂ W_ij = x_i * ∂ cost / ∂ y_j"""

        grad_wrt_biases = grad_wrt_y.copy()
        """∂ cost / ∂ b_i = ∂ cost / ∂ y_i"""

        # Apply gradient descent

        self.add_weights(-1 * self.learning_rate * grad_wrt_weights)
        self.add_biases(-1 * self.learning_rate * grad_wrt_biases)

        # Calculate gradients w.r.t. inputs (i.e. ∂ cost / ∂ x)

        grad_wrt_x = np.sum(grad_wrt_weights, axis=1)
        """∂ cost / ∂ x_i = Σ_j ( ∂ cost / ∂ W_ij )"""

        return grad_wrt_x

    def set_weights(self, weights: npt.NDArray) -> None:
        assert weights.shape == (self.input_n,self.output_n), "Invalid weight matrix shape"
        self._weights = weights

    def add_weights(self, add_weights: npt.NDArray) -> None:
        """Adds the specified amount to the weights"""
        assert add_weights.shape == self.weights.shape, "Invalid add_weights shape"
        self._weights += add_weights

    def set_bias(self, bias: npt.NDArray) -> None:
        assert bias.shape == (self.output_n,), "Invalid bias array shape"
        self._bias = bias

    def add_biases(self, add_biases: npt.NDArray) -> None:
        """Adds the specified amount to the biases"""
        assert add_biases.shape == self.bias.shape, "Invalid add_biases shape"
        self._bias += add_biases
