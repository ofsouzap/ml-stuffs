from typing import Optional, List
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from math_util.vector_functions import DiffVectorVectorFunction, relu, sigmoid


_DEFAULT_DTYPE: npt.DTypeLike = np.float64


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
    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        return str(self)

    def forwards_single(self, x: npt.NDArray) -> npt.NDArray:
        """Performs forwards-propagation on a single input"""
        return self.forwards_multi(x[np.newaxis,:])[0]

    @abstractmethod
    def forwards_multi(self, xs: npt.NDArray) -> npt.NDArray:
        """Performs forwards-propagation on multiple inputs"""
        pass

    def backwards_single(self, x: npt.NDArray, grad_wrt_y: npt.NDArray) -> npt.NDArray:
        """Performs backwards-propagation but only on a single input"""
        return self.backwards_multi(x[np.newaxis,:], grad_wrt_y[np.newaxis,:])[0]

    @abstractmethod
    def backwards_multi(self, xs: npt.NDArray, grads_wrt_ys: npt.NDArray) -> npt.NDArray:
        """Performs backwards-propagation on multiple inputs, altering the parameters of the layer according to the learning rate.

Parameters:

    xs - the input vectors that were passed to the layer

    grads_wrt_ys - the gradient vectors of the cost function with respect to the outputs of the function.

Returns:

    grad_wrt_xs - the gradient vectors of the cost function with respect to the inputs of the function.
"""
        pass


class ActivationLayer(LayerBase):

    def __init__(self, n: int, learning_rate: float, func: DiffVectorVectorFunction):
        super().__init__(n, n, learning_rate)
        self._n = n
        self._func = func

    @property
    def n(self) -> int:
        return self._n

    @property
    def func(self) -> DiffVectorVectorFunction:
        return self._func

    def forwards_multi(self, xs: npt.NDArray) -> npt.NDArray:
        assert xs.ndim == 2, "Input must be two-dimensional"
        return self.func.f_multi(xs)

    def backwards_multi(self, xs: npt.NDArray, grads_wrt_ys: npt.NDArray) -> npt.NDArray:
        assert xs.ndim == grads_wrt_ys.ndim == 2, "Inputs must be two-dimensional"
        assert xs.shape[0] == grads_wrt_ys.shape[0], "Inputs must have same number of values"
        return grads_wrt_ys * self.func.grad_f_multi(xs)


class ReluActivationLayer(ActivationLayer):
    def __init__(self, n: int, learning_rate: float):
        super().__init__(n, learning_rate, relu)
    def __str__(self) -> str:
        return f"ReLU ({self.n})"


class SigmoidActivationLayer(ActivationLayer):
    def __init__(self, n: int, learning_rate: float):
        super().__init__(n, learning_rate, sigmoid)
    def __str__(self) -> str:
        return f"σ ({self.n})"


class DenseLayer(LayerBase):

    def __init__(self, n: int, m: int, learning_rate: float, weights: Optional[npt.NDArray] = None, bias: Optional[npt.NDArray] = None):
        super().__init__(n, m, learning_rate)

        dtype: npt.DTypeLike

        if weights is not None:
            dtype = weights.dtype
        elif bias is not None:
            dtype = bias.dtype
        else:
            dtype = _DEFAULT_DTYPE

        self._weights: npt.NDArray
        self._bias: npt.NDArray

        if weights is not None:
            assert weights.ndim == 2, "Weights must be two-dimensional array"
            assert weights.shape[0] == self.input_n, "Weights has incorrect number of input values"
            assert weights.shape[1] == self.output_n, "Weights has incorrect number of output values"
            self._weights = weights
        else:
            self._weights = np.ones(shape=(self.input_n,self.output_n), dtype=dtype)

        if bias is not None:
            assert bias.ndim == 1, "Bias must be one-dimensional array"
            assert bias.shape[0] == self.output_n, "Bias has incorrect number of values"
            self._bias = bias
        else:
            self._bias = np.zeros(shape=(self.output_n,), dtype=dtype)

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

    def __str__(self) -> str:
        output_form_strs: List[str] = []
        for j in range(self.output_n):
            output_form_strs.append(
                " + ".join(
                    [f"{self.weights[i,j]}*x_{i}" for i in range(self.input_n)]
                ) + \
                f" + {self.bias[j]}"
            )
        return " & ".join([f"({s})" for s in output_form_strs])

    def forwards_multi(self, xs: npt.NDArray) -> npt.NDArray:
        assert xs.ndim == 2, "Input must be two-dimensional"

        mat_xs = xs[:,np.newaxis,:]
        mat_result: npt.NDArray = np.matmul(mat_xs, self.weights) + self.bias
        result = mat_result[:,0,:]  # The results of the matrix multiplication was an array of row vectors, this just takes each row vector as a 1D array

        assert result.ndim == 2
        assert result.shape[0] == xs.shape[0]

        return result

    def backwards_multi(self, xs: npt.NDArray, grads_wrt_ys: npt.NDArray) -> npt.NDArray:
        assert xs.ndim == grads_wrt_ys.ndim == 2, "Inputs must be two-dimensional"
        assert xs.shape[0] == grads_wrt_ys.shape[0], "Inputs must have same number of values"
        assert xs.shape[1] == self.input_n, "xs should have same number of values as layer has inputs"
        assert grads_wrt_ys.shape[1] == self.output_n, "grads_wrt_ys should have same number of values as layer has outputs"

        # Calculate gradients w.r.t. inputs (i.e. ∂ cost / ∂ x) before changing parameter values

        grads_wrt_xs: npt.NDArray = np.sum(grads_wrt_ys[:,np.newaxis,:] * self.weights[np.newaxis,:,:], axis=2)
        """∂ cost_i / ∂ x_j = Σ_k ( ∂ cost_i / ∂ y_k ) * W_jk"""

        # Calculate gradients w.r.t. parameters

        grads_wrt_weights = xs[:,:,np.newaxis] * grads_wrt_ys[:,np.newaxis,:]
        """∂ cost_i / ∂ W_jk = x_j * ∂ cost_i / ∂ y_k"""

        grads_wrt_biases = grads_wrt_ys.copy()
        """∂ cost_i / ∂ b_j = ∂ cost_i / ∂ y_j"""

        # Apply gradient descent

        self._add_weights_multi(-1 * self.learning_rate * grads_wrt_weights)
        self._add_biases_multi(-1 * self.learning_rate * grads_wrt_biases)

        # Check output

        assert grads_wrt_xs.ndim == 2
        assert grads_wrt_xs.shape[0] == xs.shape[0]
        assert grads_wrt_xs.shape[1] == xs.shape[1]

        # Return output

        return grads_wrt_xs

    def set_weights(self, weights: npt.NDArray) -> None:
        assert weights.shape == (self.input_n,self.output_n), "Invalid weight matrix shape"
        self._weights = weights

    def _add_weights_multi(self, add_weights: npt.NDArray) -> None:
        """Adds the specified amounts to the weights"""
        assert add_weights.ndim == 3, "Input must be three-dimensional"
        assert add_weights.shape[1] == self.weights.shape[0], "Incorrect add weights shape"
        assert add_weights.shape[2] == self.weights.shape[1], "Incorrect add weights shape"
        self._weights += np.sum(add_weights, axis=0)

    def set_bias(self, bias: npt.NDArray) -> None:
        assert bias.shape == (self.output_n,), "Invalid bias array shape"
        self._bias = bias

    def _add_biases_multi(self, add_biases: npt.NDArray) -> None:
        """Adds the specified amounts to the biases"""
        assert add_biases.ndim == 2, "Input must be two-dimensional"
        assert add_biases.shape[1] == self.bias.shape[0], "Incorrect number of bias values"
        self._bias += np.sum(add_biases, axis=0)


class PolynomialLayer(LayerBase):

    def __init__(self,
                 n: int,
                 m: int,
                 learning_rate: float,
                 order_weights: Optional[npt.NDArray] = None,
                 bias: Optional[npt.NDArray] = None):
        super().__init__(n, m, learning_rate)

        # Determine dtype

        dtype: npt.DTypeLike

        if bias is not None:
            dtype = bias.dtype
        elif (order_weights is not None) and (len(order_weights) >= 1):
            dtype = order_weights[0].dtype
        else:
            dtype = np.float64

        # Use parameter order weights and biases or use default

        self._order_weights: npt.NDArray
        """The weights for each order of term except constant terms. Therefore the weight matrix for x^n is at _order_weights[n-1]"""
        self._bias: npt.NDArray

        if order_weights is not None:
            assert order_weights.ndim == 3, "Order weights must be three-dimensional array"
            assert order_weights.shape[1] == self.input_n, "Order weights matrices have incorrect number of input values"
            assert order_weights.shape[2] == self.output_n, "Order weights matrices have incorrect number of output values"
            self._order_weights = order_weights
        else:
            self._order_weights = np.ones(shape=(1,n,m), dtype=dtype)

        if bias is not None:
            assert bias.ndim == 1, "Bias must be one-dimensional array"
            assert bias.shape[0] == self.output_n, "Bias has incorrect number of values"
            self._bias = bias
        else:
            self._bias = np.zeros(shape=(self.output_n,), dtype=dtype)

    @property
    def order(self) -> int:
        return self._order_weights.shape[0]

    @property
    def order_weights(self) -> npt.NDArray:
        """Weights matrix for each order of term"""
        return self._order_weights

    def get_order_weight(self, order: int) -> npt.NDArray:
        """Gets a copy of the weight matrix for a certain order of term"""
        assert 0 < order <= self.order, "Invalid order"
        return self._order_weights[order-1,:,:].copy()

    @property
    def bias(self) -> npt.NDArray:
        return self._bias

    def __str__(self) -> str:
        # TODO - proper implementation. Use DenseLayer.__str__ as reference
        return "Polynomial Layer"

    def forwards_multi(self, xs: npt.NDArray) -> npt.NDArray:

        assert xs.ndim == 2, "Input must be two-dimensional"

        mat_xs = xs[:,np.newaxis,:]
        """The value at [i,0,j] gives the i'th input vector's j'th component"""

        mat_result: npt.NDArray = np.zeros(shape=(xs.shape[0],1,self.output_n), dtype=xs.dtype)

        # Add polynomial terms

        mat_xs_tiled = np.swapaxes(
            np.tile(mat_xs, (self.order,1,1,1)),
            0,
            1
        )
        """4D array with the input vectors repeated. The value at [i,j,0,k] gives, for all valid j, the i'th input vector's k'th component"""

        pows = np.arange(1, self.order+1, dtype=xs.dtype)
        """1D array of powers to raise inputs to"""

        mat_xs_powed = np.power(mat_xs_tiled, pows[np.newaxis,:,np.newaxis,np.newaxis])
        """4D array with the input vectors with their values to the appropriate powers. \
The value at [i,j,0,k] gives the i'th input vector's k'th component to the power of (j+1)"""

        pows_result = np.matmul(mat_xs_powed, self.order_weights)
        """4D array with outputs from matrix multiplications on input vectors to different powers. \
The row vector at [i,j,0,:] is the output value for the i'th input vector using the j'th power using the weight matrix for that power"""

        pows_result_summed = np.sum(pows_result, axis=1)
        """3D array with the output row vectors for each input vector (without the bias). \
The value at [i,0,j] gives the j'th component of the output vector for the i'th input vector"""

        mat_result += pows_result_summed

        # Add constant terms (bias values)

        mat_result += self.bias[np.newaxis,np.newaxis,:]

        # Return final result

        result = mat_result[:,0,:]  # The results of the matrix multiplication was an array of row vectors, this just takes each row vector as a 1D array

        assert result.ndim == 2
        assert result.shape[0] == xs.shape[0]

        return result

    def backwards_multi(self, xs: npt.NDArray, grads_wrt_ys: npt.NDArray) -> npt.NDArray:
        assert xs.ndim == grads_wrt_ys.ndim == 2, "Inputs must be two-dimensional"
        assert xs.shape[0] == grads_wrt_ys.shape[0], "Inputs must have same number of values"
        assert xs.shape[1] == self.input_n, "xs should have same number of values as layer has inputs"
        assert grads_wrt_ys.shape[1] == self.output_n, "grads_wrt_ys should have same number of values as layer has outputs"

        # Compute useful versions of inputs

        xs_tiled = np.swapaxes(
            np.tile(xs, (self.order,1,1)),
            0,
            1
        )
        """3D array with the input vectors repeated. The value at [i,j,k] gives, for all valid j, the i'th input vector's k'th component"""

        pows = np.arange(1, self.order+1, dtype=xs.dtype)
        """1D array of powers to raise inputs to"""

        xs_powed = np.power(xs_tiled, pows[np.newaxis,:,np.newaxis])
        """3D array with the input vectors with their values to the appropriate powers. \
The value at [i,j,k] gives the i'th input vector's k'th component to the power of (j+1)"""

        xs_powed_one_less = np.empty_like(xs_powed)
        """3D array with the input vectors with their values to one less than their appropriate powers. \
The value at [i,j,k] gives the i'th input vector's k'th component to the power of j"""

        xs_powed_one_less[:,0,:] = 1  # For the power of 0

        if self.order > 1:
            xs_powed_one_less[:,1:,:] = xs_powed[:,:-1,:]  # For the positive powers

        # Calculate gradients w.r.t. inputs (i.e. ∂ cost / ∂ x) before changing parameter values

        grads_wrt_xs: npt.NDArray = np.sum(
            grads_wrt_ys[:,np.newaxis,:] * np.sum(
                pows[np.newaxis,np.newaxis,np.newaxis,:] * xs_powed_one_less[:,:,np.newaxis,:] * np.transpose(
                    self.order_weights,
                    axes=(1,2,0)  # TODO - check if this transposing is really what I mean
                )[np.newaxis,:,:,:],
                axis=3
            ),
            axis=2
        )
        """∂ cost_i / ∂ x_j = Σ_k ( ∂ cost_i / ∂ y_k  * Σ_l ( l * (x_j)^(l-1) * W_ljk ) )"""

        # Calculate gradients w.r.t. parameters

        grads_wrt_order_weights = grads_wrt_ys[:,np.newaxis,np.newaxis,:] * xs_powed[:,:,:,np.newaxis]
        """∂ cost_i / ∂ W_jkl = ∂ cost_i / ∂ y_l * (x_k)^j"""

        assert grads_wrt_order_weights.ndim == 4

        grads_wrt_biases = grads_wrt_ys.copy()
        """∂ cost_i / ∂ b_j = ∂ cost_i / ∂ y_j"""

        assert grads_wrt_biases.ndim == 2

        # Apply gradient descent

        self._add_order_weights_multi(-1 * self.learning_rate * grads_wrt_order_weights)
        self._add_biases_multi(-1 * self.learning_rate * grads_wrt_biases)

        # Check output

        assert grads_wrt_xs.ndim == 2
        assert grads_wrt_xs.shape[0] == xs.shape[0]
        assert grads_wrt_xs.shape[1] == xs.shape[1]

        # Return output

        return grads_wrt_xs

    def set_order_weights(self, order_weights: npt.NDArray) -> None:
        assert order_weights.shape == self.order_weights.shape, "Invalid order weight array shape"
        self._order_weights = order_weights

    def _add_order_weights_multi(self, add_order_weights: npt.NDArray) -> None:
        """Adds the specified amounts to the order weights"""
        assert add_order_weights.ndim == 4, "Input must be four-dimensional"
        assert add_order_weights.shape[1:] == self.order_weights.shape, "Incorrect add order weights shape"
        self._order_weights += np.sum(add_order_weights, axis=0)

    def set_bias(self, bias: npt.NDArray) -> None:
        assert bias.shape == (self.output_n,), "Invalid bias array shape"
        self._bias = bias

    def _add_biases_multi(self, add_biases: npt.NDArray) -> None:
        """Adds the specified amounts to the biases"""
        assert add_biases.ndim == 2, "Input must be two-dimensional"
        assert add_biases.shape[1] == self.bias.shape[0], "Incorrect number of bias values"
        self._bias += np.sum(add_biases, axis=0)
