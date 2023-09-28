from typing import Iterable, Iterator, List, Optional
import numpy as np
import numpy.typing as npt
from math_util.vector_functions import NNCostFunction
from .layers import LayerBase


class EmptyNetworkException(Exception):
    """Exception thrown when trying to access a layer of a network without any layers"""
    pass


class IncompatibleLayerException(Exception):
    """Exception thrown when trying to modify a network with a layer that doesn't fit with the existing layers"""
    pass


class Network:

    def __init__(self,
                 layers: Optional[Iterable[LayerBase]] = None):

        # Initialise

        self._layers: List[LayerBase] = []

        # Use optional parameters

        if layers is not None:
            self.set_layers(layers)

    @property
    def has_layers(self) -> bool:
        return len(self._layers) > 0

    @property
    def layers(self) -> List[LayerBase]:
        return self._layers

    @property
    def input_n(self) -> int:
        if self._layers:
            return self._layers[0].input_n
        else:
            raise EmptyNetworkException()

    @property
    def output_n(self) -> int:
        if self._layers:
            return self._layers[-1].output_n
        else:
            raise EmptyNetworkException()

    def __str__(self) -> str:
        return " -> ".join(["{"+str(layer)+"}" for layer in self.layers])

    def __repr__(self) -> str:
        return str(self)

    def __layer_compatible_for_start(self, layer: LayerBase) -> bool:
        return (not self.has_layers) or (self.input_n == layer.output_n)

    def __layer_compatible_for_end(self, layer: LayerBase) -> bool:
        return (not self.has_layers) or (self.output_n == layer.input_n)

    def add_layer_to_start(self, layer: LayerBase) -> None:
        if self.__layer_compatible_for_start(layer):
            self._layers.insert(0, layer)
        else:
            raise IncompatibleLayerException()

    def add_layer_to_end(self, layer: LayerBase) -> None:
        if self.__layer_compatible_for_end(layer):
            self._layers.append(layer)
        else:
            raise IncompatibleLayerException()

    def set_layers(self, layers: Iterable[LayerBase]) -> None:
        self._layers.clear()
        for layer in layers:
            self.add_layer_to_end(layer)

    def full_forwards_single(self, x: npt.NDArray) -> Iterator[npt.NDArray]:
        """Performs a single iteration of forwards propagation for a single input vector through the whole network and returns the outputs of each layer.

Parameters:

    x - the input vector to provide to the network

Returns:

    ys - an iterator of the output vectors from each layer
"""
        for val in self.full_forwards_multi(x[np.newaxis,:]):
            yield val[0]

    def forwards_single(self, x: npt.NDArray) -> npt.NDArray:
        """Performs a single iteration of forwards propagation for a single input vector and returns the output of the network.

Parameters:

    x - the input vector to provide to the network

Returns:

    y - the output vector from the forwards propagation
"""
        return self.forwards_multi(x[np.newaxis,:])[0]

    def full_forwards_multi(self, xs: npt.NDArray) -> Iterator[npt.NDArray]:
        """Performs a single iteration of forwards propagation for multiple input vectors through the whole network and returns the outputs of each layer.

Parameters:

    xs - the input vectors to provide to the network

Returns:

    seqs - an iterator of the output vectors from each layer for each input vector
"""

        currs = xs
        yield currs.copy()

        for layer in self._layers:
            currs = layer.forwards_multi(currs)
            yield currs.copy()

    def forwards_multi(self, xs: npt.NDArray) -> npt.NDArray:
        """Performs a single iteration of forwards propagation for multiple input vectors and returns the output of the network.

Parameters:

    xs - the input vectors to provide to the network

Returns:

    ys - the output vectors from the forwards propagation
"""

        if not self.has_layers:
            raise EmptyNetworkException()
        else:
            *_, res = self.full_forwards_multi(xs)
            return res

    def calculate_cost_single(self, inp: npt.NDArray, exp: npt.NDArray, cost_func: NNCostFunction) -> float:
        """Calculate the cost for the network given a single input vector and a single expected output vector"""
        return self.calculate_cost_multi(inp[np.newaxis,:], exp[np.newaxis,:], cost_func)[0]

    def calculate_cost_multi(self, inps: npt.NDArray, exps: npt.NDArray, cost_func: NNCostFunction) -> npt.NDArray:
        """Calculate the cost for the network given multiple input vectors and multiple expected output vectors"""
        assert inps.ndim == exps.ndim == 2, "Input value array and expected value array must be two-dimensional"
        assert inps.shape[0] == exps.shape[0], "Input and expected value arrays must have same number of vectors"

        return cost_func.f_multi(self.forwards_multi(inps), exps)

    def __full_backwards_multi(self, seqs: List[npt.NDArray], grads_wrt_ys: npt.NDArray) -> Iterator[npt.NDArray]:
        """Performs a single iteration of backwards propagation through the whole network from the results for multiple input vectors \
adjusting the layers' parameters' values along the way \
and returns the gradients of the cost function w.r.t. the outputs of each layer.

Parameters:

    seqs - the vectors at each stage of a forwards propagation. `seq[:,0]` should be the original input vectors and `seq[:,-1]` should be the final output vectors.

    grads_wrt_ys - the gradients of the cost function w.r.t. the values of each outputs of the last layer of the network

Returns:

    grads_wrt_xs_revs - a reversed iterator of the gradients of the cost function w.r.t. each of the vectors at each stage of the network. \
The first value repeats the grads_wrt_ys input. \
The final value gives the gradients of the cost function w.r.t. the input vectors to the network.
"""
        assert len(seqs) == len(self.layers) + 1, f"Incorrect number of seqs elements. Should have one more than the number of layers. Expected {len(self.layers)} elements, got {len(seqs)} elements"
        assert grads_wrt_ys.ndim == 2, "Gradients array must be two-dimensional"
        assert grads_wrt_ys.shape[1] == self.output_n, "Vectors in gradients array must have same number of elements as output of this network"
        assert seqs[-1].shape[1] == self.output_n, "Final seqs vectors must have same number of elements as output of this network"
        assert all(map(lambda x: x.shape[0] == grads_wrt_ys.shape[0], seqs)), "All seqs elements must have same number of vectors as gradients array"

        if not self.has_layers:
            raise EmptyNetworkException()

        curr_grads = grads_wrt_ys
        yield curr_grads.copy()

        for layer, xs in zip(
            reversed(self._layers),
            reversed(seqs[:-1]),  # Don't include last value as it is the output of the entire network
        ):

            curr_grads = layer.backwards_multi(xs, curr_grads)
            yield curr_grads.copy()

    def learn_step_single(self, x: npt.NDArray, exp: npt.NDArray, cost_func: NNCostFunction) -> npt.NDArray:
        """Performs a single iteration of forwards propagation using a single sample \
and then a single iteration of backwards propagation \
using the results to reduce the value of the cost function

Parameters:

    x - the input vector to provide for the forwards propagation

    exp - the expected output vector for the given input vector

    cost_func - the cost function to use

Returns:

    grad_wrt_x - the calculated rate of change of the cost function w.r.t each input value (i.e. ∂ cost / ∂ x)
"""

        return self.learn_step_multi(
            x[np.newaxis,:],
            exp[np.newaxis,:],
            cost_func
        )[0]

    def learn_step_multi(self, xs: npt.NDArray, exps: npt.NDArray, cost_func: NNCostFunction) -> npt.NDArray:
        """Performs a single iteration of forwards propagation using multiple sample input vectors \
and then a single iteration of backwards propagation \
using the results to reduce the value of the cost function

Parameters:

    xs - the input vectors to provide for the forwards propagation

    exps - the expected output vectors for the given input vectors

    cost_func - the cost function to use

Returns:

    grads_wrt_xs - the calculated rate of change of the cost function w.r.t each input value (i.e. ∂ cost / ∂ x)
"""

        if not self.has_layers:
            raise EmptyNetworkException()

        assert xs.ndim == exps.ndim == 2, "Input values arrays and expected output values array must be two-dimensional"
        assert xs.shape[0] == exps.shape[0], "Input and expected output values arrays must have same number of vectors"
        assert xs.shape[1] == self.input_n, "Input vectors must have same number of elements as this network has inputs"
        assert exps.shape[1] == self.output_n, "Expected output vectors must have same number of elements as this network has outputs"

        # Forward propagation

        seqs: List[npt.NDArray] = list(self.full_forwards_multi(xs))

        # Cost calculations

        grads_wrt_ys: npt.NDArray = cost_func.grad_f_multi(seqs[-1], exps)

        # Backwards propagation

        *_, grad_wrt_x = self.__full_backwards_multi(seqs, grads_wrt_ys)

        return grad_wrt_x
