from typing import Iterable, Iterator, List, Optional
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

    def full_forwards(self, x: npt.NDArray) -> Iterator[npt.NDArray]:
        """Performs a single iteration of forwards propagation through the whole network and returns the outputs of each layer.

Parameters:

    x - the input vector to provide to the network

Returns:

    ys - an iterator of the output vectors from each layer
"""

        curr = x
        yield curr.copy()

        for layer in self._layers:
            curr = layer.forwards(curr)
            yield curr.copy()

    def forwards(self, x: npt.NDArray) -> npt.NDArray:
        """Performs a single iteration of forwards propagation and returns the output of the network.

Parameters:

    x - the input vector to provide to the network

Returns:

    y - the output vector from the forwards propagation
"""

        if not self.has_layers:
            raise EmptyNetworkException()
        else:
            *_, res = self.full_forwards(x)
            return res

    def calculate_cost(self, inp: npt.NDArray, exp: npt.NDArray, cost_func: NNCostFunction) -> float:
        """Calculate the cost for the network given some input and some expected output values"""
        return cost_func.f(self.forwards(inp), exp)

    def __full_backwards(self, xs: List[npt.NDArray], grad_wrt_y: npt.NDArray) -> Iterator[npt.NDArray]:
        """Performs a single iteration of backwards propagation through the whole network \
adjusting the layers' parameters' values along the way \
and returns the gradients of the cost function w.r.t. the outputs of each layer.

Parameters:

    xs - the vectors at each stage of a forwards propagation. `xs[0]` should be the original inputs and `xs[-1]` should be the final output.

    grad_wrt_y - the gradients of the cost function w.r.t. the values of each output of the last layer of the network

Returns:

    grads_wrt_xs_rev - a reversed iterator of the gradients of the cost function w.r.t. each of the vectors at each stage of the network. \
The first value repeats the grad_wrt_y input. \
The final value gives the gradients of the cost function w.r.t. the inputs to the network.
"""
        assert len(xs) == len(self.layers) + 1, f"Incorrect number of xs elements. Should have one more than the number of layers. Expected {len(self.layers)} elements, got {len(xs)} elements"
        assert grad_wrt_y.shape == (self.output_n,), f"Invalid grad_wrt_y shape. Expected {self.output_n} values but got shape of {grad_wrt_y.shape}"
        assert xs[-1].shape == (self.output_n,), f"Invalid final xs element shape. Expected {self.output_n} values but got shape of {xs[-1].shape}"

        if not self.has_layers:
            raise EmptyNetworkException()

        curr_grad = grad_wrt_y
        yield curr_grad.copy()

        for layer, x in zip(
            reversed(self._layers),
            reversed(xs[:-1]),  # Don't include last value as it is the output of the entire network
        ):

            curr_grad = layer.backwards(x, curr_grad)
            yield curr_grad.copy()

    def learn_single(self, x: npt.NDArray, exp: npt.NDArray, cost_func: NNCostFunction) -> npt.NDArray:
        """Performs a single iteration of forwards propagation \
and then a single iteration of backwards propagation \
using the results to reduce the value of the cost function

Parameters:

    x - the input vector to provide for the forwards propagation

    exp - the expected output vector for the given input vector

    cost_func - the cost function to use

Returns:

    grad_wrt_x - the calculated rate of change of the cost function w.r.t each input value (i.e. ∂ cost / ∂ x)
"""

        if not self.has_layers:
            raise EmptyNetworkException()

        assert x.shape == (self.layers[0].input_n,), f"Invalid x shape. Network takes {self.layers[0].input_n} input values but shape of x was {x.shape}"
        assert exp.shape == (self.layers[-1].output_n,), f"Invalid exp shape. Network outputs {self.layers[-1].output_n} values but shape of exp was {exp.shape}"

        # Forward propagation

        xs: List[npt.NDArray] = list(self.full_forwards(x))

        # Cost calculations

        grad_wrt_y: npt.NDArray = cost_func.grad_f(xs[-1], exp)

        # Backwards propagation

        *_, grad_wrt_x = self.__full_backwards(xs, grad_wrt_y)

        return grad_wrt_x
