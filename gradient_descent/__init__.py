from typing import Callable, Optional, List, SupportsFloat
import numpy as np
import numpy.typing as npt
from check import check_is_vec, check_vec_dim, check_is_mat, check_mat_dim


_DEFAULT_EPS = 1e-06


# TODO - vectorize this stuff. Ie. have the functions work with multiple positions at a time stored in numpy arrays to make more efficient


def estimate_grads(
    vec_field_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    poss: npt.NDArray[np.float64],
    eps: Optional[SupportsFloat] = None) -> npt.NDArray[np.float64]:
    """Approximates the gradient vector of a scalar field at some positions. At singularities, a grad value of 0 is used

Parameters:

    vec_field_func - a function that takes a 2D array of position vectors in the field and returns the field's values at those positions

    poss - the positions at which to evaluate the gradient
"""

    if poss.ndim != 2:
        raise ValueError("Input positions must be a 2D array")

    if poss.shape[0] == 0:
        return np.zeros_like(poss)

    if eps is None:
        eps = _DEFAULT_EPS
    else:
        eps = float(eps)

    grads = np.empty_like(poss)

    for i in range(grads.shape[1]):

        # Create the epsilon vector (the small amount to move in each direction)

        eps_vec = np.zeros(shape=poss.shape[1])
        eps_vec[i] += eps

        # Calculate the gradients on either side

        right_grads = (vec_field_func(poss + eps_vec) - vec_field_func(poss)) / eps
        left_grads = (vec_field_func(poss) - vec_field_func(poss - eps_vec)) / eps

        paired_grads = np.stack([right_grads, left_grads]).T

        avg_grads: npt.NDArray[np.float64] = np.average(paired_grads, axis=1)

        grads[:, i] = np.where(
            np.isinf(vec_field_func(poss)),
            np.zeros(shape=(avg_grads.shape[0],)),
            avg_grads
        )

    check_is_mat(grads)

    return grads


def find_any_extremum(
    vec_field_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    start: npt.NDArray[np.float64],
    learning_rate: SupportsFloat = 1e-03,
    maximum_iterations: Optional[int] = None,
    gradient_estimation_eps: Optional[SupportsFloat] = None,
    stop_gradient: SupportsFloat = 1e-08) -> npt.NDArray[np.float64]:
    """Performs gradient to find any single extreme value of a N-D field.

Parameters:

    vec_field_func - a function taking M N-dimensional vectors as input and producing a M real outputs corresponding to each of the input vectors. \
This is the function that the extreme value is trying to be found for.

    start - the starting input to use.

    learning_rate - the factor to change input values by when searching for an extremum.

    maximum_iterations (optional) - if provided, determines the maximum number of iterations to run before stopping the calculation.

    gradient_estimation_eps (optional) - how far on either side of a point to look when estimating the gradient.

    stop_gradient - how stationary a point must seem to be before stopping the algorithm.

Returns:

    stationary_point - the input values for the stationary point discovered.
"""

    # Input checking

    if start.ndim != 1:
        raise ValueError("Start position must be 1-dimensional")

    dims = start.shape[0]

    learning_rate = float(learning_rate)

    if learning_rate == 0:
        raise ValueError("Learning rate must be non-zero")

    if (maximum_iterations is not None) and (maximum_iterations < 1):
        raise ValueError("Maximum number of iterations must be positive or None")

    stop_gradient = float(stop_gradient)

    if stop_gradient <= 0:
        raise ValueError("Stop gradient must be positive")

    # Main learning loop

    curr_pos = start.copy()

    iteration_idx = 0
    while (maximum_iterations is None) or (iteration_idx < maximum_iterations):

        # Calculate gradient

        grad = estimate_grads(
            vec_field_func=vec_field_func,
            poss=curr_pos[np.newaxis,:],
            eps=gradient_estimation_eps
        )[0]

        check_is_vec(grad, "Gradient isn't returned as a vector")

        grad_mag = np.sum(np.square(grad))
        grad_dir = grad / grad_mag

        # Check if algorithm can terminate

        if grad_mag <= stop_gradient:
            break

        # Move in direction of gradient

        delta_pos = -grad_dir * learning_rate

        curr_pos += delta_pos

        # Increment iteration index

        iteration_idx += 1

    return curr_pos
