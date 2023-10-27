from neural_networks.networks import Network
from neural_networks.layers import PolynomialLayer
from math_util.vector_functions import sum_of_squares_cost
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Tuple

LEARNING_RATES = [1e-1, 9e-2, 7e-2, 5e-2, 3e-2, 1e-2, 1e-3]
ITERATION_COUNT = 1000
LAYER_ORDER = 4

# Example Neural Network with only a single, polynomial layer taking 3 inputs and giving 1 output
# The expected output's formula is given by,
#     y = 0.1 - x[0] + 7*x[2] + x[1]^2 - 2*x[2]^3 + x[0]^4

# Prepare input data

X = np.column_stack([
    np.linspace(0, 1, 100),
    np.linspace(-0.5, 0.5, 100),
    np.linspace(-0.1, 0, 100),
])
y = 0.1 - (X[:,0]) + (7*X[:,2]) + (X[:,1]**2) - (2*X[:,2]**3) + (X[:,0]**4)
Y = y[:,np.newaxis]

# Network training function

def train_network(learning_rate: float, iteration_count: int = ITERATION_COUNT) -> Tuple[npt.NDArray, Network]:

    # Create network

    nn = Network([
        PolynomialLayer(
            3, 1,
            learning_rate=learning_rate,
            order=LAYER_ORDER,
        ),
    ])

    # Train the network

    avg_costs = []

    for costs in nn.learn_stochastic(
        xs=X,
        exps=Y,
        cost_func=sum_of_squares_cost,
        sample_size=10,
        iteration_count=iteration_count,
        provide_cost_output=True,
    ):
        avg_costs.append(np.mean(costs))

    # Output

    return np.array(avg_costs), nn

# Outputs

fig,ax = plt.subplots(1,1)

for lr in LEARNING_RATES:

    avg_costs, _ = train_network(learning_rate=lr)

    ax.plot(
        np.arange(1, avg_costs.shape[0]+1),
        np.log(avg_costs),
        label=f"learning_rate={lr}",
    )

ax.set_ylim(None, 50)  # type: ignore

ax.set_title("Accuracy of Model over Learning Iterations for different Learning Rates")
ax.set_xlabel("Iteration Number")
ax.set_ylabel("ln (average cost)")
ax.legend()

plt.show()
