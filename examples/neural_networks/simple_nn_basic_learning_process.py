from neural_networks.networks import Network
from neural_networks.layers import DenseLayer, ReluActivationLayer, SigmoidActivationLayer
from math_util.vector_functions import sum_of_squares_cost
import matplotlib.pyplot as plt
import numpy as np

# Simple example Neural Network trying to determine if the sum of the first 2 inputs is greater than the sum of the last 2

learning_rate = 1e-1
it_count = 1000

nn = Network([
    DenseLayer(4, 4, learning_rate),
    ReluActivationLayer(4, learning_rate),
    DenseLayer(4, 1, learning_rate),
    SigmoidActivationLayer(1, learning_rate),
])

# Prepare the sample inputs

inps = np.array([
    [ 1, 4, 2, 0 ],
    [ 3, 3, 5, 4 ],
    [ 9, 9, 2, 9 ],
    [ 9, 8, 6, 5 ],
    [ 9, 0, 6, 6 ],
    [ 6, 7, 6, 2 ],
    [ 9, 9, 9, 6 ],
    [ 3, 6, 5, 8 ],
], dtype=np.float64)
sample_count = inps.shape[0]

inps[:] /= np.max(inps[:,:])

# Calculate expected outputs

exps = np.where(
    np.sum(inps[:,:2], axis=1) > np.sum(inps[:,2:], axis=1),
    1,
    0,
)[:,np.newaxis]

# Prepare cost results

avg_costs = np.zeros(shape=(it_count+1,), dtype=np.float64)

# Perform iterations

for it_idx in range(it_count):

    # Add new cost value

    costs = [nn.calculate_cost(inps[i], exps[i], sum_of_squares_cost) for i in range(sample_count)]
    avg_cost = sum(costs) / len(costs)
    avg_costs[it_idx] = avg_cost

    # Perform learning iterations

    for sample_idx in range(sample_count):
        inp = inps[sample_idx]
        exp = exps[sample_idx]
        nn.learn_step_single(inp, exp, sum_of_squares_cost)

# Add new cost value

costs = [nn.calculate_cost(inps[i], exps[i], sum_of_squares_cost) for i in range(sample_count)]
avg_cost = sum(costs) / len(costs)
avg_costs[-1] = avg_cost

# Plot results

plt.plot(avg_costs[1:])
plt.show(block=True)
