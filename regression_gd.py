import sys
from typing import List, Tuple, Callable
from numpy import random
from common import plot, mth


def run_linear_regression_gradient_descent():
    data = generate_data(-3.2, -2.76, 100, 48)
    theta, f = gradient_descent(data, 1_000_000)
    print(f"Solution: y = {theta[0]} + {theta[1]}x")

    plot.plot_data(data)
    plot.plot_linear_fn(f, 0, 100)
    plot.show_plot()


def gradient_descent(data: List[Tuple[float, float]], iterations: int = sys.maxsize) \
                     -> Tuple[Tuple[float, float], Callable[[float], float]]:
    theta = (0, 1)
    cost = cost_function(linear_fn(*theta), data)
    step = [0.1, 0.1]       # Initial step for partial derivatives

    for i in range(iterations):
        new_step = []
        for t in range(len(theta)):
            new_theta = tuple((x + d if k == t else x) for (k, (x, d)) in enumerate(zip(theta, step)))
            new_cost = cost_function(linear_fn(*new_theta), data)
            new_step += [-1 * learning_rate() * mth.safediv((new_cost - cost), (new_theta[t] - theta[t]))]

        step = new_step[:]
        theta = tuple(x + d for (x, d) in zip(theta, step))
        cost = cost_function(linear_fn(*theta), data)

        if i % 10000 == 0: print(f"{i}: J{theta}={cost} -> step={step}")
        if all(x == 0 for x in step): break     # Convergence

    return theta, linear_fn(*theta)


# Returns a functor based on the given linear function parameters
def linear_fn(theta0, theta1) -> Callable[[float], float]:
    return lambda x: theta0 + (theta1 * x)


# Generate data about a base linear function y = theta0 + theta1*x.  n items will be generated which have a
# spread about the function that is normally-distributed
def generate_data(theta0, theta1, n, spread):
    f = linear_fn(theta0, theta1)
    return [(x, f(x) + (random.normal() * spread)) for x in range(n)]


# Evaluate cost function for a given hypothesis function and dataset, using a minimised-sum-of-squares approach
def cost_function(fn, data: List[Tuple[float, float]]):
    return (1 / (2 * len(data))) * sum((fn(x) - y)**2 for (x, y) in data)


def learning_rate():
    return 0.0001

