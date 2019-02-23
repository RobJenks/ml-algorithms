from typing import List, Tuple
import matplotlib.pyplot as plt


def plot_data(data: List[Tuple[float, float]]):
    plt.plot([x for (x, _) in data], [y for (_, y) in data], 'o')


def plot_line(x: List[float], y: List[float]):
    plt.plot(x, y, lw=2)


def plot_trend(xmin, xmax, theta0, theta1):
    plot_line([xmin, xmax], [theta0 + (theta1 * x) for x in [xmin, xmax]])


def plot_linear_fn(fn, xmin, xmax):
    plot_line([xmin, xmax], [fn(xmin), fn(xmax)])


def plot_fn(fn, x: List[float]):
    plot_line(x, [fn(x) for x in x])


def show_plot():
    plt.show()