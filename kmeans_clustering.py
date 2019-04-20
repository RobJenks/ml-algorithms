import random
from typing import List, Tuple
from common import plot


def run_kmeans_clustering():
    n = 1000
    k = 3
    dimensions = 2

    data = generate_data(dimensions, -10, 10, n, k)

    clusters = perform_kmeans_clustering(data, k)

    if dimensions == 2:
        plot.plot_data_grouped(data, clusters)
        plot.show_plot()


# Generate a 'dims'-dimensional dataset of size n, attempting to maintain k approximate clusters
def generate_data(dims, minbound, maxbound, n, k) -> List[Tuple]:

    # Generate random dataset which we then perturb into approximate clusters
    data = [tuple(random.uniform(minbound, maxbound) for _ in range(dims)) for _ in range(n)]
    centroids = [tuple(random.uniform(minbound, maxbound) for _ in range(dims)) for _ in range(k)]

    points = []
    for pt in data:
        dsq, centroid = min((distsq(pt, x), x) for x in centroids)
        delta = vsub(centroid, pt)

        points.append(vadd(pt, vscale(delta, random.uniform(0.0, 0.4))))

    return points


def perform_kmeans_clustering(data, k):
    n = len(data)
    dims = len(data[0])
    clusters = [-1 for _ in range(n)]

    # Select random points to act as the initial centroids; perturb to avoid identical points
    centroids = tuple(data[random.randint(0, n)] + (random.uniform(0.0, 1.0), ) * dims for _ in range(k))

    for i in range(MAX_ITERATIONS):
        new_clusters, centroids = perform_kmeans_iteration(data, centroids)

        if new_clusters == clusters:
            print("K-Means clustering converged after {} iterations".format(i))
            return clusters

        if VISUALISE_CLUSTER_CHANGES and dims == 2:
            plot.plot_data_grouped(data + centroids, [1 if x != y else 0 for (x, y) in zip(clusters, new_clusters)] + [2] * k)
            plot.show_plot()

        clusters = new_clusters

    print("WARNING: K-Means clustering did not converge after {} iterations; result may be innacurate".format(MAX_ITERATIONS))


def perform_kmeans_iteration(data, centroids):
    dims = len(data[0])
    clusters: List[int] = []

    # Allocate to cluster for the closest centroid
    for pt in data:
        _, cluster = min((distsq(pt, x), i) for (i, x) in enumerate(centroids))
        clusters.append(cluster)

    # Recalculate cluster positions as centroid of all assigned points
    cdata = [((0,)*dims, 0) for _ in range(len(centroids))]
    for (pt, cluster) in zip(data, clusters):
        cdata[cluster] = (vadd(cdata[cluster][0], pt), cdata[cluster][1] + 1)

    new_centroids = [(0,)*dims if n == 0 else vscale(x, 1.0 / n) for (x, n) in cdata]
    return clusters, new_centroids


# Force-stop in case of no convergence
MAX_ITERATIONS = 10_000


# Visualisation option to show the points changing cluster assignment each iteration
VISUALISE_CLUSTER_CHANGES = True


# Vector operations
def distsq(v0: Tuple, v1: Tuple):
    return sum(pow(abs(x0 - x1), 2) for (x0, x1) in zip(v0, v1))


def vadd(v0: Tuple, v1: Tuple):
    return tuple(x + y for (x, y) in zip(v0, v1))


def vsub(v0: Tuple, v1: Tuple):
    return tuple(x - y for (x, y) in zip(v0, v1))


def vscale(v: Tuple, s):
    return tuple(x * s for x in v)

