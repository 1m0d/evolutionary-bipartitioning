from os import path
import numpy as np
import matplotlib.pyplot as plt
from constants import ROOT_DIR
from src.metaheuristics import (
    genetic_local_search,
    iterated_local_search,
    multi_start_local_search,
)
from src.graph import Graph
from scipy.stats import mannwhitneyu
from concurrent.futures import ProcessPoolExecutor


def run_experiment_once(args):
    graph = Graph.from_file(path.join(ROOT_DIR, "graph.txt"))
    algorithm, params = args
    _, crossing_edges = algorithm(graph, **params)
    return crossing_edges


def run_experiment_parallel(algorithm, params, n_runs=20):
    with ProcessPoolExecutor() as executor:
        results = list(
            executor.map(
                run_experiment_once, [(algorithm, params) for _ in range(n_runs)]
            )
        )
    return np.array(results)


def plot_boxplots(data, labels, title):
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.xlabel("Algorithms")
    plt.ylabel("Crossing Edges")
    plt.show()


algorithms = [multi_start_local_search, iterated_local_search, genetic_local_search]
algorithm_names = ["MLS", "ILS", "GLS"]

params_msls = {"max_iterations": 10000}
params_ils = {"max_iterations": 10000, "perturbation_factor": 0.006}
params_gls = {"population_size": 70, "max_iterations": 10000}

params_list = [params_msls, params_ils, params_gls]

# Run the experiment
n_runs = 20
results = []

for algorithm, params in zip(algorithms, params_list):
    result = run_experiment_parallel(algorithm, params, n_runs)
    results.append(result)

# Perform the Wilcoxon-Mann-Whitney U test
for i in range(len(algorithms)):
    for j in range(i + 1, len(algorithms)):
        u_stat, p_value = mannwhitneyu(results[i], results[j], alternative="two-sided")
        print(
            f"{algorithm_names[i]} vs {algorithm_names[j]}: p-value = {p_value}, u-stat = {u_stat}"
        )

# Plot the results
plot_boxplots(results, algorithm_names, "Comparison of Algorithms")
