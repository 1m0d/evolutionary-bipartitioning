import matplotlib.pyplot as plt
from src.metaheuristics import genetic_local_search
from src.graph import Graph
from constants import ROOT_DIR
from concurrent.futures import ProcessPoolExecutor
from os import path
import numpy as np


def run_gls(population_size, max_iterations):
    graph = Graph.from_file(path.join(ROOT_DIR, "graph.txt"))
    _, crossing_edges = genetic_local_search(graph, population_size, max_iterations)
    return crossing_edges


def perform_experiment(population_sizes, max_iterations, num_runs):
    crossing_edges_results = []

    with ProcessPoolExecutor() as executor:
        for population_size in population_sizes:
            current_population_results = []

            futures = [
                executor.submit(run_gls, population_size, max_iterations)
                for _ in range(num_runs)
            ]

            for future in futures:
                crossing_edges = future.result()
                current_population_results.append(crossing_edges)

            crossing_edges_results.append(current_population_results)

    return crossing_edges_results


def plot_boxplot(population_sizes, crossing_edges_results):
    fig, ax = plt.subplots()
    ax.boxplot(crossing_edges_results, showmeans=True)

    # Add mean and standard deviation as labels
    means = [np.mean(result) for result in crossing_edges_results]
    stds = [np.std(result) for result in crossing_edges_results]

    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(
            i + 1,
            mean,
            f"{mean:.2f} ± {std:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="blue",
        )

    ax.set_xticklabels(population_sizes)
    ax.set_xlabel("Population Size")
    ax.set_ylabel("Crossing Edges")
    ax.set_title("Genetic Local Search Performance with Different Population Sizes")
    plt.savefig("test.png")


# Usage example
population_sizes = [30, 40, 50, 60, 70]
max_iterations = 10000
num_runs = 20

crossing_edges_results = perform_experiment(population_sizes, max_iterations, num_runs)

# Plot the results using a boxplot
plot_boxplot(population_sizes, crossing_edges_results)
