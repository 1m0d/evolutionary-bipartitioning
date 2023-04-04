import numpy as np
import random
from src.graph import GRAPH_SIZE, Graph
from src.fm import fm_pass
from src.crossover import uniform_crossover
from time import perf_counter


def _random_gene() -> np.ndarray:
    gene = np.ones(GRAPH_SIZE)
    gene[: GRAPH_SIZE // 2] = 0
    np.random.shuffle(gene)
    return gene


def multi_start_local_search(graph, max_iterations, verbose=False):
    best_gene = None
    best_crossing_edges = float("inf")

    for _ in range(max_iterations):
        # Generate a random initial solution
        gene = _random_gene()
        graph.set_partitions(gene=gene)

        # Perform the FM algorithm as the local search operator
        graph, crossing_edges = fm_pass(graph, verbose=False)

        # Update the best solution found so far
        if crossing_edges < best_crossing_edges:
            best_gene = graph.get_gene()
            best_crossing_edges = crossing_edges

        if verbose:
            print(
                "crossing_edges:",
                crossing_edges,
                "best_crossing_edges:",
                best_crossing_edges,
            )

    graph.set_partitions(gene=best_gene)
    return graph, best_crossing_edges


def _invert_bits(gene, perturbation_size):
    zeros_indices = np.where(gene == 0)[0]
    ones_indices = np.where(gene == 1)[0]

    # Randomly select indices to invert
    selected_zeros_indices = np.random.choice(
        zeros_indices, size=perturbation_size // 2, replace=False
    )
    selected_ones_indices = np.random.choice(
        ones_indices, size=perturbation_size // 2, replace=False
    )

    # Invert the selected values
    gene[selected_zeros_indices] = 1
    gene[selected_ones_indices] = 0
    return gene

def multi_start_local_search_with_timelimit(graph, time_limit, verbose=False):
    best_graph = graph
    best_crossing_edges = float("inf")
    start = perf_counter()
    time = start

    while True:
        if time > start + time_limit:
            break
        # Generate a random initial solution
        gene = _random_gene()
        graph.set_partitions(gene=gene)

        # Perform the FM algorithm as the local search operator
        graph = fm_pass(graph, verbose=False)

        # Update the best solution found so far
        crossing_edges = graph.count_crossing_edges()
        if crossing_edges < best_crossing_edges:
            best_graph = graph
            best_crossing_edges = crossing_edges

        if verbose:
            print(
                "crossing_edges:",
                crossing_edges,
                "best_crossing_edges:",
                best_crossing_edges,
            )

        time = perf_counter()

    return best_graph, best_crossing_edges


def iterated_local_search(graph, max_iterations, perturbation_factor, verbose=False):
    # Generate a random initial solution
    gene = _random_gene()
    graph.set_partitions(gene=gene)

    best_graph = graph
    best_crossing_edges = graph.count_crossing_edges()

    perturbation_size = int(perturbation_factor * GRAPH_SIZE)

    for _ in range(max_iterations):
        # Perturb the current solution by swapping a random subset of nodes between the partitions
        inverted_gene = _invert_bits(
            gene=graph.get_gene(), perturbation_size=perturbation_size
        )
        graph.set_partitions(gene=inverted_gene)

        # Perform local search on the perturbed solution using the FM algorithm
        perturbed_graph, crossing_edges = fm_pass(graph, verbose=False)

        # Accept the new solution if it improves the objective function
        if crossing_edges < best_crossing_edges:
            best_graph = perturbed_graph
            best_crossing_edges = crossing_edges

        if verbose:
            print(crossing_edges, best_crossing_edges)

        # Use the best solution found so far as the new starting point
        graph = best_graph

    return best_graph, best_crossing_edges

def iterated_local_search_with_timelimit(graph, perturbation_factor, time_limit, verbose=False):
    # Generate a random initial solution
    gene = _random_gene()
    graph.set_partitions(gene=gene)

    best_graph = graph
    best_crossing_edges = graph.count_crossing_edges()

    perturbation_size = int(perturbation_factor * GRAPH_SIZE)

    start = perf_counter()
    time = start

    while True:
        if time > start + time_limit:
            break
         
        # Perturb the current solution by swapping a random subset of nodes between the partitions
        nodes_to_swap = np.random.choice(
            list(graph.nodes.keys()), size=perturbation_size, replace=False
        )
        for node_index in nodes_to_swap:
            graph.nodes[node_index].partition = 1 - graph.nodes[node_index].partition

        # Perform local search on the perturbed solution using the FM algorithm
        perturbed_graph = fm_pass(graph, verbose=False)

        # Accept the new solution if it improves the objective function
        crossing_edges = perturbed_graph.count_crossing_edges()
        if crossing_edges < best_crossing_edges:
            best_graph = perturbed_graph
            best_crossing_edges = crossing_edges

        if verbose:
            print(crossing_edges, best_crossing_edges)

        # Use the best solution found so far as the new starting point
        graph = best_graph

        time = perf_counter()

    return best_graph, best_crossing_edges


def genetic_local_search(graph: Graph, population_size: int, max_iterations: int):
    population = [_random_gene() for _ in range(population_size)]

    for i in range(population_size):
        graph.set_partitions(gene=population[i])
        graph, _ = fm_pass(graph, verbose=False)
        population[i] = graph.get_gene()

    for _ in range(max_iterations):
        parent1, parent2 = random.sample(population, 2)
        child_gene = uniform_crossover(parent1, parent2)

        graph.set_partitions(gene=child_gene)
        optimized_child, child_crossing_edges = fm_pass(graph, verbose=False)
        optimized_child_gene = optimized_child.get_gene()

        crossing_edges_list = [
            graph.set_partitions(gene=g).count_crossing_edges() for g in population
        ]
        worst_index = np.argmax(crossing_edges_list)
        worst_crossing_edges = crossing_edges_list[worst_index]

        if child_crossing_edges <= worst_crossing_edges:
            population[worst_index] = optimized_child_gene

    best_index = np.argmin(
        [graph.set_partitions(gene=g).count_crossing_edges() for g in population]
    )
    best_gene = population[best_index]
    graph.set_partitions(gene=best_gene)
    best_crossing_edges = graph.count_crossing_edges()

    return graph, best_crossing_edges

def genetic_local_search_with_timelimit(graph: Graph, population_size: int, time_limit):
    population = [_random_gene() for _ in range(population_size)]

    for i in range(population_size):
        graph.set_partitions(gene=population[i])
        graph = fm_pass(graph, verbose=False)
        population[i] = graph.get_gene()

    start = perf_counter()
    time = start

    while True:
        if time > start + time_limit:
            break

        parent1, parent2 = random.sample(population, 2)
        child_gene = uniform_crossover(parent1, parent2)

        graph.set_partitions(gene=child_gene)
        optimized_child = fm_pass(graph, verbose=False)
        child_crossing_edges = optimized_child.count_crossing_edges()
        optimized_child_gene = optimized_child.get_gene()

        crossing_edges_list = [
            graph.set_partitions(gene=g).count_crossing_edges() for g in population
        ]
        worst_index = np.argmax(crossing_edges_list)
        worst_crossing_edges = crossing_edges_list[worst_index]

        if child_crossing_edges <= worst_crossing_edges:
            population[worst_index] = optimized_child_gene

        time = perf_counter()   
        
    best_index = np.argmin(
        [graph.set_partitions(gene=g).count_crossing_edges() for g in population]
    )
    best_gene = population[best_index]
    graph.set_partitions(gene=best_gene)
    best_crossing_edges = graph.count_crossing_edges()


    return graph, best_crossing_edges
