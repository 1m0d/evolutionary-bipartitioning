import numpy as np
from src.graph import GRAPH_SIZE
from src.FM import fm_pass


def multi_start_local_search(graph, num_starts):
    best_graph = None
    best_crossing_edges = float("inf")

    for _ in range(num_starts):
        # Generate a random initial solution
        gene = np.concatenate((np.zeros(GRAPH_SIZE // 2), np.ones(GRAPH_SIZE // 2)))
        np.random.shuffle(gene)
        graph.set_partitions(gene=gene)

        # Perform the FM algorithm as the local search operator
        graph = fm_pass(graph, verbose=False)

        # Update the best solution found so far
        crossing_edges = graph.count_crossing_edges()
        print(
            "crossing_edges:",
            crossing_edges,
            "best_crossing_edges:",
            best_crossing_edges,
        )
        if crossing_edges < best_crossing_edges:
            best_graph = graph
            best_crossing_edges = crossing_edges

    return best_graph, best_crossing_edges


def iterated_local_search(graph, max_iterations, perturbation_factor):
    # Generate a random initial solution
    gene = np.concatenate((np.zeros(GRAPH_SIZE // 2), np.ones(GRAPH_SIZE // 2)))
    np.random.shuffle(gene)
    graph.set_partitions(gene=gene)

    best_graph = graph
    best_crossing_edges = graph.count_crossing_edges()

    for _ in range(max_iterations):
        # Perturb the current solution by swapping a random subset of nodes between the partitions
        perturbation_size = int(perturbation_factor * GRAPH_SIZE)
        nodes_to_swap = np.random.choice(
            list(graph.nodes.keys()), size=perturbation_size, replace=False
        )
        for node_index in nodes_to_swap:
            graph.nodes[node_index].partition = 1 - graph.nodes[node_index].partition

        # Perform local search on the perturbed solution using the FM algorithm
        perturbed_graph = fm_pass(graph, verbose=False)

        # Accept the new solution if it improves the objective function
        crossing_edges = perturbed_graph.count_crossing_edges()
        print(crossing_edges, best_crossing_edges)
        if crossing_edges < best_crossing_edges:
            best_graph = perturbed_graph
            best_crossing_edges = crossing_edges
        print(best_crossing_edges)

        # Use the best solution found so far as the new starting point
        graph = best_graph

    return best_graph, best_crossing_edges
