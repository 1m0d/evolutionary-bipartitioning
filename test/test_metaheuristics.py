import pytest
from os import path
from constants import ROOT_DIR
from src.metaheuristics import (
    genetic_local_search,
    iterated_local_search,
    multi_start_local_search,
    multi_start_local_search_with_timelimit
)
from src.graph import Graph
import numpy as np


@pytest.fixture
def graph():
    return Graph.from_file(path.join(ROOT_DIR, "graph.txt"))


def test_multi_start_local_search(graph):
    graph, best_crossing_edges = multi_start_local_search(
        graph=graph, max_iterations=50
    )

    assert best_crossing_edges >= 0

    gene = graph.get_gene()
    assert np.count_nonzero(gene == -1) == 0
    assert np.count_nonzero(gene == 0) == 250
    assert np.count_nonzero(gene == 1) == 250


def test_iterated_local_search(graph):
    graph, best_crossing_edges = iterated_local_search(
        graph=graph, perturbation_factor=0.01, max_iterations=50
    )

    assert best_crossing_edges >= 0

    gene = graph.get_gene()
    assert np.count_nonzero(gene == -1) == 0
    assert np.count_nonzero(gene == 0) == 250
    assert np.count_nonzero(gene == 1) == 250


def test_genetic_local_search(graph):
    graph, best_crossing_edges = genetic_local_search(
        graph=graph, population_size=50, max_iterations=50
    )

    assert best_crossing_edges >= 0

    gene = graph.get_gene()
    assert np.count_nonzero(gene == -1) == 0
    assert np.count_nonzero(gene == 0) == 250
    assert np.count_nonzero(gene == 1) == 250

def test_multi_start_local_search_with_timelimit(graph):
       graph, best_crossing_edges = multi_start_local_search_with_timelimit(
        graph=graph, max_iterations=50000, time_limit=5
    )
