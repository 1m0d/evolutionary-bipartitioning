from os import path
import numpy as np

from constants import ROOT_DIR
from src.graph import Graph


def test_from_file():
    graph = Graph.from_file(path.join(ROOT_DIR, "graph.txt"))
    assert len(graph.nodes) == 500


def test_set_partitions():
    gene = np.random.randint(0, 1, 500)
    graph = Graph.from_file(path.join(ROOT_DIR, "graph.txt"))
    graph.set_partitions(gene=gene)
    for idx, partition in enumerate(gene, 1):
        assert graph.nodes[idx].partition == partition


def test_count_crossing_edges():
    gene = np.random.randint(0, 2, 500)
    graph = Graph.from_file(path.join(ROOT_DIR, "graph.txt"))
    graph.set_partitions(gene=gene)
    graph.count_crossing_edges()
    assert isinstance(graph.crossing_edges, int)
    assert graph.crossing_edges > 0
