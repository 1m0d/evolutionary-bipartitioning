from os import path
import numpy as np

from constants import ROOT_DIR
from src.graph import Graph, GRAPH_SIZE
from src.FM import fm_pass


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


def test_FM_pass():
    # np.random.seed(41)
    graph = Graph.from_file(path.join(ROOT_DIR, "graph.txt"))

    # Create an array with equal amount zeros and ones and shuffle to give first random partition
    gene = np.concatenate((np.zeros(GRAPH_SIZE // 2), np.ones(GRAPH_SIZE // 2)))
    np.random.shuffle(gene)
    graph.set_partitions(gene=gene)

    print("# crossing edges before FM pass:", graph.count_crossing_edges())
    graph = fm_pass(graph, verbose=False)
    print("# crossing edges after FM pass:", graph.count_crossing_edges())


test_FM_pass()