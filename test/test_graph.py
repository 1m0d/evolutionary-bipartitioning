from os import path

from constants import ROOT_DIR
from src.graph import Graph


def test_from_file():
    graph = Graph.from_file(path.join(ROOT_DIR, "graph.txt"))
    assert len(graph.nodes) == 500
    assert isinstance(graph.crossing_edges, int)
    assert graph.crossing_edges > 0


def test_set_partitions():
    binary_string = "0" * 250 + "1" * 250
    graph = Graph.from_file(path.join(ROOT_DIR, "graph.txt"))
    graph.set_partitions(binary_string=binary_string)
    assert graph.nodes[1].partition == graph.nodes[250].partition == "0"
    assert graph.nodes[251].partition == graph.nodes[500].partition == "1"
