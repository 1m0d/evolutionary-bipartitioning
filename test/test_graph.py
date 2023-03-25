from os import path

from constants import ROOT_DIR
from src.graph import Graph


def test_from_file():
    graph = Graph.from_file(path.join(ROOT_DIR, "graph.txt"))
    assert len(graph.nodes) == 500
    assert isinstance(graph.crossing_edges, int)
    assert graph.crossing_edges > 0
