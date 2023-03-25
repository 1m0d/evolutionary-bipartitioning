from constants import ROOT_DIR
from src.graph import Graph
from os import path


def test_from_file():
    graph = Graph.from_file(path.join(ROOT_DIR, "graph.txt"))
    assert len(graph.nodes) == 500
