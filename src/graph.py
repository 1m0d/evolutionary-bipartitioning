from typing import Dict, Final, Literal, Set
import numpy as np

GRAPH_SIZE: Final = 500


class Node:
    def __init__(self, idx: int):
        self.idx: int = idx
        self.adjacent: Set[Node] = set()
        self.partition: Literal[0, 1]
        self.gain: int
        self.switched = False

    def add_neighbor(self, neighbor: "Node"):
        self.adjacent.add(neighbor)

    def calculate_gain(self):
        internal, external = 0, 0

        for neighbour in self.adjacent:
            if neighbour.partition == self.partition:
                internal += 1
            else:
                external += 1
        gain = external - internal
        self.gain = gain
        return gain

    def switch_partition(self):
        self.partition = 1 - self.partition
        self.switched = True

    def __str__(self):
        return (
            f"[{self.partition}] Node {self.idx}:"
            f" {[node.idx for node in self.adjacent]}"
        )


class Graph:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.crossing_edges: int = 0

    def add_node(self, idx: int) -> Node:
        node = Node(idx)
        self.nodes[idx] = node
        return node

    def add_edge(self, source: int, dest: int) -> None:
        if source not in self.nodes:
            self.add_node(source)
        if dest not in self.nodes:
            self.add_node(dest)
        self.nodes[source].add_neighbor(self.nodes[dest])
        self.nodes[dest].add_neighbor(self.nodes[source])

    @classmethod
    def from_file(cls, file_path: str) -> "Graph":
        graph = cls()

        with open(file_path, encoding="utf-8") as file:
            for line in file:
                data = line.split()
                idx = data[0]

                if len(data) == 1:
                    graph.add_node(int(idx))
                    continue

                for node in data[1:]:
                    graph.add_edge(int(idx), int(node))

        return graph

    def count_crossing_edges(self) -> int:
        crossing_edges = 0
        for node in self.nodes.values():
            if node.partition != 0:
                continue

            for adjacent in node.adjacent:
                if adjacent.partition == 1:
                    crossing_edges += 1

        self.crossing_edges = crossing_edges
        return self.crossing_edges

    def set_partitions(self, gene: np.ndarray) -> "Graph":
        for idx, partition in enumerate(gene, 1):
            self.nodes[idx].partition = partition

        self.crossing_edges = 0
        return self

    def get_gene(self) -> np.ndarray:
        gene = np.full(GRAPH_SIZE, -1)
        for idx in range(GRAPH_SIZE):
            gene[idx] = self.nodes[idx + 1].partition

        return gene
