import random
from typing import Dict, Literal, Optional, Set


class Node:
    def __init__(self, idx: int):
        self.idx: int = idx
        self.adjacent: Set[Node] = set()
        self.partition: Literal["0", "1"] = random.choice(("0", "1"))

    def add_neighbor(self, neighbor: "Node"):
        self.adjacent.add(neighbor)

    def __str__(self):
        return (
            f"[{self.partition}] Node {self.idx}:"
            f" {[node.idx for node in self.adjacent]}"
        )


class Graph:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.crossing_edges: int

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

    def get_node(self, idx: int) -> Optional[Node]:
        return self.nodes.get(idx)

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

        graph.count_crossing_edges()
        return graph

    def count_crossing_edges(self) -> int:
        crossing_edges = 0
        for node in self.nodes.values():
            if node.partition != "0":
                continue

            for adjacent in node.adjacent:
                if adjacent.partition == "1":
                    crossing_edges += 1

        self.crossing_edges = crossing_edges
        return self.crossing_edges
