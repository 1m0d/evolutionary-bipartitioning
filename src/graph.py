from typing import Dict, List, Literal, Optional

class Node:
    def __init__(self, idx: int):
        self.idx: int = idx
        self.adjacent: List[Node] = []
        self.partition : Optional[Literal["0", "1"]] = None

    def add_neighbor(self, neighbor: 'Node'):
        self.adjacent.append(neighbor)

class Graph:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}

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
    def from_file(cls, file_path: str) -> 'Graph':
        graph = cls()

        with open(file_path) as file:
            for line in file:
                data = line.split()
                id = data[0]

                if len(data) == 1:
                    graph.add_node(int(id))
                    continue

                for node in data[1:]:
                    graph.add_edge(int(id), int(node))

        return graph

