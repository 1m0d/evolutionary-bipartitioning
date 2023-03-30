from os import path
import numpy as np

from constants import ROOT_DIR
from src.graph import Graph, Node, GRAPH_SIZE

from collections import deque
from collections import defaultdict

graph = Graph.from_file(path.join(ROOT_DIR, "graph.txt"))
gene = np.random.randint(0, 2, GRAPH_SIZE)
graph.set_partitions(gene=gene)
graph.count_crossing_edges()


## datastructure for gainbuckets 
def fm_pass(graph: Graph):
    left, right, highest_gain_left, highest_gain_right = initialize_buckets(graph)

    


def initialize_buckets(graph: Graph):
    left = defaultdict(deque)
    right = defaultdict(deque)
    
    highest_gain_left = float("-inf")
    highest_gain_right = float("-inf")

    for node in graph.nodes.values():
        internal, external = 0, 0 

        for neighbour in node.adjacent:
            if neighbour.partition == node.partition:
                internal += 1
            else:
                external += 1
        gain = internal - external
        if node.partition == 0:
            if gain > highest_gain_left:
                highest_gain_left = gain
            left[gain].append(node)
        else:
            if gain > highest_gain_right:
                highest_gain_right = gain
            right[gain].append(node)

    return left, right, highest_gain_left, highest_gain_right

fm_pass(graph)