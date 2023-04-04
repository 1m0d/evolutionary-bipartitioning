from collections import defaultdict, deque
from typing import List, Tuple

import numpy as np

from src.graph import GRAPH_SIZE, Graph, Node


def fm_pass(graph: Graph, verbose=False) -> Graph:
    # create the gain buckets datastructure and add all nodes with their respective gains
    left, right, highest_gain_left, highest_gain_right = initialize_buckets(graph)
    # determine the cut size of the current partition
    original_cut = graph.count_crossing_edges()
    new_cut = original_cut
    # switch_order will keep track of the order in which nodes are switched and the corresponding cut value
    switch_order = [(-1, original_cut)]

    for i in range(GRAPH_SIZE):
        # select a node to switch from left to right and vice versa continuously making sure the gainbuckets are never empty
        if i % 2 == 0:
            if highest_gain_left != float("-inf"):
                node, left, highest_gain_left = select_node_max_gain(
                    left, highest_gain_left
                )
            else:
                break
        else:
            if highest_gain_right != float("-inf"):
                node, right, highest_gain_right = select_node_max_gain(
                    right, highest_gain_right
                )
            else:
                break

        # switch node partition
        node.switch_partition()

        # find every neighbour that's still free, remove it from the gain bucket it's in, recalculate the gain and put it into the new bucket
        for neighbour in node.adjacent:
            left, right, highest_gain_left, highest_gain_right = update_neighbour(
                node, neighbour, left, right, highest_gain_left, highest_gain_right
            )

        # update the cut value of the partition after switching and add switched node to the switch order
        new_cut -= node.gain
        graph.crossing_edges = new_cut
        switch_order.append((node, new_cut))

    if verbose:
        print_solution(switch_order)

    # recover the solution by rolling back all the switches in backwards order until you arrive at best_partition
    return roll_back_to_best(graph, switch_order)


def initialize_buckets(graph: Graph) -> Tuple[defaultdict, defaultdict, int, int]:
    left, right = defaultdict(deque), defaultdict(deque)
    highest_gain_left, highest_gain_right = float("-inf"), float("-inf")

    for node in graph.nodes.values():
        gain = node.calculate_gain()
        if node.partition == 0:
            if gain > highest_gain_left:
                highest_gain_left = gain
            left[gain].append(node)
        else:
            if gain > highest_gain_right:
                highest_gain_right = gain
            right[gain].append(node)

    return left, right, highest_gain_left, highest_gain_right


def update_neighbour(
    node: Node,
    neighbour: Node,
    left: defaultdict,
    right: defaultdict,
    highest_gain_left: int,
    highest_gain_right: int,
):
    if neighbour.partition == 0:
        # find the deque and gain it's currently in and remove it being careful to update pointer if one or more buckets become empty
        # if the node is already fixed nothing happens
        if neighbour.gain in left.keys() and neighbour in left[neighbour.gain]:
            left, highest_gain_left = remove_from_bucket(
                neighbour, left, highest_gain_left
            )
            new_gain = recalculate_gain(node, neighbour)
            left, highest_gain_left = add_to_bucket(
                neighbour, new_gain, left, highest_gain_left
            )
    else:
        if neighbour.gain in right.keys() and neighbour in right[neighbour.gain]:
            right, highest_gain_right = remove_from_bucket(
                neighbour, right, highest_gain_right
            )
            new_gain = recalculate_gain(node, neighbour)
            right, highest_gain_right = add_to_bucket(
                neighbour, new_gain, right, highest_gain_right
            )

    return left, right, highest_gain_left, highest_gain_right


def select_node_max_gain(
    buckets: defaultdict, highest_gain: int
) -> Tuple[Node, defaultdict, int]:
    node = buckets[highest_gain].pop()
    new_highest_gain = update_highest_gain_after_pop(buckets, highest_gain)

    return node, buckets, new_highest_gain


def update_highest_gain_after_pop(buckets: defaultdict, highest_gain: int) -> int:
    new_highest_gain = highest_gain
    # if the bucket becomes empty
    if len(buckets[highest_gain]) == 0:
        # remove the deque corresponding to that gain
        del buckets[highest_gain]
        # and store the new highest gain
        if len(buckets.keys()) == 0:
            new_highest_gain = float("-inf")
            return new_highest_gain
        new_highest_gain = sorted(buckets.keys(), reverse=True)[0]

    return new_highest_gain


def remove_from_bucket(
    node: Node, buckets: defaultdict, highest_gain: int
) -> Tuple[defaultdict, int]:
    old_gain = node.gain
    buckets[old_gain].remove(node)
    new_highest_gain = update_highest_gain_after_remove(buckets, highest_gain, old_gain)

    return buckets, new_highest_gain


def update_highest_gain_after_remove(
    buckets: defaultdict, highest_gain: int, old_gain: int
) -> int:
    new_highest_gain = highest_gain
    if old_gain == highest_gain:
        new_highest_gain = update_highest_gain_after_pop(buckets, highest_gain)
    else:
        if len(buckets[old_gain]) == 0:
            del buckets[old_gain]

    return new_highest_gain


def recalculate_gain(switch_node: Node, neighbour_node: Node) -> int:
    new_partition_switch = switch_node.partition
    if neighbour_node.partition == new_partition_switch:
        neighbour_node.gain -= 2
    else:
        neighbour_node.gain += 2

    return neighbour_node.gain


def add_to_bucket(
    node: Node, new_gain: int, buckets: defaultdict, highest_gain: int
) -> Tuple[defaultdict, int]:
    buckets[new_gain].append(node)

    new_highest_gain = highest_gain
    if new_gain > highest_gain:
        new_highest_gain = new_gain

    return buckets, new_highest_gain


def roll_back_to_best(graph: Graph, switch_order: List[Tuple[Node, int]]) -> Graph:
    # recover the solution by rolling back all the switches in backwards order until you arrive at best_partition
    order = np.argsort([cut[1] for cut in switch_order])
    best_iteration = order[0]
    i = 0
    while best_iteration % 2 != 0:
        i += 1
        best_iteration = order[i]


    for i in range(len(switch_order) - 1, best_iteration, -1):
        node = switch_order[i][0]
        node.switch_partition()

    return graph


def print_solution(switch_order):
    print("# crossing edges:", [cut[1] for cut in switch_order])
    print("Node switching order:", [cut[0].idx for cut in switch_order[1:]])
    print("Final # crossing edges:", graph.count_crossing_edges())
    print("Best # crossing edges:", np.min([cut[1] for cut in switch_order]))
