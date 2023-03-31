from os import path
import numpy as np

from constants import ROOT_DIR
from src.graph import Graph, Node, GRAPH_SIZE

from collections import deque
from collections import defaultdict

np.random.seed(41)
graph = Graph.from_file(path.join(ROOT_DIR, "graph.txt"))

# Create an array with zeros and ones
gene = np.concatenate((np.zeros(GRAPH_SIZE // 2), np.ones(GRAPH_SIZE // 2)))
np.random.shuffle(gene)
# gene = np.random.randint(0, 2, GRAPH_SIZE)
graph.set_partitions(gene=gene)
graph.count_crossing_edges()


## datastructure for gainbuckets 
def fm_pass(graph: Graph):
    left, right, highest_gain_left, highest_gain_right = initialize_buckets(graph)
    original_cut = graph.count_crossing_edges()
    best_cut = original_cut
    new_cut = original_cut
    switch_order = [(-1, original_cut)]
    best_partition = -1

    for i in range(GRAPH_SIZE):
        # print(i)

        if i % 2 == 0:
            if highest_gain_left != float("-inf"):
                node, left, highest_gain_left = select_node_max_gain(left, highest_gain_left)
            else: 
                break
            #print("pop node left with gain ", node.gain)
        else:
            if highest_gain_right != float("-inf"):
                node, right, highest_gain_right = select_node_max_gain(right, highest_gain_right)
            else:
                break
            #print("pop node right with gain ", node.gain)
        
        # print("free in left", free_in_partition(left))
        # print("free in right", free_in_partition(right))
        
        # print(f"{node.gain=}, {node.calculate_gain()=}")
        # assert node.gain == node.calculate_gain()
        #show_buckets(True, left, highest_gain_left)
        #show_buckets(False, right, highest_gain_right)

        # for neighbour in node.adjacent:
        #     assert neighbour.gain == neighbour.calculate_gain2()

        node.partition = 1 - node.partition
        # assert new_cut == graph.count_crossing_edges()


        for neighbour in node.adjacent:
            if neighbour.partition == 0:
                # find the deque and gain it's currently in and remove it being careful to update pointer if one or more buckets become empty
                # if the node is already fixed nothing happens  
                if neighbour.gain in left.keys() and neighbour in left[neighbour.gain]:
                    left, highest_gain_left = remove_from_bucket(neighbour, left, highest_gain_left)
                    new_gain = recalculate_gain(node, neighbour)
                    left, highest_gain_left = add_to_bucket(neighbour, new_gain, left, highest_gain_left)
            else:
                if neighbour.gain in right.keys() and neighbour in right[neighbour.gain]:
                    right, highest_gain_right = remove_from_bucket(neighbour, right, highest_gain_right)
                    new_gain = recalculate_gain(node, neighbour)
                    right, highest_gain_right = add_to_bucket(neighbour, new_gain, right, highest_gain_right)


            # calculate the new gain depending on partition of node and neighbor
            # old_gain = neighbour.gain
            # incorrect_gain = recalculate_gain(node, neighbour)
            # correct_gain = neighbour.calculate_gain2()
            # print(f"{old_gain=}, {incorrect_gain=}, {correct_gain=}")
            # assert incorrect_gain == correct_gain
            # new_gain = correct_gain
            # new_gain = recalculate_gain(node, neighbour)
            # print()
            # print("incorrectly calculated new gain", new_gain)
            # # print(f"{neighbour.gain=}")
            # # print(f"{neighbour.calculate_gain2()=}")
            # # print()
            # print("old gain", neighbour.gain)
            # new_gain = neighbour.calculate_gain()
            # print("new gain as it shoudl be", new_gain)
            # assert neighbour.gain == neighbour.calculate_gain2()

            # if neighbour.partition == 0:
            #     print("remove neighbour left with gain", new_gain)
            #     show_buckets(True, left, highest_gain_left)
            # else:
            #     print("remove neighbour right with gain", new_gain)
            #     show_buckets(False, right, highest_gain_right)

            # put it in the new deque with new gain possibly updating the pointer 
            # if neighbour.partition == 0:
            #     left, highest_gain_left = add_to_bucket(neighbour, new_gain, left, highest_gain_left)
            #     #print("add neighbour left with gain ", new_gain)
            #     #show_buckets(True, left, highest_gain_left)   
            # else:
            #     right, highest_gain_right = add_to_bucket(neighbour, new_gain, right, highest_gain_right)
                #print("add neighbour to right with gain ", new_gain)
                #show_buckets(False, right, highest_gain_right)

            # show_buckets(True, left, highest_gain_left)
            # print(f"{highest_gain_left=}")
            # print(f"{highest_gain_right=}")
            # show_buckets(False, right, highest_gain_right)

            # for key in left.keys():
            #     assert len(left[key]) != 0
            # assert sorted(left.keys(), reverse=True)[0] == highest_gain_left
            # for key in right.keys():
            #     assert len(right[key]) != 0
            # assert sorted(right.keys(), reverse=True)[0] == highest_gain_right


        # update the number of crossing edges
        new_cut -= node.gain
        # print(f"{new_cut=}, {graph.count_crossing_edges()=}, {node.gain=},")
        graph.crossing_edges = new_cut
        # assert new_cut == graph.count_crossing_edges2()
        switch_order.append((node, new_cut))

        if new_cut < best_cut:
            best_partition = i
    
    # recover the solution by rolling back all the switches in backwards order until you arrive at best_partition
    # return this partition and the corresponding value
    print("# crossing edges:", [cut[1] for cut in switch_order])
    print("Node switching order:", [cut[0].idx for cut in switch_order[1:]])
    print("Final # crossing edges:", graph.count_crossing_edges())
    print("Best # crossing edges:", np.min([cut[1] for cut in switch_order]))

    return 


def initialize_buckets(graph: Graph):
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

def select_node_max_gain(buckets: defaultdict, highest_gain: int):
    node = buckets[highest_gain].pop()
    # print("size of bucket after remove", len(buckets[highest_gain]))
    new_highest_gain = update_highest_gain_after_pop(buckets, highest_gain)
    # for key in buckets.keys():
    #     assert len(buckets[key]) != 0
    # assert sorted(buckets.keys(), reverse=True)[0] == new_highest_gain
    return node, buckets, new_highest_gain    

def update_highest_gain_after_pop(buckets: defaultdict, highest_gain: int):
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

    # for key in buckets.keys():
    #     assert len(buckets[key]) != 0
    # assert sorted(buckets.keys(), reverse=True)[0] == new_highest_gain
    return new_highest_gain

def remove_from_bucket(node: Node, buckets: defaultdict, highest_gain: int):
    old_gain = node.gain
    # new_highest_gain = highest_gain
    # this checks if the node is still free, if it's free it will be in here
  
    # if old_gain in buckets.keys() and node in buckets[old_gain]:
    buckets[old_gain].remove(node)
        # print("size of bucket after remove", len(buckets[old_gain]))
    new_highest_gain = update_highest_gain_after_remove(buckets, highest_gain, old_gain)

    # for key in buckets.keys():
    #     assert len(buckets[key]) != 0
    # assert sorted(buckets.keys(), reverse=True)[0] == new_highest_gain


    return buckets, new_highest_gain

def update_highest_gain_after_remove(buckets: defaultdict, highest_gain: int, old_gain: int):
    new_highest_gain = highest_gain
    if old_gain == highest_gain:
        new_highest_gain = update_highest_gain_after_pop(buckets, highest_gain)
    else:
        if len(buckets[old_gain]) == 0:
            del buckets[old_gain]
    
    # for key in buckets.keys():
    #     assert len(buckets[key]) != 0
    # assert sorted(buckets.keys(), reverse=True)[0] == new_highest_gain
    return new_highest_gain

def recalculate_gain(switch_node: Node, neighbour_node: Node):
    # print()
    # print(neighbour_node.gain)
    # print(neighbour_node.calculate_gain2())
    new_partition_switch = switch_node.partition
    if neighbour_node.partition == new_partition_switch:
        neighbour_node.gain -= 2
    else:
        neighbour_node.gain += 2

    # assert neighbour_node.gain == neighbour_node.calculate_gain2()

    return neighbour_node.gain

def add_to_bucket(node: Node, new_gain: int, buckets: defaultdict, highest_gain: int):
    # for key in buckets.keys():
    #     assert len(buckets[key]) != 0
    # print()
    # print()
    # print("to add", new_gain)
    # show_buckets(True, buckets, highest_gain)
    # print("bucket exists?", new_gain in buckets.keys())
    # print("true highest bucket", sorted(buckets.keys(), reverse=True)[0])

    buckets[new_gain].append(node)

    # print(f"size of bucket {new_gain} after add", len(buckets[new_gain]))
    # print(f"{highest_gain=}", f"{new_gain=}")

    new_highest_gain = highest_gain
    if new_gain > highest_gain:
        new_highest_gain = new_gain

    # for key in buckets.keys():
    #     assert len(buckets[key]) != 0
    # assert sorted(buckets.keys(), reverse=True)[0] == new_highest_gain
    return buckets, new_highest_gain

# def show_buckets(left: bool, buckets: defaultdict, highest_gain):
#     if left:
#         print("LEFT")
#     else:
#         print("RIGHT")
#     print(sorted(buckets.keys()))
#     for key in sorted(buckets.keys()):
#         if key == highest_gain:
#             print("*" + str(len(buckets[key])), end="  ")
#         else:    
#             print(len(buckets[key]), end="  ")
#     print("\n")

# def free_in_partition(buckets: defaultdict):
#     s = 0
#     for x in buckets.keys():
#         s += len(buckets[x])
#     return s


fm_pass(graph)

