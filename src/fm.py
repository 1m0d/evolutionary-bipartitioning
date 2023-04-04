from src.graph import GRAPH_SIZE


class GainBuckets:
    def __init__(self):
        self.buckets = {}

    def insert(self, gain, node):
        if gain in self.buckets:
            self.buckets[gain].add(node)
        else:
            self.buckets[gain] = {node}

    def delete(self, gain, node):
        self.buckets[gain].remove(node)
        if not self.buckets[gain]:
            del self.buckets[gain]

    def get_max_gain(self):
        max_gain = max(self.buckets.keys())
        return max_gain

    def pop_node_with_max_gain(self):
        max_gain = self.get_max_gain()
        node = self.buckets[max_gain].pop()
        if not self.buckets[max_gain]:
            del self.buckets[max_gain]
        return node, max_gain


def fm_pass(graph, verbose=False):
    # Initialize gain buckets for each partition
    gain_buckets_0 = GainBuckets()
    gain_buckets_1 = GainBuckets()

    # Calculate initial gains and populate gain buckets
    for node in graph.nodes.values():
        gain = node.calculate_gain()
        if node.partition == 0:
            gain_buckets_0.insert(gain, node)
        else:
            gain_buckets_1.insert(gain, node)

    move_history = []
    best_move_index = -1
    best_crossing_edges = current_crossing_edges = graph.count_crossing_edges()

    for i in range(GRAPH_SIZE):
        if i % 2 == 0:
            node, max_gain = gain_buckets_0.pop_node_with_max_gain()
        else:
            node, max_gain = gain_buckets_1.pop_node_with_max_gain()

        node.switch_partition()
        move_history.append(node)
        current_crossing_edges -= max_gain
        if i % 2 == 1:
            if current_crossing_edges <= best_crossing_edges:
                best_crossing_edges = current_crossing_edges
                best_move_index = i

        # Update gains for neighboring nodes
        for neighbor in node.adjacent:
            if neighbor.switched:
                continue

            old_gain = neighbor.gain
            new_gain = neighbor.gain = (
                old_gain - 2 if neighbor.partition == node.partition else old_gain + 2
            )

            # Update the gain buckets
            if neighbor.partition == 0:
                gain_buckets_0.delete(old_gain, neighbor)
                gain_buckets_0.insert(new_gain, neighbor)
            else:
                gain_buckets_1.delete(old_gain, neighbor)
                gain_buckets_1.insert(new_gain, neighbor)

    # Rollback to the best graph
    for i in range(GRAPH_SIZE - 1, best_move_index, -1):
        move_history[i].switch_partition()

    for node in graph.nodes.values():
        node.switched = False

    return graph
