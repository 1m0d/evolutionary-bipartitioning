import numpy as np

from graph import GRAPH_SIZE


def child_generator(number_of_0_needed):
    count = 0
    while True:
        if count < number_of_0_needed:
            yield 0
        else:
            yield 1
        count += 1


def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    hamming_distance = np.average(parent1 != parent2)

    if hamming_distance > GRAPH_SIZE / 2:
        parent1 = np.invert(parent1)

    empty_positions = []
    child = np.array(GRAPH_SIZE)
    for i in range(GRAPH_SIZE):
        if parent1[i] == parent2[i]:
            child[i] = parent1[i]
        else:
            empty_positions.append(i)

    if not empty_positions:
        return child

    count_0 = np.count_nonzero(child == 0)
    number_of_0_needed = GRAPH_SIZE / 2 - count_0
    child_gen = child_generator(number_of_0_needed)

    np.random.shuffle(empty_positions)
    for empty_index in empty_positions:
        child[empty_index] = next(child_gen)

    return child
