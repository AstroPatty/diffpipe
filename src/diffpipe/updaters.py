import numpy as np


def local_index_updater(sources, full_map):
    running_sum = 0
    output_data = []
    for source in sources:
        data = source[:]
        data = full_map[data + running_sum]
        output_data.append(data)
        running_sum += len(data)

    return np.concat(output_data)


UPDATERS = {"top_host_idx": local_index_updater}
