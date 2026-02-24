import numpy as np


def local_index_updater(sources, full_map):
    running_sum = 0
    output_data = []
    destination_map = np.argsort(full_map)
    for source in sources:
        data = source[:]
        data = destination_map[data + running_sum]
        output_data.append(data)
        running_sum += len(data)

    output = np.concat(output_data)
    return output


UPDATERS = {"top_host_idx": local_index_updater}
