import numpy as np

import jenkspy


def get_split(loss):
    return jenkspy.jenks_breaks(loss, nb_class=2)


def get_pop_rank(pop): return pop.rank


if __name__ == "__main__":
    pass
