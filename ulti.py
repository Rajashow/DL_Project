import numpy as np

import jenkspy


def get_split(loss):
    return jenkspy.jenks_breaks(loss, nb_class=2)


def get_pop_rank(pop): return pop.rank


def get_pop_fitness(pop): return pop.fitness


def get_pop_loss(pop): return 1/pop.fitness


if __name__ == "__main__":
    pass
