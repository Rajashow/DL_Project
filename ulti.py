import numpy as np

import jenkspy

import functools
import operator

from unidip import UniDip


def should_speciate(loss):
    intervals = UniDip(loss).run()
    return len(intervals) != 1


def get_split(loss):
    """given a list of values find the range of value that creates a split.

    Arguments:
        loss {list/iterable} -- list of losses to split

    Returns:
        list -- a list with a lower bound, break point and upper bound
    """
    return jenkspy.jenks_breaks(loss, nb_class=2)


def get_pop_rank(pop): return pop.rank


def get_pop_fitness(pop): return pop.fitness


def get_pop_loss(pop): return 1/pop.fitness


def functools_reduce_iconcat(a):
    """flatten python lists fast

    Arguments:
        a {list of lists} -- list

    Returns:
        list -- flattend list
    """
    return functools.reduce(operator.iconcat, a, [])


if __name__ == "__main__":
    pass
