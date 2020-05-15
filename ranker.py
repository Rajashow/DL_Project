import torch
import torch.functional
import numpy as np

import numpy as np
import warnings


def nsga_sort(obj_vals, return_fronts=False):
    """Returns ranking of objective values based on non-dominated sorting.
    Optionally returns fronts (useful for visualization).

    NOTE: Assumes maximization of objective function

    Args:
      obj_vals - (np_array) - Objective values of each individual
                [nInds X nObjectives]

    Returns:
      rank    - (np_array) - Rank in population of each individual
              int([nIndividuals X 1])
      front   - (np_array) - Pareto front of each individual
              int([nIndividuals X 1])

    Todo:
      * Extend to N objectives
    """
    # Sort by dominance into fronts
    fronts = get_fronts(obj_vals)

    # Rank each front by crowding distance
    for f in range(len(fronts)):
        x1 = obj_vals[fronts[f], 0]
        x2 = obj_vals[fronts[f], 1]
        crowd_dist = get_crowding_dist(x1) + get_crowding_dist(x2)
        front_rank = np.argsort(-crowd_dist)
        fronts[f] = [fronts[f][i] for i in front_rank]

    # Convert to ranking
    tmp = [ind for front in fronts for ind in front]
    rank = np.empty_like(tmp)
    rank[tmp] = np.arange(len(tmp))

    if return_fronts is True:
        return rank, fronts
    else:
        return rank


def get_fronts(obj_vals):
    """Fast non-dominated sort.

    Args:
      obj_vals - (np_array) - Objective values of each individual
                [nInds X nObjectives]

    Returns:
      front   - [list of lists] - One list for each front:
                                  list of indices of individuals in front

    Todo:
      * Extend to N objectives

    [adapted from: https://github.com/haris989/NSGA-II]
    """

    values1 = obj_vals[:, 0]
    values2 = obj_vals[:, 1]

    S = [[] for i in range(len(values1))]
    front = [[]]
    n = [0]*len(values1)
    rank = [0]*len(values1)

    # Get dominance relations
    for p in range(len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) \
                    or (values1[p] >= values1[q] and values2[p] > values2[q]) \
                    or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) \
                    or (values1[q] >= values1[p] and values2[q] > values2[p]) \
                    or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if not n[p]:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    # Assign fronts
    i = 0
    while(front[i]):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if(n[q] == 0):
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)
    del front[len(front)-1]
    return front


def get_crowding_dist(obj_vector):
    """Returns crowding distance of a vector of values, used once on each front.

    Note: Crowding distance of individuals at each end of front is infinite, as
    they don't have a neighbor.

    Args:
      obj_vector - (np_array) - Objective values of each individual
                  [nInds X nObjectives]

    Returns:
      dist      - (np_array) - Crowding distance of each individual
                  [nIndividuals X 1]
    """
    # Order by objective value
    key = np.argsort(obj_vector)
    sorted_obj = obj_vector[key]

    # Distance from values on either side
    # Edges have infinite distance
    shift_vec = np.r_[np.inf, sorted_obj, np.inf]
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning)  # inf on purpose
    prev_dist = np.abs(sorted_obj-shift_vec[:-2])
    next_dist = np.abs(sorted_obj-shift_vec[2:])
    crowd = prev_dist+next_dist
    if (sorted_obj[-1]-sorted_obj[0]) > 0:
        # Normalize by fitness range
        crowd *= abs((1/sorted_obj[-1]-sorted_obj[0]))

    # Restore original order
    dist = np.empty(len(key))
    dist[key] = crowd[:]

    return dist


def rank_array(obj):
    """rank an array based of the first objective

    Arguments:
        obj {list} -- list of objective

    Returns:
        list -- list of ranks
    """
    tmp = np.argsort(obj)
    rank = np.empty_like(tmp)
    rank[tmp] = np.arange(len(obj))
    return rank
