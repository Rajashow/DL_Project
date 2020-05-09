import torch
from ranker import nsga_sort, rank_array
import numpy as np


class Train():
    def __init__(self, wann_class, init_class_args, n_pop, hyper_params):
        self.pop = []
        self.ind = wann_class
        self.init_class_args = init_class_args
        self.n_pop = n_pop
        self.sep = None
        self.hyp = hyper_params
        self._reap_per_gen = int(self.n_pop*self.hyp["%_reap"])

        self._sample_p = 1/((np.arange(2, self.n_pop+1, 2))**2)

        self.gen = 0

    def populate(self):
        self.pop = [self.ind(self.init_class_args)] * self.n_pop

    def mutate(self):

        n_samples = self.n_pop-len(self.pop)
        sample_idxs = np.random.choice(
            n_samples, len(self.pop), p=self._sample_p[:n_samples])

        for index in sample_idxs:
            self.pop.append(self.pop[index].mutate())

    def give_rank(self):

        mean_fit = torch.Tensor([ind.fitness for ind in self.pop])
        n_conns = torch.Tensor([ind.n_conn for ind in self.pop])

        # No connections is pareto optimal but boring...
        n_conns[n_conns == 0] = 1
        obj_vals = torch.cat((mean_fit, 1/n_conns), 1)  # Maximize

        # Alternate between two objectives and single objective
        if self.hyp['p_weighed_rank'] < torch.rand():
            # rank based on loss and connections
            rank, _ = nsga_sort(obj_vals[:, [0, 1]])
        else:  # rank based on loss
            rank = rank_array(-obj_vals[:, 0])

        # Assign ranks
        for (pop, rank) in zip(self.pop, rank):
            pop.rank = rank

    def reap(self):
        def get_pop_rank(pop): return pop.rank
        self.pop.sort(key=get_pop_rank)
        self.pop = self.pop[:len(self.pop)-self._reap_per_gen]

    def train(self, x, y, loss):
        for pop in self.pop:
            y_ = pop.forward(x, self.hyp["w"])
            pop.fitness = -loss(y_, y)

    def iterate(self, x, y, loss):
        if self.pop:
            self.populate()
        else:
            self.mutate()
        self.train(x, y, loss)
        self.give_rank()
        self.reap()

        self.gen += 1
