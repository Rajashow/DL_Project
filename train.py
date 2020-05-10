import matplotlib.pyplot as plt
import torch
from ranker import nsga_sort, rank_array
import numpy as np
from wann import wann


class Train():
    def __init__(self, wann_class, init_class_args, n_pop, hyper_params):
        self.pop = []
        self.ind = wann_class
        self.init_class_args = init_class_args
        self.n_pop = n_pop
        self.sep = None
        self.hyp = hyper_params
        self._reap_per_gen = int(self.n_pop*self.hyp["%_reap"])

        self._sample_p = 1/((2*np.arange(1, self.n_pop+1))**2)

        self.gen = 0
        self.history = []

    def populate(self):
        self.pop = [self.ind(**self.init_class_args)] * self.n_pop

    def mutate(self):
        if len(self.pop) != self.n_pop:
            n_samples = self.n_pop-len(self.pop)
            p = self._sample_p[:n_samples]
            p /= np.sum(p)
            sample_idxs = np.random.choice(
                n_samples, len(self.pop), p=p)

            for index in sample_idxs:
                self.pop.extend(self.pop[index].mutate(1))

    def give_rank(self):

        mean_fit = np.asarray([ind.fitness for ind in self.pop])
        n_conns = np.asarray([ind.edge_count() for ind in self.pop])

        # No connections is pareto optimal but boring...
        n_conns[n_conns == 0] = 1

        obj_vals = np.c_[mean_fit, 1/n_conns]  # Maximize
        # to do rank fix
        if self.hyp['p_weighed_rank'] < np.random.rand():
            rank = nsga_sort(obj_vals[:, [0, 1]])
        else:  # Single objective
            rank = rank_array(-obj_vals[:, 0])

        # Assign ranks
        # for i in range(len(self.pop)):
        #     self.pop[i].rank = rank[i]

        # Assign ranks
        for (pop, rank) in zip(self.pop, rank):
            pop.rank = rank

    def reap(self):
        def get_pop_rank(pop): return pop.rank
        self.pop.sort(key=get_pop_rank)
        self.pop = self.pop[:len(self.pop)-self._reap_per_gen]

    def train(self, x, y, loss, print_fit=False):
        mean_fitnesses = 0
        for pop in self.pop:
            y_ = pop.forward(x, self.hyp["w"])
            pop.fitness = 1/loss(y_, y)
            mean_fitnesses += pop.fitness
        mean_fitnesses /= self.n_pop

        self.history.append(mean_fitnesses)
        if print_fit:
            print(f"#{self.gen+1} mean fitness {mean_fitnesses}")

    def iterate(self, x, y, loss):
        if not self.pop:
            self.populate()
            self.mutate()
        else:
            self.mutate()
        self.train(x, y, loss)
        self.give_rank()
        self.reap()

        self.gen += 1

    def plot_fitness(self):
        plt.figure()
        plt.plot(self.history, label="fitness")
        plt.xlabel("Gen")
        plt.ylabel("Fitness (1/loss)")
        plt.title("Mean Fitness")
        plt.show()


if __name__ == "__main__":
    wann_class = wann

    hyper_params = {"p_weighed_rank": .5, "w": -2, "%_reap": .5}
    class_args = {"input_dim": (784), "num_classes": 300}
    trainer = Train(wann_class, class_args, 100, hyper_params)

    x = torch.rand((10, 784))
    y = torch.randint(300, (10,))
    loss = torch.nn.CrossEntropyLoss()
    for i in range(50):
        trainer.iterate(x, y, loss)
        # trainer.pop[0].visualize()
    trainer.plot_fitness()
