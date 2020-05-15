import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from ranker import nsga_sort, rank_array
from scipy.ndimage.filters import gaussian_filter1d
from sketchdataset import SketchDataSet
from species import Species
from torch.utils.data import DataLoader
from utils import get_pop_rank, functools_reduce_iconcat, should_speciate
from wann import wann, wannModel


class Train():
    def __init__(self, wann_class, init_class_args, n_pop, hyper_params):
        """Create a class that trains the NN and finds a WANNs for you.

        Arguments:
            wann_class {class} -- class of an individual WANN
            init_class_args {dict} -- dict that can be passed into the WANN for init
            n_pop {int} -- size of the training population
            hyper_params {dict} -- training hyper param:
                                    must have:
                                    - "%_reap": percent of pop that is kill in each gen
                                    - "w" : deafult weight of WANN NN
                                    - "p_weighed_rank": percent of times you try to maximize fitness
                                        for number of connetions
        """
        self.pop = []
        self.ind = wann_class
        self.init_class_args = init_class_args
        self.n_pop = n_pop
        self.hyp = hyper_params
        self._reap_per_gen = int(self.n_pop*self.hyp["%_reap"])

        self._sample_p = 2*np.arange(self.n_pop+1, 0, -1)

        self.gen = 0
        self.history = []
        self.running_fitness = 0

        self.sep = []

    def populate(self):
        """
        Create a population of max size allowed with the init param that was passed
        """
        self.pop = [self.ind(**self.init_class_args)
                    for i in range(self.n_pop)]
        self.sep.append(Species(self.pop))

    def replace_reaped_with_mutated(self):
        """
        Based on replace the reaped population with mutation of surviving population
        """
        # if the pop isn't full
        if len(self.pop) != self.n_pop:
            # get the difference
            n_samples = self.n_pop-len(self.pop)
            # get the pop of having them
            p = self._sample_p[:n_samples]
            # standardized
            p = p / np.sum(p)
            # sample based on the pop with replacement
            sample_idxs = np.random.choice(
                n_samples, len(self.pop), replace=True, p=p)
            # create pop
            for index in sample_idxs:
                self.pop.extend(self.pop[index].mutate(1))
                self.pop[-1].species.species_list.append(self.pop[-1])

    def give_rank(self):
        """
        For each wann in the population give them a rank.
        """
        mean_fit = np.asarray([ind.fitness for ind in self.pop])
        n_conns = np.asarray([ind.edge_count() for ind in self.pop])

        # No connections is pareto optimal can useless
        n_conns[n_conns == 0] = 1

        # objectives to Maximize
        obj_vals = np.c_[mean_fit, 1/n_conns]

        if self.hyp['p_weighed_rank'] < np.random.rand():
            rank = nsga_sort(obj_vals[:, [0, 1]])
        else:  # Single objective
            rank = rank_array(-obj_vals[:, 0])

        # Assign ranks
        for (pop, rank) in zip(self.pop, rank):
            pop.rank = rank

    def reap(self):
        """
        Reap/kill the % of population that didn't perform well
        """
        self.pop = []
        for sep in self.sep:
            sep.reap(self.hyp["%_reap"])
            self.pop.extend(sep.species_list)
        self.pop.sort(key=get_pop_rank)
        if len(self.pop) > self.n_pop:
            self.pop = self.pop[:self.n_pop]

    def _speciate(self):
        n_sep = []
        for sep in self.sep:
            if should_speciate(sep.get_fitness()):
                n_sep.extend(sep.speciate())
            else:
                n_sep.append(sep)
        self.sep = n_sep

    def train(self, x, y, loss, print_fit=False):
        """For each wann perfom a forward pass and get it's fitness

        Arguments:
            x {torch_tensor} -- x inputs
            y {torch_tensor} -- y target
            loss {func} -- used to get the loss of the predicted and the inverse of this is used as fitness

        Keyword Arguments:
            print_fit {bool} -- print the fitness of a given gen (default: {False})
        """
        mean_fitnesses = 0
        for pop in self.pop:
            y_ = pop.forward(x, self.hyp["w"])
            pop.fitness = (1/loss(y_, y)).item()
            mean_fitnesses += pop.fitness

        mean_fitnesses /= self.n_pop

        # track fitness
        self.running_fitness += mean_fitnesses
        self.history.append(mean_fitnesses)

        if print_fit:
            print(f"#{self.gen+1} mean fitness {mean_fitnesses}")

    def _self_mutate(self):
        """
        replace all of the population with it's own mutation
        """
        for i in range(len(self.pop)):
            self.pop[i] = self.pop[i].mutate(1)[0]

        self.sep = []
        self.sep.append(Species(self.pop))

    def iterate(self, x, y, loss, init_mutate=1_000_000):
        """Primary method used to interface with wanns

        Arguments:
            x {torch_tensor} -- input
            y {torch_tensor} -- target
            loss {func} -- loss function
            init_mutate {int} -- number of mutation from init for init pop. (default: {100})
        """
        if not self.pop:
            print("Creating new population")
            self.populate()
            for _ in range(init_mutate):
                self._self_mutate()
            print("Done creating a population")
        else:
            self.replace_reaped_with_mutated()
        self.train(x, y, loss)
        self.give_rank()
        self._speciate()
        self.reap()

        self.gen += 1

    def plot_fitness(self, smooth=0):
        """plot the losses for each iter

        Keyword Arguments:
            smooth {int} -- power level for smoothing must be odd (default: {0})
        """
        plt.figure()
        if smooth:
            plt.plot(gaussian_filter1d(self.history, smooth), label="fitness")
        else:
            plt.plot(self.history, label="fitness")
        plt.xlabel("Gen")
        plt.ylabel("Fitness (1/loss)")
        plt.title("Mean Fitness")
        plt.show()

    def visualize_sample(self, shape=(5, 5), plot_args=None):
        """Create a plot grid with the size shape where each cell has a wann network

        Keyword Arguments:
            shape {tuple} -- the shape of grid (default: {(5, 5)})
            plot_args {dict} -- not implemented (default: {None})
        """

        w, h = shape
        samples = np.random.choice(
            self.pop, w*h, replace=len(self.pop) < (w*h))
        fig, axs = plt.subplots(*shape, figsize=(w*10, h*10))

        for ax, pop in zip(functools_reduce_iconcat(axs), samples):
            pop.visualize({"ax": ax})
            ax.set_title(f"#{pop.rank} with {pop.fitness:.3f}")
        plt.show()


def get_parser():
    parser = argparse.ArgumentParser(description='WANN training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--name', type=str, required=False,
                        default="default-experiment")
    parser.add_argument('--pweighedrank', type=float, default=0.5)
    parser.add_argument('--sharedweight', type=float, default=-2.0)
    parser.add_argument('--reap', type=float, default=0.5)
    parser.add_argument('--populationsize', type=int, default=100)
    return parser


def parse_args(args_dict):
    num_epochs = args_dict['epochs']
    generations = args_dict['generations']
    batch_size = args_dict['batch']
    experiment_name = args_dict['name']
    hyper_params = {"p_weighed_rank": args_dict["pweighedrank"],
                    "w": args_dict["sharedweight"], "%_reap": args_dict["reap"]}
    population_size = args_dict["populationsize"]

    for key in args_dict.keys():
        print('parsed {} for {}'.format(args_dict[key], key))

    return num_epochs, batch_size, experiment_name, hyper_params, population_size, generations


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    num_epochs, batch_size, experiment_name, hyper_params, population_size, generations = parse_args(
        vars(args))

    wann_class = wann

    class_args = {"input_dim": (784), "output_dim": 300}
    trainer = Train(wann_class, class_args, population_size, hyper_params)

    train_dataset = SketchDataSet("./data/", is_train=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print('Loaded %d train images' % len(train_dataset))
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = torch.nn.CrossEntropyLoss()
    trainer.populate()

    with torch.no_grad():
        for _ in range(generations):
            for _, (images, target) in enumerate(train_loader):
                images = images.reshape(batch_size, 784).to(device)
                target = target.to(device)
                trainer.iterate(images, target, criterion)
                if trainer.gen > generations:
                    break

    wann.save_json(trainer.pop[0], f"best_wann_model")

    net = wannModel(trainer.pop[0]).cuda()
    optimizer = torch.optim.Adam(net.parameters(), .003)
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, target) in enumerate(train_loader):
            images, target = images.to(device), target.to(device)
            images = images.reshape(batch_size, 784).to(device)
            target = target.to(device)
            pred = net(images)
            loss = criterion(pred, target)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"{epoch}: {total_loss/(i+1)}")
    torch.save(net.state_dict(), "latest_best_model.pth")
