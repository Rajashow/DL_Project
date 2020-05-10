from train import Train
from wann import wann
from ulti import get_pop_fitness, get_pop_rank, get_split, get_pop_loss
import bisect


class Species():
    def __init__(self):
        self.max_fit = 0
        self.mean_fit = 0
        self.species_list = []

    def reap(self, p):
        numb_to_reap = int(len(self.species_list)*p)

        self.species_list.sort(key=get_pop_rank)
        self.species_list = self.species_list[:len(
            self.species_list)-numb_to_reap]

    def speciete(self):
        self.species_list.sort(key=get_pop_loss)
        fitness = [get_pop_loss(s) for s in self.species_list]

        bounds = get_split(fitness)
        split = bisect.bisect(fitness, bounds[1])

        new_Species = Species()

        new_Species.species_list = self.species_list[:split]
        self.species_list = self.species_list[split:]

        new_Species.max_fit = max(fitness[:split])
        new_Species.mean_fit = sum(fitness[:split])/len(fitness[:split])

        self.max_fit = max(fitness[split:])
        self.mean_fit = sum(fitness[split:])/len(fitness[split:])

        return self, new_Species


if __name__ == "__main__":
    wann_class = wann

    hyper_params = {"p_weighed_rank": .5, "w": -2, "%_reap": .5}
    class_args = {"input_dim": (784), "num_classes": 300}
    trainer = Train(wann_class, class_args, 100, hyper_params)
    trainer.populate()
    for i in range(len(trainer.pop)):
        if i > 50:
            trainer.pop[i].fitness = 1/((i+1)+2)
        else:
            trainer.pop[i].fitness = 1/(i+1)
    sp = Species()
    sp.species_list = trainer.pop
    sp.speciete()
