from train import Train
from wann import wann
from ulti import get_pop_rank, get_split
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
        self.species_list.sort(key=get_pop_rank)
        fitness = [1/s.fitness for s in self.species_list]

        bounds = get_split(fitness)
        split = bisect.bisect_right(fitness, bounds[1])
        new_Species = Species()

        new_Species.species_list = self.species_list[:split+1]
        self.species_list = self.species_list[split+1:]

        new_Species.max_fit = max(fitness[:split+1])
        new_Species.mean_fit = sum(fitness[:split+1])/len(fitness[:split+1])

        self.max_fit = max(fitness[split+1:])
        self.mean_fit = sum(fitness[split+1:])/len(fitness[split+1:])

        return self, new_Species


if __name__ == "__main__":
    wann_class = wann

    hyper_params = {"p_weighed_rank": .5, "w": -2, "%_reap": .5}
    class_args = {"input_dim": (784), "num_classes": 300}
    trainer = Train(wann_class, class_args, 100, hyper_params)

    for i in range(len(trainer.pop)):
        if i > 50:
            trainer.pop[i].fitness = i*20
        else:
            trainer.pop[i].fitness = i
    sp = Species()
    sp.species_list = trainer.pop
    sp.speciete()
