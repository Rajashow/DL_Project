from ulti import get_pop_rank, get_split
import bisect


class Species():
    def __init__(self):
        self.max_fit = 0
        self.mean_fit = 0
        self.species_list = []
        self.parent = []

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
        new_Species = self.species_list[:split+1]
