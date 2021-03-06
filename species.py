from utils import get_pop_fitness, get_pop_rank, get_split, get_pop_loss
import bisect


class Species():
    def __init__(self, pops):
        """Create a species with a given population

        Arguments:
            pops {list} -- list of population
        """
        self.max_fit = 0
        self.mean_fit = 0
        self.species_list = [pop for pop in pops]

        for pop in self.species_list:
            pop.species = self

    def reap(self, p):
        """reaps a percents of the pop based on fitness

        Arguments:
            p {float} -- percent of pop to reap
        """
        numb_to_reap = int(len(self.species_list)*p)

        self.species_list.sort(key=get_pop_rank)
        self.species_list = self.species_list[:len(
            self.species_list)-numb_to_reap]

    def get_fitness(self):
        """return the fitness of all species

        Returns:

            list -- list of fitness for all of species(type float)
        """
        return [get_pop_fitness(s) for s in self.species_list]

    def speciate(self):
        """split a species into two based on fitness

        Returns:
            tuple -- tuple of 2 species(self, new species)
        """
        self.species_list.sort(key=get_pop_loss)
        fitness = self.get_fitness()

        bounds = get_split(fitness)
        split = bisect.bisect(fitness, bounds[1])

        new_Species = Species([])
        split = split if len(fitness) != split and split else len(fitness)//2

        new_Species.species_list = self.species_list[:split]
        self.species_list = self.species_list[split:]

        new_Species.max_fit = max(fitness[:split])
        new_Species.mean_fit = sum(fitness[:split])/len(fitness[:split])

        self.max_fit = max(fitness[split:])
        self.mean_fit = sum(fitness[split:])/len(fitness[split:])

        for sep in self.species_list:
            sep.species = self
        for sep in new_Species.species_list:
            sep = new_Species

        return [self, new_Species]
