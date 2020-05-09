import numpy as np
import copy
import json
import torch
from ranker import nsga_sort, rankArray


class Neat():
    """
    NEAT main class. Evolves population given fitness values of individuals.
    """

    def __init__(self, hyp, individual):
        """
        Initialize NEAT algorithm with hyperparameters
        Args:
          hyp - (dict) - algorithm hyperparameters
          individual - (class) - individual class with genes, network, and fitness
        Attributes:
          p       - (dict)     - algorithm hyperparameters
          pop     - (Ind)      - Current population
          species - (Species)  - Current species
          innov   - (torch_tensor) - innovation record
                    [5 X nUniqueGenes]
                    [0,:] == Innovation Number
                    [1,:] == Source
                    [2,:] == Destination
                    [3,:] == New Node?
                    [4,:] == Generation evolved
          gen     - (int)      - Current generation
        """
        self.hyp = hyp
        self.pop = []
        self.species = []
        self.innov = None
        self.gen = 0
        self.indType = individual

    def ask(self):
        """
        Returns newly evolved population
        """
        if not self.pop:
            self.initPop()      # Initialize population
        else:
            self.probMoo()      # Rank population according to objectives
            raise NotImplementedError("need to implement speciate")
            self.speciate()     # Divide population into species
            raise NotImplementedError("need to implement evolvePop")
            self.evolvePop()    # Create child population

        return self.pop       # Send child population for evaluation

    def tell(self, reward):
        """
        Assigns fitness to current population

        Args:
          reward - (torch_tensor) - fitness value of each individual
                   [nInd X 1]

        """
        for i in range(np.shape(reward)[0]):
            self.pop[i].fitness = reward[i]
            self.pop[i].n_conn = self.pop[i].n_conn

    def initPop(self):
        """
        Initialize population with a list of random individuals
        """
        # Create base individual
        p = self.hyp  # readability

        # - Create Nodes -
        nodeId = torch.arange(
            0, self.hyp['ann_nInput'] + self.hyp['ann_nOutput']+1, 1)
        node = torch.empty((3, len(nodeId)))
        node[0, :] = nodeId

        # Node types: [1:input, 2:hidden, 3:bias, 4:output]
        node[1, 0] = 4  # Bias
        node[1, 1:p['ann_nInput']+1] = 1  # Input Nodes
        node[1, (p['ann_nInput']+1):
             (p['ann_nInput']+p['ann_nOutput']+1)] = 2  # Output Nodes

        # Node Activations
        node[2, :] = self.hyp['ann_initAct']
        # - Create Conns -
        n_conn = (p['ann_nInput']+1) * self.hyp['ann_nOutput']
        # Input and Bias Ids
        ins = torch.arange(0, self.hyp['ann_nInput']+1, 1)
        outs = (p['ann_nInput']+1) + torch.arange(0,
                                                  self.hyp['ann_nOutput'])  # Output Ids

        conn = torch.empty((5, n_conn,))
        conn[0, :] = torch.arange(0, n_conn, 1)      # Connection Id
        conn[1, :] = ins.repeat(len(outs))  # Source Nodes
        conn[2, :] = torch.repeat_interleave(
            outs, len(ins), dim=1)  # Destination Nodes
        conn[3, :] = float("nan")                   # Weight Values
        conn[4, :] = 1                         # Enabled?

        # Create population of individuals with varied weights
        pop = []
        for _ in range(p['popSize']):
            newInd = self.indType(conn, node)
            newInd.conn[3, :] = (
                2*(torch.rand(1, n_conn)-0.5))*p['ann_absWCap']
            newInd.conn[4, :] = torch.rand(
                1, n_conn) < self.hyp['prob_initEnable']
            newInd.express()
            newInd.birth = 0
            pop.append(copy.deepcopy(newInd))
        # - Create Innovation Record -
        innov = torch.zeros([5, n_conn])
        innov[0:3, :] = pop[0].conn[0:3, :]
        innov[3, :] = -1

        self.pop = pop
        self.innov = innov

    def probMoo(self):
        """
        Rank population according to Pareto dominance.
        """
        # Compile objectives
        mean_fit = torch.Tensor([ind.fitness for ind in self.pop])
        n_conns = torch.Tensor([ind.n_conn for ind in self.pop])
        # No connections is pareto optimal but boring...
        n_conns[n_conns == 0] = 1
        obj_vals = torch.cat((mean_fit, 1/n_conns), 1)  # Maximize

        # Alternate between two objectives and single objective
        if self.hyp['alg_probMoo'] < torch.rand():
            rank, _ = nsga_sort(obj_vals[:, [0, 1]])
        else:  # Single objective
            rank = rankArray(-obj_vals[:, 0])

        # Assign ranks
        for i in range(len(self.pop)):
            self.pop[i].rank = rank[i]

    def _assignSpecies(self):
        """Assigns each member of the population to a species.
        Fills each species class with nearests members, assigns a species Id to each
        individual for record keeping
        Args:
            species - (Species) - Previous generation's species
            .seed       - (Ind) - center of species
            pop     - [Ind]     - unassigned individuals
            p       - (Dict)    - algorithm hyperparameters
        Returns:
            species - (Species) - This generation's species
            .seed       - (Ind) - center of species
            .members    - [Ind] - parent population
            pop     - [Ind]     - individuals with species ID assigned
        """

        # Get Previous Seeds
        if len(self.species):
            # Remove existing members
            for iSpec in range(len(self.species)):
                self.species[iSpec].members = []

        else:
            # Create new species if none exist
            self.species = [self.pop[0].new_species()]
            self.species[0].nOffspring = self.hyp['popSize']
            iSpec = 0

        # Assign members of population to first species within compat distance
        for i in range(len(self.pop)):
            assigned = False
            for iSpec in range(len(self.species)):
                ref = self.species[iSpec].seed.conn.clone()
                ind = self.pop[i].conn.clone()
                cDist = self.compatDist(ref, ind)
                if cDist < self.hyp['spec_thresh']:
                    self.pop[i].species = iSpec
                    self.species[iSpec].members.append(self.pop[i])
                    assigned = True
                    break

                # If no seed is close enough, start your own species
                if not assigned:
                    self.pop[i].species = iSpec+1
                    self.species.append(self.pop[i].new_species())

    def speciate(self):
        """
        Divides population into species and assigns each a number of offspring/
        """

        # Adjust species threshold to track desired number of species
        if len(self.species) > self.hyp['spec_target']:
        self.hyp['spec_thresh'] += self.hyp['spec_compatMod']

        if len(self.species) < self.hyp['spec_target']:
        self.hyp['spec_thresh'] -= self.hyp['spec_compatMod']

        self.hyp['spec_thresh'] = max(
            self.hyp['spec_thresh'], self.hyp['spec_threshMin'])

        self._assignSpecies()
        raise NotImplemented("_assignOffspring")
        self._assignOffspring()

    def evolvePop(self):
        pass
