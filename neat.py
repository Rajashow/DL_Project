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
        self.species = None
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
            self.pop[i].nConn = self.pop[i].nConn

    def initPop(self):
        """
        Initialize population with a list of random individuals
        """
        # Create base individual
        p = self.hyp  # readability

        # - Create Nodes -
        nodeId = torch.arange(0, p['ann_nInput'] + p['ann_nOutput']+1, 1)
        node = torch.empty((3, len(nodeId)))
        node[0, :] = nodeId

        # Node types: [1:input, 2:hidden, 3:bias, 4:output]
        node[1, 0] = 4  # Bias
        node[1, 1:p['ann_nInput']+1] = 1  # Input Nodes
        node[1, (p['ann_nInput']+1):
             (p['ann_nInput']+p['ann_nOutput']+1)] = 2  # Output Nodes

        # Node Activations
        node[2, :] = p['ann_initAct']
        # - Create Conns -
        nConn = (p['ann_nInput']+1) * p['ann_nOutput']
        # Input and Bias Ids
        ins = torch.arange(0, p['ann_nInput']+1, 1)
        outs = (p['ann_nInput']+1) + torch.arange(0,
                                                  p['ann_nOutput'])  # Output Ids

        conn = torch.empty((5, nConn,))
        conn[0, :] = torch.arange(0, nConn, 1)      # Connection Id
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
                2*(torch.rand(1, nConn)-0.5))*p['ann_absWCap']
            newInd.conn[4, :] = torch.rand(1, nConn) < p['prob_initEnable']
            newInd.express()
            newInd.birth = 0
            pop.append(copy.deepcopy(newInd))
        # - Create Innovation Record -
        innov = torch.zeros([5, nConn])
        innov[0:3, :] = pop[0].conn[0:3, :]
        innov[3, :] = -1

        self.pop = pop
        self.innov = innov

    def probMoo(self):
        """
        Rank population according to Pareto dominance.
        """
        # Compile objectives
        meanFit = torch.Tensor([ind.fitness for ind in self.pop])
        nConns = torch.Tensor([ind.nConn for ind in self.pop])
        # No connections is pareto optimal but boring...
        nConns[nConns == 0] = 1
        objVals = torch.cat((meanFit, 1/nConns), 1)  # Maximize

        # Alternate between two objectives and single objective
        if self.hyp['alg_probMoo'] < torch.rand():
            rank, _ = nsga_sort(objVals[:, [0, 1]])
        else:  # Single objective
            rank = rankArray(-objVals[:, 0])

        # Assign ranks
        for i in range(len(self.pop)):
            self.pop[i].rank = rank[i]
