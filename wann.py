import json
import networkx as nx
import matplotlib.pyplot as plt
import random
import torch
from torch import softmax
import torch.nn as nn
import torch.nn.functional as F

from networkx.algorithms.dag import topological_sort
from networkx.readwrite import json_graph

from CustomizedLinear import CustomizedLinear


class wann:
    def __init__(self, input_dim, output_dim):
        """
        Initializes a WANN with no connections.
        Each input node is of the form "ix".
        Each output node is of the form "ox".
        Each hidden node is an integer x.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = 0
        self.g = nx.DiGraph()
        self.activations = {}
        self.fitness = None
        self.rank = None
        self.species = None

        for i in range(input_dim):
            v = "i" + str(i)
            self.g.add_node(v)
            self.activations[v] = None
        for i in range(output_dim):
            v = "o" + str(i)
            self.g.add_node(v)
            self.activations[v] = None

    def edge_count(self):
        return self.g.size()

    def copy(self):
        """
        Copies WANN.
        """
        copy = wann(self.input_dim, self.output_dim)
        copy.hidden = self.hidden
        copy.g = self.g.copy()
        copy.activations = dict(self.activations)
        return copy

    def add_node(self, u, v, activation):
        """
        Splits the edge u->v to u->x->v and sets activation of node x.
        """
        x = self.hidden
        self.hidden += 1
        self.g.add_node(x)
        self.g.remove_edge(u, v)
        self.g.add_edge(u, x)
        self.g.add_edge(x, v)
        self.activations[x] = activation

    def add_edge(self, u, v):
        """
        Adds edge u->v.
        """
        self.g.add_edge(u, v)

    def change_activation(self, v, activation):
        """
        Changes activation of node v.
        """
        self.activations[v] = activation

    def mutate(self, num_children):
        """
        Returns a list of children.
        """
        activations = [torch.sigmoid, F.relu, torch.tanh]
        children = []
        edges = list(nx.edges(self.g))
        non_edges = []

        input_nodes = set("i" + str(i) for i in range(self.input_dim))
        for i in range(self.input_dim):
            v = "i" + str(i)
            for w in set(nx.non_neighbors(self.g, v)) - input_nodes:
                non_edges.append((v, w))
        for i in range(self.hidden):
            for w in set(nx.non_neighbors(self.g, i)) - input_nodes:
                non_edges.append((i, w))

        for i in range(num_children):
            child = self.copy()

            if self.hidden >= 1:
                upper = 3
            else:
                upper = 2
            if self.edge_count() >= 1:
                lower = 1
            else:
                lower = 2

            r = random.randint(lower, upper)

            if r == 1:
                # split edge into node and two edges
                edge = random.choice(edges)
                child.add_node(edge[0], edge[1], random.choice(activations))
            elif r == 2:
                # add random edge
                edge = random.choice(non_edges)
                while nx.has_path(self.g, edge[1], edge[0]):
                    edge = random.choice(non_edges)
                child.add_edge(edge[0], edge[1])
            else:
                # change activation
                v = random.randint(0, self.hidden - 1)
                a = self.activations[v]
                activations.remove(a)
                child.change_activation(v, random.choice(activations))
                activations.append(a)
            # make child part of species
            child.species = self.species
            children.append(child)
        return children

    def forward(self, x, weight):
        """
        Calculates forward pass using shared weight.
        No gradients are calculated.
        x should be dimensions: batch size * input_dim.
        Softmax at end for probabilities of each class.
        """
        assert len(x.shape) == 2 and x.shape[1] == self.input_dim
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            output = {i: torch.zeros(x.shape[0], device=device)
                      for i in range(self.hidden)}
            for i in range(self.input_dim):
                output["i" + str(i)] = x[:, i]
            for i in range(self.output_dim):
                output["o" + str(i)] = torch.zeros(x.shape[0], device=device)

            for v in topological_sort(self.g):
                activation = self.activations[v]
                if activation is not None:
                    output[v] = activation(output[v])
                for w in self.g.neighbors(v):
                    output[w] += weight * output[v]

            final = torch.stack(
                tuple(output["o" + str(i)] for i in range(self.output_dim))).transpose(0, 1)
            return F.softmax(final, dim=1)

    def visualize(self, arg_kwargs=None):
        """
      Creates diagram of WANN.
      """
        arg_kwargs = arg_kwargs or {}

        # position map
        pos = {}
        layered_pos = nx.nx_pydot.graphviz_layout(self.g, prog='dot')
        min_x = float("inf")
        max_x = -float("inf")
        min_y = float("inf")
        max_y = -float("inf")
        for v, (x, y) in layered_pos.items():
            min_x = min(min_x, -y)
            max_x = max(max_x, -y)
            if not isinstance(v, str) or v[0] != 'o':
                min_y = min(min_y, x)
                max_y = max(max_y, x)
        if self.hidden == 0:
            max_x = min_x + max_y - min_y

        for v, (x, y) in layered_pos.items():
            if isinstance(v, str):
                if v[0] == 'i':
                    pos[v] = (min_x, (max_y - min_y) * int(v[1:]) /
                              (self.input_dim - 1) + min_y)
                else:
                    pos[v] = (max_x, (max_y - min_y) * int(v[1:]) /
                              (self.output_dim - 1) + min_y)
            else:
                pos[v] = (-y, x)

        # labels
        labels = {}
        for i in range(self.hidden):
            labels[i] = wann.act_to_str(self.activations[i])
        if self.input_dim <= 20:
            for i in range(self.input_dim):
                labels["i" + str(i)] = "i" + str(i)
        if self.output_dim <= 20:
            for i in range(self.output_dim):
                labels["o" + str(i)] = "o" + str(i)

        # colors
        color_map = []
        for v in self.g:
            if isinstance(v, int):
                color_map.append(0.5)
            elif v[0] == "i":
                color_map.append(0.2)
            else:
                color_map.append(0.7)

        nx.draw(self.g, with_labels=True, pos=pos, labels=labels, node_size=400, node_color=color_map,
                cmap=plt.cm.Blues,
                vmin=0, vmax=1, **arg_kwargs)

    @staticmethod
    def act_to_str(activation):
        if activation is None:
            return "None"
        elif len(activation.__name__) <= 4:
            return activation.__name__
        return activation.__name__[:3]

    @staticmethod
    def str_to_act(s):
        if s == "None":
            return None
        elif s == "relu":
            return F.relu
        elif s == "sig":
            return torch.sigmoid
        else:
            return torch.tanh

    @staticmethod
    def save_json(wann, filename):
        """
      Saves WANN as a json file.
      """
        with open(filename + ".json", "w") as wann_file:
            json.dump({
                "input_dim": wann.input_dim,
                "output_dim": wann.output_dim,
                "hidden": wann.hidden,
                "graph": json_graph.node_link_data(wann.g),
                "activations": {v: wann.act_to_str(a) for v, a in wann.activations.items()}
            }, wann_file)

    @staticmethod
    def load_json(filename):
        """
      Returns a WANN from a json file.
      """
        with open(filename + ".json", "r") as wann_file:
            data = json.load(wann_file)
            w = wann(data["input_dim"], data["output_dim"])
            w.hidden = data["hidden"]
            w.g = json_graph.node_link_graph(data["graph"])
            w.activations = {v: wann.str_to_act(
                s) for v, s in data["activations"].items()}
            return w


def get_tensor_mask(g, nodes, init_weight=1):

    mask = nx.to_numpy_matrix(g, nodes)[:-1, -1]
    mask = init_weight*(mask*mask.T)
    return torch.from_numpy(mask).float()


class wannModel(nn.Module):
    def __init__(self, wann,):
        super(wannModel, self).__init__()
        self.wann = wann.copy()

        self.top_sort = topological_sort(self.wann.g)

        self.start_node = nn.Linear(wann.input_dim, wann.input_dim)
        self.weights = nn.ModuleList()

        for v in self.top_sort:
            nodes = [f"i{i}" for i in range(self.wann.input_dim)]
            if v not in nodes:
                nodes.append(v)
                self.weights.append(CustomizedLinear(
                    get_tensor_mask(self.wann.g, nodes)))

        self.output_layer = nn.Linear(
            self.wann.input_dim, self.wann.output_dim)

    def forward(self, x):
        x = self.start_node(x)
        x_ = x
        for layer in self.weights:
            x_ = layer(x_)
            x_ = F.relu(x_)
            x_ = x_ + x
        x_ = self.output_layer(x_)
        return F.softmax(x_, dim=1)
