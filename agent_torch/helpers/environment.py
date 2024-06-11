import torch
import torch.nn as nn
import networkx as nx


def grid_network(params):
    G = nx.grid_graph(dim=tuple(params["shape"]))
    A = torch.tensor(nx.adjacency_matrix(G).todense())

    return G, A
