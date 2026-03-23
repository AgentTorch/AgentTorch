import torch
import torch.nn as nn
import networkx as nx


def grid_network(params):
    G = nx.grid_graph(dim=tuple(params["shape"]))
    A = torch.tensor(nx.adjacency_matrix(G).todense())
    return G, A


def small_world(params):
    """Watts-Strogatz small-world graph."""
    n = params.get("n", params.get("num_agents", 1000))
    k = params.get("k", 10)
    p = params.get("p", 0.1)
    G = nx.watts_strogatz_graph(n, k, p)
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    edge_attr = torch.ones(edge_index.shape[1], 1)
    return G, (edge_index, edge_attr)


def erdos_renyi(params):
    """Erdos-Renyi random graph."""
    n = params.get("n", params.get("num_agents", 1000))
    p = params.get("p", 0.05)
    G = nx.erdos_renyi_graph(n, p)
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    edge_attr = torch.ones(edge_index.shape[1], 1)
    return G, (edge_index, edge_attr)


def scale_free(params):
    """Barabasi-Albert scale-free graph."""
    n = params.get("n", params.get("num_agents", 1000))
    m = params.get("m", 3)
    G = nx.barabasi_albert_graph(n, m)
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    edge_attr = torch.ones(edge_index.shape[1], 1)
    return G, (edge_index, edge_attr)
