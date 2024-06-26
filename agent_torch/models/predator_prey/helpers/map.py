# helpers/map.py

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import osmnx as ox


def map_network(params):
    coordinates = (40.78264403323726, -73.96559413265355)  # central park
    distance = 550

    graph = ox.graph_from_point(
        coordinates, dist=distance, simplify=True, network_type="walk"
    )
    adjacency_matrix = nx.adjacency_matrix(graph).todense()

    return graph, torch.tensor(adjacency_matrix)
