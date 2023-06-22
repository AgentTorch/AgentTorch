import torch
import torch.nn as nn
import networkx as nx

from utils.general import *

dtype_dict = {
        'int': torch.int32,
        'float': torch.float32
        }

def zeros(shape, params):
    processed_shape = [process_shape_omega(s) for s in shape]
    value = torch.zeros(size=processed_shape, dtype=dtype_dict[params['dtype']])
    return value

def constant(shape, params):
    processed_shape = [process_shape_omega(s) for s in shape]
    value = params['value']*torch.ones(size=processed_shape)
    return value

def random_normal(shape, params):
    processed_shape = [process_shape_omega(s) for s in shape]
    value = params['mu'] + params['sigma']*torch.randn(size=processed_shape)
    return value

def random_normal_col_by_col(shape, params):
    processed_shape = [process_shape_omega(s) for s in shape]
    value_dims = []
        
    for dim in range(processed_shape[1]): #Hard coded for 2d tensors
        value_dims.append(torch.clip(params[f'mu_{dim}'] + 
                         params[f'sigma_{dim}']* 
                         torch.randn(size=(processed_shape[0],)), min=0.0))
    value = torch.stack(value_dims, dim=1)
    return value


def grid_network(params):
    G = nx.grid_graph(dim=tuple(params['shape']))
    A = torch.tensor(nx.adjacency_matrix(G).todense())
    
    return G, A

def watt_strogatz_network(params):
    pass

def custom_network(params):
    pass

    
