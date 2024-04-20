import torch
import torch.nn as nn

import pandas as pd


def random_normal_col_by_col(shape, params):
    processed_shape = shape
    value_dims = []
        
    for dim in range(processed_shape[1]): #Hard coded for 2d tensors
        value_dims.append(torch.clip(params[f'mu_{dim}'] + 
                         params[f'sigma_{dim}']* 
                         torch.randn(size=(processed_shape[0],)), min=0.0))
    value = torch.stack(value_dims, dim=1)
    return value

def load_population_attribute(shape, params):
    """
    Load population data from a pandas dataframe
    """
    # Load population data
    df = pd.read_pickle(params['file_path'])
    att_tensor = torch.from_numpy(df.values).float()
    return att_tensor

def get_population_size(shape, params):
    """
    Get the population size from a pandas dataframe
    """
    # Load population data
    df = pd.read_pickle(params['file_path'])
    pop_size = df['synpop'].shape[0]
    return pop_size

def initialize_id(shape, params):
    """
    Initialize a unique ID for each agent
    """
    return torch.arange(0, shape[0]).reshape(-1, 1).float()