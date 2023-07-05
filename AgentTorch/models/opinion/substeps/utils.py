import torch
import torch.nn as nn

def random_normal_col_by_col(shape, params):
    processed_shape = shape
    value_dims = []
        
    for dim in range(processed_shape[1]): #Hard coded for 2d tensors
        value_dims.append(torch.clip(params[f'mu_{dim}'] + 
                         params[f'sigma_{dim}']* 
                         torch.randn(size=(processed_shape[0],)), min=0.0))
    value = torch.stack(value_dims, dim=1)
    return value