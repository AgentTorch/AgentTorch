import pandas as pd
import torch
from scipy.stats import gamma

def read_from_file(inputs):
    try:
        filename = inputs['filename']
    except:
        raise ValueError('Missing variable "filename" in inputs.')
    if filename[-4:] == ".csv":
        data = pd.read_csv(filename)
    else:
        raise ValueError("Unknown file type for reading input.")
    

def initialize_constant(arguments):
    try:
        value = arguments['constant']
        shape = tuple(arguments['shape'])
        if type(value) == list:
            res = torch.tensor(value)
            assert shape == res.shape, "Shape specified and values given don't match."
        else:
            res = value*torch.ones(shape)
    except:
        raise ValueError('Missing variables in arguments - needed "constant" and "shape".')
    return res

def get_lam_gamma_integrals(inputs):
        scale, rate, t = inputs['scale'], inputs['rate'], inputs['t']
        b = rate * rate / scale
        a = scale / b  # / b
        res = [(gamma.cdf(t_i, a=a, loc=0, scale=b) - gamma.cdf(t_i-1, a=a, loc=0, scale=b)) for t_i in range(t)]
        return torch.tensor(res).float()