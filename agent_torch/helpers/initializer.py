import torch
import torch.nn as nn
from .general import *

dtype_dict = {"int": torch.int32, "float": torch.float32}


def zeros(shape, params):
    processed_shape = [s for s in shape]
    value = torch.zeros(size=processed_shape, dtype=dtype_dict[params["dtype"]])
    return value


def constant(shape, params):
    processed_shape = [s for s in shape]
    value = params["value"] * torch.ones(size=processed_shape)
    return value


def random_normal(shape, params):
    processed_shape = [s for s in shape]
    value = params["mu"] + params["sigma"] * torch.randn(size=processed_shape)
    return value
