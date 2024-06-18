import re
from functools import reduce
import operator
import torch
from torch import nn
import copy
from omegaconf import OmegaConf
import pandas as pd


def get_by_path(root, items):
    r"""
    Access a nested object in root by item sequence
    """
    property_obj = reduce(operator.getitem, items, root)
    if isinstance(property_obj, nn.ModuleDict):
        return property_obj
    elif isinstance(property_obj, nn.Module):
        return property_obj()
    else:
        return property_obj


def set_by_path(root, items, value):
    r"""Set a value in a nested object in root by item sequence"""
    val_obj = get_by_path(root, items[:-1])

    if isinstance(val_obj, nn.ModuleDict):
        print("set_by_path on nn.ModuleDict breaks gradient currently!")
        val_obj[items[-1]].param.data.copy_(value)
        val_obj[items[-1]].param.requires_grad = value.requires_grad
    else:
        val_obj[items[-1]] = value
        return root


def del_by_path(root, items):
    """Delete a key-value in a nested object in root by item sequence."""
    del get_by_path(root, items[:-1])[items[-1]]


def copy_module(dict_to_copy):
    r"""
    Creates a new dictionary with a copy of each PyTorch tensor in the input dictionary.
    Handles nested dictionaries of PyTorch tensors of variable depth.
    """
    copied_dict = {}
    for key, value in dict_to_copy.items():
        if torch.is_tensor(value):
            copied_dict[key] = torch.clone(value)
        elif isinstance(value, dict):
            copied_dict[key] = copy_module(value)
        elif not torch.is_tensor(value):
            copied_dict[key] = copy.deepcopy(value)
        else:
            raise TypeError("Type error.. ", type(value))

    return copied_dict


def to_cpu(dict_to_copy):
    r"""
    Creates a new dictionary with a copy of each PyTorch tensor in the input dictionary.
    Handles nested dictionaries of PyTorch tensors of variable depth.
    """
    copied_dict = {}
    for key, value in dict_to_copy.items():
        # value = dict_to_copy[key]
        if torch.is_tensor(value):
            copied_dict[key] = torch.clone(value).cpu()
        elif isinstance(value, dict):
            copied_dict[key] = to_cpu(value)
        elif not torch.is_tensor(value):
            copied_dict[key] = value
        else:
            raise TypeError("Type error.. ", type(value))

    del dict_to_copy

    return copied_dict


def process_shape(config, s):
    if type(s) == str:
        return get_by_path(config, re.split("/", s))
    else:
        return s


def register_resolver(name, resolver):
    OmegaConf.register_new_resolver(name, resolver)


def read_config(config_file):
    register_resolver("sum", lambda x, y: x + y)
    register_resolver("multiply", lambda x, y: x * y)
    register_resolver("divide", lambda x, y: x // y)

    if config_file[-5:] != ".yaml":
        raise ValueError("Config file type should be yaml")
    try:
        config = OmegaConf.load(config_file)
        config = OmegaConf.to_object(config)
    except Exception as e:
        raise ValueError(
            f"Could not load config file. Please check path ad file type. Error message is {str(e)}"
        )

    return config


def read_from_file(shape, params):
    file_path = params["file_path"]

    if file_path[-3:] == "csv":
        data = pd.read_csv(file_path)

    data_values = data.values
    assert data_values.shape == tuple(shape)

    data_tensor = torch.from_numpy(data_values)

    return data_tensor


def memory_checkpoint(name):
    print("Checkpoint: ", name)
    checkpoint_allocated = torch.cuda.memory_allocated()
    checkpoint_reserved = torch.cuda.memory_reserved()

    print("Allocated: ", checkpoint_allocated, " Reserved: ", checkpoint_reserved)

    return checkpoint_allocated, checkpoint_reserved
