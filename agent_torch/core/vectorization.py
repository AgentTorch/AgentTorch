"""
Vectorization utilities for AgentTorch.

This module provides decorators and utilities for implementing and using 
vectorized operations in AgentTorch simulations.
"""

import torch
from torch.func import vmap
from functools import wraps
import inspect


def vectorized(fn=None, *, in_dims=0, out_dims=0):
    """
    Decorator that marks a function or method as vectorized.

    This decorator adds a _vectorized flag to the function, which is used by
    the VectorizedRunner to determine if a substep is implemented in a vectorized way.

    Args:
        fn: The function to decorate
        in_dims: Input dimensions specification for vmap (used when automatically applying vmap)
        out_dims: Output dimensions specification for vmap (used when automatically applying vmap)

    Returns:
        The decorated function with a _vectorized flag
    """

    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)

        # Mark the function as vectorized
        wrapped_func._vectorized = True
        wrapped_func._vmap_in_dims = in_dims
        wrapped_func._vmap_out_dims = out_dims
        wrapped_func._original_func = func

        return wrapped_func

    if fn is None:
        return decorator
    return decorator(fn)


def is_vectorized(fn):
    """
    Check if a function or method is marked as vectorized.

    Args:
        fn: The function to check

    Returns:
        bool: True if the function is vectorized, False otherwise
    """
    return hasattr(fn, "_vectorized") and fn._vectorized


def get_batch_size(tensor_list):
    """
    Get the batch size from a list of tensors.

    Args:
        tensor_list: List of tensors

    Returns:
        int: The batch size (size of first dimension) of the first non-empty tensor
    """
    for tensor in tensor_list:
        if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
            return tensor.size(0)
    return 0


def apply_vmap(fn, *args, in_dims=0, out_dims=0, **kwargs):
    """
    Apply vmap to a function with the given arguments.

    Args:
        fn: The function to vectorize
        args: Arguments to pass to the function
        in_dims: Input dimensions specification for vmap
        out_dims: Output dimensions specification for vmap
        kwargs: Keyword arguments to pass to the function

    Returns:
        The result of applying vmap to the function
    """

    # Create a new function that handles kwargs
    def wrapped(*a):
        return fn(*a, **kwargs)

    # Apply vmap to the wrapped function
    vmapped_fn = vmap(wrapped, in_dims=in_dims, out_dims=out_dims)

    # Call the vmapped function with args
    return vmapped_fn(*args)


def reshape_for_vmap(tensor, batch_size):
    """
    Reshape a tensor for use with vmap if needed.

    Args:
        tensor: The tensor to reshape
        batch_size: Target batch size

    Returns:
        torch.Tensor: Reshaped tensor
    """
    if tensor.dim() == 0:
        # Scalar tensor, expand to [batch_size, 1]
        return tensor.expand(batch_size, 1)
    elif tensor.size(0) == 1 and batch_size > 1:
        # Single-item tensor, expand to [batch_size, ...]
        return tensor.expand(batch_size, *tensor.shape[1:])
    elif tensor.dim() == 1:
        # 1D tensor, add a dimension: [batch_size] -> [batch_size, 1]
        return tensor.unsqueeze(1)
    else:
        # Already properly shaped tensor
        return tensor


def get_vmapped_output(fn, args_dict, batch_size):
    """
    Apply a vectorized function to a dictionary of arguments.

    Args:
        fn: The function to apply
        args_dict: Dictionary of arguments to pass to the function
        batch_size: Batch size for reshaping arguments

    Returns:
        The result of applying the function to the arguments
    """
    # Reshape arguments for vmap
    reshaped_args = {k: reshape_for_vmap(v, batch_size) for k, v in args_dict.items()}

    # Get in_dims and out_dims from the function if available
    in_dims = getattr(fn, "_vmap_in_dims", 0)
    out_dims = getattr(fn, "_vmap_out_dims", 0)

    # Apply vmap
    return apply_vmap(fn, *reshaped_args.values(), in_dims=in_dims, out_dims=out_dims)


# Register vectorized versions of common PyTorch functions
@vectorized
def zeros_like(input):
    return torch.zeros_like(input)


@vectorized
def ones_like(input):
    return torch.ones_like(input)


@vectorized
def full_like(input, fill_value):
    return torch.full_like(input, fill_value)


@vectorized
def cat(tensors, dim=0):
    return torch.cat(tensors, dim=dim)


@vectorized
def stack(tensors, dim=0):
    return torch.stack(tensors, dim=dim)
