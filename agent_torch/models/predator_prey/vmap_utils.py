"""
vmap_utils.py - Utility functions for vectorized operations in predator-prey model
"""
import torch
from typing import Callable, Any, List, Tuple, Dict
from functools import wraps

def vmap_batch(func: Callable, in_dims=0, out_dims=0) -> Callable:
    """
    Creates a vectorized version of a function using torch.vmap.
    
    Args:
        func: Function to vectorize
        in_dims: Input dimensions specification for vmap
        out_dims: Output dimensions specification for vmap
    
    Returns:
        Vectorized function
    """
    return torch.vmap(func, in_dims=in_dims, out_dims=out_dims)

def batch_process(func: Callable, inputs: List[Any], batch_size: int = 32) -> List[Any]:
    """
    Process inputs in batches using a vectorized function.
    
    Args:
        func: Function to apply to each batch
        inputs: List of inputs to process
        batch_size: Size of each batch
    
    Returns:
        List of results for each input
    """
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        # Pad batch if necessary
        if len(batch) < batch_size:
            pad_size = batch_size - len(batch)
            # Padding depends on the type of batch elements
            if isinstance(batch[0], torch.Tensor):
                padding = [torch.zeros_like(batch[0]) for _ in range(pad_size)]
            else:
                padding = [None] * pad_size
            padded_batch = batch + padding
            batch_result = func(padded_batch)
            results.extend(batch_result[:len(batch)])
        else:
            batch_result = func(batch)
            results.extend(batch_result)
    return results

def vectorized_substep(in_dims=0, out_dims=0):
    """
    Decorator for vectorizing substep forward methods.
    
    Args:
        in_dims: Input dimensions specification for vmap
        out_dims: Output dimensions specification for vmap
    
    Returns:
        Decorated function
    """
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Create vectorized version of the method on first call
            if not hasattr(self, f"_vectorized_{method.__name__}"):
                vectorized_method = vmap_batch(
                    lambda *inner_args: method(self, *inner_args),
                    in_dims=in_dims,
                    out_dims=out_dims
                )
                setattr(self, f"_vectorized_{method.__name__}", vectorized_method)
            
            # Call the vectorized method
            return getattr(self, f"_vectorized_{method.__name__}")(*args, **kwargs)
        return wrapper
    return decorator

# Position utilities
def position_to_node(position, max_y):
    """Convert (x,y) position to node index"""
    x, y = position
    return (max_y * x) + y

def node_to_position(node, max_y):
    """Convert node index to (x,y) position"""
    y = int(node % max_y)
    x = int((node - y) / max_y)
    return torch.tensor([x, y])

# Agent utilities
def agent_alive(energy):
    """Check if an agent is alive based on energy"""
    return energy > 0

# Batched agent checks
check_agents_alive = vmap_batch(agent_alive, in_dims=0, out_dims=0)

# Comparison utilities
def positions_equal(pos1, pos2):
    """Check if two positions are equal"""
    return torch.all(pos1 == pos2).item()

# Batched comparisons
def check_all_positions_equal(positions, reference_pos):
    """Check if any positions match the reference position"""
    check_single_pos = lambda pos: positions_equal(pos, reference_pos)
    return vmap_batch(check_single_pos, in_dims=0, out_dims=0)(positions)

# Energy update utilities
def update_energy(energy, amount):
    """Update energy by a given amount"""
    return energy + amount

# Batched energy updates
def update_all_energies(energies, amounts):
    """Update multiple energy values at once"""
    return vmap_batch(update_energy, in_dims=(0, 0), out_dims=0)(energies, amounts)

# Grass utilities
def process_grass_growth(growth, countdown, regrowth_time):
    """Process growth for a single grass cell"""
    new_countdown = countdown - 1
    new_growth = 1.0 if new_countdown <= 0 else 0.0
    return new_growth, new_countdown

# Batched grass growth
def process_all_grass(growth, countdown, regrowth_time):
    """Process growth for all grass cells"""
    return vmap_batch(
        lambda g, c: process_grass_growth(g, c, regrowth_time),
        in_dims=(0, 0),
        out_dims=(0, 0)
    )(growth, countdown)