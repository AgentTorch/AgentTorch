# agent_torch/models/predator_prey/vmap_substeps/grow.py
import torch

def grow_grass_batch(growth, countdown, regrowth_time):
    """
    Process growth for all grass cells using tensor operations rather than conditionals.
    
    Args:
        growth: Tensor containing growth states (0 or 1)
        countdown: Tensor containing growth countdown values
        regrowth_time: Time needed for grass to regrow
        
    Returns:
        Tuple of (new_growth, new_countdown)
    """
    # Decrease countdown
    new_countdown = countdown - 1
    
    # Set growth based on countdown - using tensor operations instead of if/else
    # This avoids the data-dependent control flow issue
    new_growth = (new_countdown <= 0).float()
    
    return new_growth, new_countdown