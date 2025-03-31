# agent_torch/models/predator_prey/vmap_substeps/eat.py
import torch

def position_to_node(pos, max_y):
    """Convert (x,y) position to node index"""
    x, y = pos
    return (max_y * x) + y

def eat_grass(prey_pos, prey_energy, grass_growth, growth_countdown, bounds, nutrition, regrowth_time):
    """
    Prey consume grass using vectorized operations.
    
    Args:
        prey_pos: Positions of all prey
        prey_energy: Energy levels of all prey
        grass_growth: Growth state of all grass cells
        growth_countdown: Countdown timers for all grass cells
        bounds: Grid boundaries (max_x, max_y)
        nutrition: Energy gained from eating grass
        regrowth_time: Time needed for grass to regrow
        
    Returns:
        Tuple of (new_prey_energy, new_grass_growth, new_growth_countdown)
    """
    max_x, max_y = bounds
    did_eat = torch.zeros(len(prey_pos), dtype=torch.bool)
    eaten_grass_nodes = []
    
    # Find grown grass
    grown_grass = (grass_growth == 1).squeeze(-1).nonzero().squeeze(-1)
    
    # Check which prey are at grown grass positions
    for node in grown_grass:
        # Convert to position
        y = int(node % max_y)
        x = int((node - y) / max_y)
        grass_pos = torch.tensor([x, y])
        
        # Check which prey are at this position using vmap
        check_prey_at_pos = torch.vmap(
            lambda prey_p: torch.all(prey_p == grass_pos),
            in_dims=0,
            out_dims=0
        )
        prey_at_pos = check_prey_at_pos(prey_pos)
        
        if torch.any(prey_at_pos):
            did_eat = did_eat | prey_at_pos
            eaten_grass_nodes.append(node)
    
    # Update prey energy using vmap and tensor operations (not conditionals)
    energy_flat = prey_energy.squeeze(-1)
    nutrition_value = torch.ones_like(energy_flat) * nutrition
    
    # Apply nutrition to prey that ate grass
    energy_update = did_eat.float() * nutrition_value
    new_energy = energy_flat + energy_update
    
    # Update grass state
    new_grass_growth = grass_growth.clone()
    new_growth_countdown = growth_countdown.clone()
    
    for node in eaten_grass_nodes:
        new_grass_growth[node] = 0
        new_growth_countdown[node] = regrowth_time
    
    return new_energy.view(-1, 1), new_grass_growth, new_growth_countdown