# agent_torch/models/predator_prey/vmap_substeps/move.py
import torch
import random

def position_to_node(pos, max_y):
    """Convert (x,y) position to node index"""
    x, y = pos
    return (max_y * x) + y

def get_neighbors(positions, adjacency, bounds):
    """
    Find valid neighbors for each position.
    
    Args:
        positions: Positions of all agents
        adjacency: Adjacency matrix for the grid
        bounds: Grid boundaries (max_x, max_y)
        
    Returns:
        List of neighbors for each position
    """
    max_x, max_y = bounds
    all_neighbors = []
    
    for pos in positions:
        x, y = pos
        node = position_to_node((x, y), max_y)
        
        # Skip invalid nodes
        if node >= adjacency.shape[0]:
            all_neighbors.append([])
            continue
            
        # Find connected nodes
        connected = adjacency[node].nonzero().squeeze(1)
        neighbors = []
        for idx in connected:
            ny = int(idx % max_y)
            nx = int((idx - ny) / max_y)
            neighbors.append(torch.tensor([nx, ny]))
            
        all_neighbors.append(neighbors)
    
    return all_neighbors

def move_agents(positions, energy, work, neighbors):
    """
    Move agents to random neighbors.
    
    Args:
        positions: Current positions of all agents
        energy: Energy levels of all agents
        work: Energy cost of movement
        neighbors: List of neighbors for each position
        
    Returns:
        Tuple of (new_positions, new_energy)
    """
    # Check which agents have energy to move
    has_energy = energy.squeeze(-1) > 0
    
    # Update energy using tensor operations (not conditionals)
    energy_flat = energy.squeeze(-1)
    new_energy = energy_flat - work
    
    # Handle neighbor selection
    next_positions = []
    for i, (pos, can_move, nbrs) in enumerate(zip(positions, has_energy, neighbors)):
        if can_move.item() and nbrs:  # Use item() to convert tensor to Python scalar
            next_positions.append(random.choice(nbrs))
        else:
            next_positions.append(pos)
    
    return torch.stack(next_positions), new_energy.view(-1, 1)