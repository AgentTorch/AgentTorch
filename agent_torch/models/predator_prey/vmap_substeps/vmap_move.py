"""
Vectorized implementation of the move substep for the predator-prey model.
"""

import math
import torch
import re
import random
from torch.func import vmap

from agent_torch.core.registry import Registry
from agent_torch.core.substep import (
    SubstepObservation,
    SubstepAction,
    SubstepTransition,
)
from agent_torch.core.helpers.general import get_by_path
from agent_torch.core.vectorization import vectorized


def get_var(state, var):
    """
    Retrieves a value from the current state of the model.
    """
    return get_by_path(state, re.split("/", var))


def get_neighbor_indices(node, adj_grid):
    """
    Get indices of neighboring nodes in the adjacency matrix.
    """
    connected = adj_grid[node].nonzero(as_tuple=True)[0]
    return connected


def idx_to_coords(idx, max_y):
    """
    Convert a node index to (x, y) coordinates.
    """
    y = idx % max_y
    x = (idx - y) // max_y
    return torch.tensor([x, y], dtype=torch.long)


def process_single_position(pos, adj_grid, bounds):
    """
    Process a single position to find its neighbors.
    This function will be vectorized with vmap.
    """
    x, y = pos
    max_x, max_y = bounds
    node = (max_y * x) + y

    # Find connected nodes (neighbors)
    neighbors_mask = adj_grid[node]
    connected_indices = neighbors_mask.nonzero().squeeze(-1)

    # Convert indices to coordinates
    if connected_indices.numel() == 0:
        # Return empty tensor with correct shape for consistent output
        return torch.zeros((0, 2), dtype=torch.long)

    # Handle single connection case
    if connected_indices.dim() == 0:
        connected_indices = connected_indices.unsqueeze(0)

    y_coords = connected_indices % max_y
    x_coords = torch.div(connected_indices - y_coords, max_y, rounding_mode="floor")

    return torch.stack([x_coords, y_coords], dim=1)


@Registry.register_substep("find_neighbors", "observation")
class FindNeighborsVmap(SubstepObservation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vectorized = True

    def forward(self, state):
        input_variables = self.input_variables
        bounds = get_var(state, input_variables["bounds"])
        adj_grid = get_var(state, input_variables["adj_grid"])
        positions = get_var(state, input_variables["positions"])

        # This is where we would use vmap, but since process_single_position
        # returns variable-length outputs, we need a list comprehension approach
        # If we could guarantee fixed output size, we could use:
        # vmapped_process = vmap(lambda pos: process_single_position(pos, adj_grid, bounds))
        # possible_neighbors = vmapped_process(positions)

        # For now, handle each position individually but leverage vectorized operations
        # within the processing of each position
        possible_neighbors = []
        for i in range(len(positions)):
            neighbors = process_single_position(positions[i], adj_grid, bounds)
            possible_neighbors.append(neighbors)

        return {self.output_variables[0]: possible_neighbors}


def decide_next_position(pos, energy, neighbors, rng_seed=None):
    """
    Decide the next position for a single agent.
    This function will be used with vmap.

    Args:
        pos: Current position
        energy: Current energy
        neighbors: List of neighboring positions
        rng_seed: Optional seed for random choice

    Returns:
        Next position for the agent
    """
    # If no energy or no neighbors, stay in place
    if energy.item() <= 0 or len(neighbors) == 0:
        return pos

    # Set a deterministic seed for this agent if provided
    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    # Choose a random neighbor
    idx = torch.randint(0, len(neighbors), (1,)).item()
    return neighbors[idx]


@Registry.register_substep("decide_movement", "policy")
class DecideMovementVmap(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vectorized = True

    def forward(self, state, observations):
        input_variables = self.input_variables
        positions = get_var(state, input_variables["positions"])
        energy = get_var(state, input_variables["energy"])
        possible_neighbors = observations["possible_neighbors"]

        # This is a case where we can't directly use vmap because:
        # 1. possible_neighbors is a list of tensors with different shapes
        # 2. We need random selection that's not easily vectorized

        # However, we can still use vectorized operations within our loop
        # and parallelize with vmap if we had uniform neighbor counts

        # Generate random seeds for each agent for reproducibility
        random_seeds = torch.randint(0, 10000, (len(positions),))

        # Choose next positions
        next_positions = []
        for i, (pos, e, neighbors, seed) in enumerate(
            zip(positions, energy, possible_neighbors, random_seeds)
        ):
            # Use our vectorized function for a single agent
            next_pos = decide_next_position(pos, e, neighbors, seed)
            next_positions.append(next_pos)

        # Stack results into a single tensor
        next_positions = torch.stack(
            [pos.squeeze() if pos.dim() > 1 else pos for pos in next_positions]
        )

        return {self.output_variables[0]: next_positions}


def update_agent_energy(energy, work):
    """
    Update energy after movement for a single agent.
    This function will be vectorized with vmap.

    Args:
        energy: Current energy value
        work: Work required for movement

    Returns:
        Updated energy value
    """
    return energy - work


@Registry.register_substep("update_positions", "transition")
class UpdatePositionsVmap(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vectorized = True

    def forward(self, state, action):
        input_variables = self.input_variables

        # Handle the case where action might be None
        if (
            action is None
            or "prey" not in action
            or action["prey"] is None
            or "predator" not in action
            or action["predator"] is None
        ):
            print("Warning: Incomplete action data in UpdatePositionsVmap")
            prey_pos = get_var(state, "agents/prey/coordinates")
            pred_pos = get_var(state, "agents/predator/coordinates")
        else:
            prey_pos = action["prey"]["next_positions"]
            pred_pos = action["predator"]["next_positions"]

        prey_energy = get_var(state, input_variables["prey_energy"])
        pred_energy = get_var(state, input_variables["pred_energy"])
        prey_work = get_var(state, input_variables["prey_work"])
        pred_work = get_var(state, input_variables["pred_work"])

        # The error is caused by prey_energy and prey_work having different batch dimensions
        # Ensure prey_work has the same batch size as prey_energy
        if prey_energy.size(0) != prey_work.size(0):
            # Expand prey_work to match the batch size of prey_energy
            prey_work_expanded = prey_work.expand(prey_energy.size(0), -1)
        else:
            prey_work_expanded = prey_work

        # Similar fix for predator energy and work
        if pred_energy.size(0) != pred_work.size(0):
            pred_work_expanded = pred_work.expand(pred_energy.size(0), -1)
        else:
            pred_work_expanded = pred_work

        # Apply the energy update separately for each agent to avoid vmap dimension issues
        prey_energy_new = prey_energy - prey_work_expanded
        pred_energy_new = pred_energy - pred_work_expanded

        return {
            self.output_variables[0]: prey_pos,
            self.output_variables[1]: prey_energy_new,
            self.output_variables[2]: pred_pos,
            self.output_variables[3]: pred_energy_new,
        }
