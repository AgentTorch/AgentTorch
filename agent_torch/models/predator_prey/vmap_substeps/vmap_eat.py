"""
Vectorized implementation of the eat substep for the predator-prey model.
"""

import torch
import re
from torch.func import vmap

from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction, SubstepTransition
from agent_torch.core.helpers.general import get_by_path
from agent_torch.core.vectorization import vectorized


def get_var(state, var):
    """
    Retrieves a value from the current state of the model.
    """
    return get_by_path(state, re.split("/", var))


def is_grass_eatable(pos, grass_growth, max_y):
    """
    Check if grass at a given position is eatable (fully grown).
    This function will be vectorized with vmap.

    Args:
        pos: Position coordinates (x, y)
        grass_growth: Growth stage tensor for all grass
        max_y: Y dimension of the grid

    Returns:
        tuple: (is_eatable, position) - where is_eatable is 1.0 if grass is eatable, 0.0 otherwise
    """
    x, y = pos
    node = (max_y * x) + y

    # Check if node is valid and grass is fully grown
    if node < len(grass_growth) and grass_growth[node] == 1:
        return 1.0, pos

    return 0.0, pos  # Not eatable


@Registry.register_substep("find_eatable_grass", "policy")
class FindEatableGrassVmap(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vectorized = True

    def forward(self, state, observations):
        input_variables = self.input_variables
        bounds = get_var(state, input_variables["bounds"])
        positions = get_var(state, input_variables["positions"])
        grass_growth = get_var(state, input_variables["grass_growth"])

        max_x, max_y = bounds

        # Apply vmap to check all positions at once
        vmapped_check = vmap(lambda pos: is_grass_eatable(pos, grass_growth, max_y))
        is_eatable, all_positions = vmapped_check(positions)

        # Filter positions where grass is eatable
        eatable_mask = is_eatable > 0.5
        eatable_indices = eatable_mask.nonzero().squeeze(-1)

        # Handle case where no grass is eatable
        if eatable_indices.numel() == 0:
            return {self.output_variables[0]: []}

        # Handle single eatable grass case
        if eatable_indices.dim() == 0:
            eatable_indices = eatable_indices.unsqueeze(0)

        # Extract eatable positions
        eatable_positions = [positions[idx] for idx in eatable_indices]

        return {self.output_variables[0]: eatable_positions}


@Registry.register_substep("eat_grass", "transition")
class EatGrassVmap(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vectorized = True

    def forward(self, state, action):
        input_variables = self.input_variables
        bounds = get_var(state, input_variables["bounds"])
        prey_pos = get_var(state, input_variables["prey_pos"])
        energy = get_var(state, input_variables["energy"])
        nutrition = get_var(state, input_variables["nutrition"])
        grass_growth = get_var(state, input_variables["grass_growth"])
        growth_countdown = get_var(state, input_variables["growth_countdown"])
        regrowth_time = get_var(state, input_variables["regrowth_time"])

        # Handle case where there's no eatable grass
        if (
            "prey" not in action
            or action["prey"] is None
            or "eatable_grass_positions" not in action["prey"]
            or len(action["prey"]["eatable_grass_positions"]) < 1
        ):
            return {}

        eatable_grass_positions = action["prey"]["eatable_grass_positions"]
        max_x, max_y = bounds

        # Create masks for updating values
        energy_mask = torch.zeros_like(energy, dtype=torch.bool)
        grass_mask = torch.zeros_like(grass_growth)
        countdown_mask = torch.zeros_like(growth_countdown)

        # Process each eatable grass position
        for pos in eatable_grass_positions:
            x, y = pos
            node = (max_y * x) + y

            # Check which prey are at this position
            prey_at_pos = torch.all(prey_pos == pos, dim=1)
            energy_mask = energy_mask | prey_at_pos.unsqueeze(1)

            # Update grass state
            if node < len(grass_growth):
                grass_mask[node] = -1
                countdown_mask[node] = regrowth_time - growth_countdown[node]

        # Define vectorized update functions
        def update_energy(e, mask, nutrition_val):
            return torch.where(mask, e + nutrition_val, e)

        def update_grass(growth, mask):
            return growth + mask

        def update_countdown(countdown, mask):
            return countdown + mask

        # Apply vectorized updates using vmap
        vmapped_energy_update = vmap(update_energy, in_dims=(0, 0, None))
        energy = vmapped_energy_update(energy, energy_mask, nutrition)

        vmapped_grass_update = vmap(update_grass)
        grass_growth = vmapped_grass_update(grass_growth, grass_mask)

        vmapped_countdown_update = vmap(update_countdown)
        growth_countdown = vmapped_countdown_update(growth_countdown, countdown_mask)

        return {
            self.output_variables[0]: energy,
            self.output_variables[1]: grass_growth,
            self.output_variables[2]: growth_countdown,
        }
