"""
Vectorized implementation of the hunt substep for the predator-prey model.
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


def has_prey_at_position(pred_pos, prey_positions):
    """
    Check if there is any prey at the predator's position.
    This function will be vectorized with vmap.

    Args:
        pred_pos: Predator position
        prey_positions: All prey positions

    Returns:
        tuple: (has_prey, position) - where has_prey is 1.0 if prey found, 0.0 otherwise
    """
    # Check if any prey is at this predator's position
    # Broadcasting to compare one predator position with all prey positions
    matches = (prey_positions == pred_pos.unsqueeze(0)).all(dim=1)

    # If any match found, return 1.0, otherwise 0.0
    if matches.any():
        return 1.0, pred_pos
    else:
        return 0.0, pred_pos


@Registry.register_substep("find_targets", "policy")
class FindTargetsVmap(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vectorized = True

    def forward(self, state, observations):
        input_variables = self.input_variables
        prey_pos = get_var(state, input_variables["prey_pos"])
        pred_pos = get_var(state, input_variables["pred_pos"])

        # Apply vmap to check all predator positions at once
        vmapped_check = vmap(lambda pos: has_prey_at_position(pos, prey_pos))
        has_prey, all_positions = vmapped_check(pred_pos)

        # Filter positions where predators find prey
        target_mask = has_prey > 0.5
        target_indices = target_mask.nonzero().squeeze(-1)

        # Handle case where no targets found
        if target_indices.numel() == 0:
            return {self.output_variables[0]: []}

        # Handle single target case
        if target_indices.dim() == 0:
            target_indices = target_indices.unsqueeze(0)

        # Extract target positions
        target_positions = [pred_pos[idx] for idx in target_indices]

        return {self.output_variables[0]: target_positions}


@Registry.register_substep("hunt_prey", "transition")
class HuntPreyVmap(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vectorized = True

    def forward(self, state, action):
        input_variables = self.input_variables
        prey_pos = get_var(state, input_variables["prey_pos"])
        prey_energy = get_var(state, input_variables["prey_energy"])
        pred_pos = get_var(state, input_variables["pred_pos"])
        pred_energy = get_var(state, input_variables["pred_energy"])
        nutrition = get_var(state, input_variables["nutritional_value"])

        # Handle case where there are no targets
        if (
            "predator" not in action
            or action["predator"] is None
            or "target_positions" not in action["predator"]
            or len(action["predator"]["target_positions"]) < 1
        ):
            return {}

        target_positions = action["predator"]["target_positions"]

        # Create energy update masks
        prey_mask = torch.zeros_like(prey_energy, dtype=torch.bool)
        pred_mask = torch.zeros_like(pred_energy, dtype=torch.bool)

        # Process each target position
        for pos in target_positions:
            # Find prey at this position
            prey_matches = (prey_pos == pos.unsqueeze(0)).all(dim=1)
            prey_mask = prey_mask | prey_matches.unsqueeze(1)

            # Find predators at this position
            pred_matches = (pred_pos == pos.unsqueeze(0)).all(dim=1)
            pred_mask = pred_mask | pred_matches.unsqueeze(1)

        # Update energies
        prey_energy = torch.where(prey_mask, torch.zeros_like(prey_energy), prey_energy)
        pred_energy = torch.where(pred_mask, pred_energy + nutrition, pred_energy)

        return {
            self.output_variables[0]: prey_energy,
            self.output_variables[1]: pred_energy,
        }
