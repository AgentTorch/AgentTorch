"""
Vectorized implementation of the grass growth substep.
"""

import torch
import re
from torch.func import vmap

from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers.general import get_by_path
from agent_torch.core.vectorization import vectorized


def get_var(state, var):
    """
    Retrieves a value from the current state of the model.
    """
    return get_by_path(state, re.split("/", var))


def process_single_grass(growth_countdown_val):
    """
    Process a single grass tile's growth.
    This function will be vectorized with vmap.

    Args:
        growth_countdown_val: Current countdown value

    Returns:
        tuple: (new_growth_stage, new_countdown)
    """
    # Decrement countdown
    new_countdown = growth_countdown_val - 1

    # Set growth stage based on countdown
    # Using torch.where instead of conditional for vmap compatibility
    new_growth_stage = torch.where(
        new_countdown <= 0,
        torch.tensor(1.0, dtype=new_countdown.dtype),
        torch.tensor(0.0, dtype=new_countdown.dtype),
    )

    return new_growth_stage, new_countdown


@Registry.register_substep("grow_grass", "transition")
class GrowGrassVmap(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vectorized = True

    def forward(self, state, action):
        input_variables = self.input_variables
        grass_growth = get_var(state, input_variables["grass_growth"])
        growth_countdown = get_var(state, input_variables["growth_countdown"])

        # An alternative approach without using process_single_grass and vmap
        # This directly applies operations to the whole tensors

        # Decrement all countdowns
        new_countdowns = growth_countdown - 1

        # Set growth stage to 1 where countdown <= 0, keep at 0 otherwise
        new_growth_stages = torch.where(
            new_countdowns <= 0,
            torch.ones_like(grass_growth),
            torch.zeros_like(grass_growth),
        )

        return {
            self.output_variables[0]: new_growth_stages,
            self.output_variables[1]: new_countdowns,
        }
