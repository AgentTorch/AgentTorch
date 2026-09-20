# substeps/eat.py
# consumption of grass by prey

import torch
import re

from agent_torch.core.registry import Registry
from agent_torch.core.substep import (
    SubstepObservation,
    SubstepAction,
    SubstepTransition,
)
from agent_torch.core.helpers import get_by_path


def get_var(state, var):
    """
    Retrieves a value from the current state of the model.
    """
    return get_by_path(state, re.split("/", var))


@Registry.register_substep("find_eatable_grass", "policy")
class FindEatableGrass(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, observations):
        input_variables = self.input_variables

        bounds = get_var(state, input_variables["bounds"])
        positions = get_var(state, input_variables["positions"])
        grass_growth = get_var(state, input_variables["grass_growth"])

        # if the grass is fully grown, i.e., its growth_stage is equal to
        # 1, then it can be consumed by prey.
        eatable_grass_positions = []
        max_x, max_y = bounds
        for pos in positions:
            x, y = pos
            node = (max_y * x) + y
            if grass_growth[node] == 1:
                eatable_grass_positions.append(pos)

        # pass on the consumable grass positions to the transition class.
        return {self.output_variables[0]: eatable_grass_positions}


@Registry.register_substep("eat_grass", "transition")
class EatGrass(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, action):
        input_variables = self.input_variables

        bounds = get_var(state, input_variables["bounds"])
        prey_pos = get_var(state, input_variables["prey_pos"])
        energy = get_var(state, input_variables["energy"])
        nutrition = get_var(state, input_variables["nutrition"])
        grass_growth = get_var(state, input_variables["grass_growth"])
        growth_countdown = get_var(state, input_variables["growth_countdown"])
        regrowth_time = get_var(state, input_variables["regrowth_time"])

        # if no grass can be eaten, skip modifying the state.
        if (action["prey"] is None) or (len(action["prey"]["eatable_grass_positions"]) < 1):
            return {}

        eatable_grass_positions = torch.stack(
            action["prey"]["eatable_grass_positions"], dim=0
        )

        max_x, max_y = bounds
        energy_mask = None
        grass_mask = torch.zeros(*grass_growth.shape)
        countdown_mask = torch.zeros(*growth_countdown.shape)

        # for each consumable grass, figure out if any prey agent is at
        # that position. if yes, then mark that position in the mask as
        # true. also, for all the grass that will be consumed, reset the
        # growth stage.
        for pos in eatable_grass_positions:
            x, y = pos
            node = (max_y * x) + y

            # TODO: make sure dead prey cannot eat
            e_m = (pos == prey_pos).all(dim=1).view(-1, 1)
            if energy_mask is None:
                energy_mask = e_m
            else:
                energy_mask = e_m + energy_mask

            grass_mask[node] = -1
            countdown_mask[node] = regrowth_time - growth_countdown[node]

        # energy + nutrition adds the `nutrition` tensor to all elements in
        # the energy tensor. the (~energy_mask) ensures that the change is
        # undone for those prey that did not consume grass.
        energy = energy_mask * (energy + nutrition) + (~energy_mask) * energy

        # these masks use simple addition to make changes to the original
        # values of the tensors.
        grass_growth = grass_growth + grass_mask
        growth_countdown = growth_countdown + countdown_mask

        return {
            self.output_variables[0]: energy,
            self.output_variables[1]: grass_growth,
            self.output_variables[2]: growth_countdown,
        }
