# substeps/move.py
# random movement of predator and prey

import math
import torch
import re
import random

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


def get_neighbors(pos, adj_grid, bounds):
    """
    Returns a list of neighbours for each position passed in the given
    `pos` tensor, using the adjacency matrix passed in `adj_grid`.
    """
    x, y = pos
    max_x, max_y = bounds

    # calculate the node number from the x, y coordinate.
    # each item (i, j) in the adjacency matrix, if 1 depicts
    # that i is connected to j and vice versa.
    node = (max_y * x) + y
    conn = adj_grid[node]

    neighbors = []
    for idx, cell in enumerate(conn):
        # if connected, calculate the (x, y) coords of the other
        # node and add it to the list of neighbors.
        if cell == 1:
            c = (int)(idx % max_y)
            r = math.floor((idx - c) / max_y)

            neighbors.append([torch.tensor(r), torch.tensor(c)])

    return torch.tensor(neighbors)


@Registry.register_substep("find_neighbors", "observation")
class FindNeighbors(SubstepObservation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state):
        input_variables = self.input_variables

        bounds = get_var(state, input_variables["bounds"])
        adj_grid = get_var(state, input_variables["adj_grid"])
        positions = get_var(state, input_variables["positions"])

        # for each agent (prey/predator) find the adjacent cells and pass
        # them on to the policy class.
        possible_neighbors = []
        for pos in positions:
            possible_neighbors.append(get_neighbors(pos, adj_grid, bounds))

        return {self.output_variables[0]: possible_neighbors}


@Registry.register_substep("decide_movement", "policy")
class DecideMovement(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, observations):
        input_variables = self.input_variables

        positions = get_var(state, input_variables["positions"])
        energy = get_var(state, input_variables["energy"])
        possible_neighbors = observations["possible_neighbors"]

        # randomly choose the next position of the agent. if the agent
        # has non-positive energy, don't let it move.
        next_positions = []
        for idx, pos in enumerate(positions):
            next_positions.append(
                random.choice(possible_neighbors[idx]) if energy[idx] > 0 else pos
            )

        return {self.output_variables[0]: torch.stack(next_positions, dim=0)}


@Registry.register_substep("update_positions", "transition")
class UpdatePositions(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, action):
        input_variables = self.input_variables

        prey_energy = get_var(state, input_variables["prey_energy"])
        pred_energy = get_var(state, input_variables["pred_energy"])
        prey_work = get_var(state, input_variables["prey_work"])
        pred_work = get_var(state, input_variables["pred_work"])

        # reduce the energy of the agent by the work required by them
        # to take one step.
        prey_energy = prey_energy + torch.full(
            prey_energy.shape, -1 * (prey_work.item())
        )
        pred_energy = pred_energy + torch.full(
            pred_energy.shape, -1 * (pred_work.item())
        )

        return {
            self.output_variables[0]: action["prey"]["next_positions"],
            self.output_variables[1]: prey_energy,
            self.output_variables[2]: action["predator"]["next_positions"],
            self.output_variables[3]: pred_energy,
        }
