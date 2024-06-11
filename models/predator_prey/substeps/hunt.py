# substeps/hunt.py
# consumption of prey by predators

import torch
import re

from agent_torch.registry import Registry
from agent_torch.substep import SubstepObservation, SubstepAction, SubstepTransition
from agent_torch.helpers import get_by_path

def get_var(state, var):
  """
    Retrieves a value from the current state of the model.
  """
  return get_by_path(state, re.split('/', var))

@Registry.register_substep("find_targets", "policy")
class FindTargets(SubstepAction):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, state, observations):
    input_variables = self.input_variables

    prey_pos = get_var(state, input_variables['prey_pos'])
    pred_pos = get_var(state, input_variables['pred_pos'])

    # if there are any prey at the same position as a predator,
    # add them to the list of targets to kill.
    target_positions = []
    for pos in pred_pos:
      if (pos == prey_pos).all(-1).any(-1) == True:
        target_positions.append(pos)

    # pass that list of targets to the transition class.
    return {
      self.output_variables[0]: target_positions
    }

@Registry.register_substep("hunt_prey", "transition")
class HuntPrey(SubstepTransition):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, state, action):
    input_variables = self.input_variables

    prey_pos = get_var(state, input_variables['prey_pos'])
    prey_energy = get_var(state, input_variables['prey_energy'])
    pred_pos = get_var(state, input_variables['pred_pos'])
    pred_energy = get_var(state, input_variables['pred_energy'])
    nutrition = get_var(state, input_variables['nutritional_value'])

    # if there are no targets, skip the state modifications.
    if len(action['predator']['target_positions']) < 1:
      return {}

    target_positions = torch.stack(action['predator']['target_positions'], dim=0)

    # these are masks similars to the ones in `substeps/eat.py`.
    prey_energy_mask = None
    pred_energy_mask = None
    for pos in target_positions:
      pye_m = (pos == prey_pos).all(dim=1).view(-1, 1)
      if prey_energy_mask is None:
        prey_energy_mask = pye_m
      else:
        prey_energy_mask = prey_energy_mask + pye_m

      pde_m = (pos == pred_pos).all(dim=1).view(-1, 1)
      if pred_energy_mask is None:
        pred_energy_mask = pde_m
      else:
        pred_energy_mask = pred_energy_mask + pde_m

    # any prey that is marked for death should be given zero energy.
    prey_energy = prey_energy_mask*0 + (~prey_energy_mask)*prey_energy
    # any predator that has hunted should be given additional energy.
    pred_energy = pred_energy_mask*(pred_energy + nutrition) + (~pred_energy_mask)*pred_energy

    return {
      self.output_variables[0]: prey_energy,
      self.output_variables[1]: pred_energy
    }
