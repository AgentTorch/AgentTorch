# substeps/grow.py
# growth of grass

import torch
import re

from AgentTorch.registry import Registry
from AgentTorch.substep import SubstepObservation, SubstepAction, SubstepTransition
from AgentTorch.helpers import get_by_path

def get_var(state, var):
  """
    Retrieves a value from the current state of the model.
  """
  return get_by_path(state, re.split('/', var))

@Registry.register_substep("grow_grass", "transition")
class GrowGrass(SubstepTransition):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, state, action):
    input_variables = self.input_variables

    grass_growth = get_var(state, input_variables['grass_growth'])
    growth_countdown = get_var(state, input_variables['growth_countdown'])

    # reduce all countdowns by 1 unit of time.
    growth_countdown_mask = torch.full(growth_countdown.shape, -1)
    growth_countdown = growth_countdown + growth_countdown_mask

    # if the countdown has reached zero, set the growth stage to 1,
    # otherwise, keep it zero.
    grass_growth_mask = (growth_countdown <= 0).all(dim=1)
    grass_growth = grass_growth_mask*(1) + (~grass_growth_mask)*(0)

    return {
      self.output_variables[0]: grass_growth.view(-1, 1),
      self.output_variables[1]: growth_countdown
    }
