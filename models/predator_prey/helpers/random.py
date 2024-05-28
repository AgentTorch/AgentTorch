# helpers/random.py
# functions that initialize properties of agents/objects

import math
import torch
from agent_torch.registry import Registry

@Registry.register_helper('random_float', 'initialization')
def random_float(shape, params):
  """
    Generates a `Tensor` of the given shape, with random floating point
    numbers in between and including the lower and upper limit.
  """

  max = params['upper_limit'] + 1 # include max itself.
  min = params['lower_limit']

  # torch.rand returns a tensor of the given shape, filled with
  # floating point numbers in the range (0, 1]. multiplying the
  # tensor by max - min and adding the min value ensure it's
  # within the given range.
  tens = (max - min) * torch.rand(shape) + min

  return tens

@Registry.register_helper('random_int', 'initialization')
def random_int(shape, params):
  """
    Generates a `Tensor` of the given shape, with random integers in
    between and including the lower and upper limit.
  """

  max = math.floor(params['upper_limit'] + 1) # include max itself.
  min = math.floor(params['lower_limit'])

  # torch.randint returns the tensor we need.
  tens = torch.randint(min, max, shape)

  return tens
