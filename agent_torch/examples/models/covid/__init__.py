"""COVID-19 simulation model."""

from agent_torch.core import Registry
from agent_torch.core.helpers import *

from .substeps import *

# Create and populate registry as a module-level variable
registry = Registry()
