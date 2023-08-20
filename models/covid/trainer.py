import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F 

from simulator import get_registry, get_runner
from AgentTorch.helpers import read_config

# *************************************************************************
# Parsing command line arguments
parser = argparse.ArgumentParser(
    description="AgentTorch: design, simulate and optimize agent-based models"
)
parser.add_argument(
    "-c", "--config", help="Name of the yaml config file with the parameters."
)
# *************************************************************************

args = parser.parse_args()
config_file = args.config

print("Config file: ", config_file)

config = read_config(config_file)
registry = get_registry()

runner = get_runner(config, registry)
runner.init()

device = torch.device(runner.config['simulation_metadata']['device'])

import ipdb; ipdb.set_trace()

print('InProgress: TODO - execute the simulations!!')