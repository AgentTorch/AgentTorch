import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F 

from simulator import OpDynRunner, opdyn_registry
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

config = read_config(config_file)
registry = opdyn_registry()

runner = OpDynRunner(config, registry)
device = torch.device(runner.config['simulation_metadata']['device'])
runner.init()

optimizer = optim.Adam(runner.parameters(), lr=1e-1)

loss_log = []
print('IN PROGRESS: To fix the RL optimization logic!!!!!!')

import ipdb; ipdb.set_trace()

runner.execute()