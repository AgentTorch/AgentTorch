'''Command: python trainer.py --c config.yaml'''

import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F 

import sys
# sys.path.insert(0, '../../')
from simulator import get_registry, get_runner
from AgentTorch.helpers import read_config

# *************************************************************************
# Parsing command line arguments
parser = argparse.ArgumentParser(
    description="AgentTorch: million-scale, differentiable agent-based models"
)
parser.add_argument(
    "-c", "--config", default="config.yaml", help="Name of the yaml config file with the parameters."
)
# *************************************************************************

args = parser.parse_args()
config_file = args.config

print("Running experiment with config file: ", config_file)

config = read_config(config_file)
registry = get_registry()

runner = get_runner(config, registry)
runner.init()

print("Runner initialized!")

device = torch.device(runner.config['simulation_metadata']['device'])

num_episodes = runner.config['simulation_metadata']['num_episodes']
num_steps_per_episode = runner.config['simulation_metadata']['num_steps_per_episode']

for episode in range(num_episodes):
    print("Executing episode: ", episode)
    # execute forward step
    runner.step(num_steps_per_episode)
    
    # reset the state configuration
    runner.reset()