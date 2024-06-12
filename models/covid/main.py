from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

def debug():
    import os
    import sys
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    package_root_directory = os.path.dirname(os.path.dirname(current_directory))
    sys.path.append(package_root_directory)

try:
    import agent_torch
except:
    debug()
    import agent_torch

import argparse
from simulator import get_registry, get_runner
from agent_torch.helpers import read_config
import torch
from tqdm import trange

# *************************************************************************
# Parsing command line arguments
parser = argparse.ArgumentParser(
    description="AgentTorch: million-scale, differentiable agent-based models"
)
parser.add_argument(
    "-c", "--config", default="yamls/config.yaml", help="config file with simulation parameters"
)
# *************************************************************************

args = parser.parse_args()
config_file = args.config

config = read_config(config_file)
registry = get_registry()
runner = get_runner(config, registry)

device = torch.device(runner.config["simulation_metadata"]["device"])
num_episodes = runner.config["simulation_metadata"]["num_episodes"]
num_steps_per_episode = runner.config["simulation_metadata"]["num_steps_per_episode"]

runner.init()

for episode in trange(num_episodes):
    print(f"\nrunning episode {episode}...")
    runner.step(num_steps_per_episode)

    runner.reset()