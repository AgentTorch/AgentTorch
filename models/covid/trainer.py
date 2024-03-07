'''Command: python trainer.py --c config.yaml'''

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import pdb

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

profiler_obj = profiler.profile(record_shapes=True, use_cuda=torch.cuda.is_available())
profiler_started = False

if not profiler_started:
    profiler_obj.__enter__()
    profiler_started = True

with profiler.record_function("init_runner"):
    runner.init()

device = torch.device(runner.config['simulation_metadata']['device'])

num_episodes = runner.config['simulation_metadata']['num_episodes']
num_steps_per_episode = runner.config['simulation_metadata']['num_steps_per_episode']

for episode in range(num_episodes):
    with profiler.record_function("step_episode"):
        runner.step(num_steps_per_episode)
    
    runner.reset()
    
if profiler_started:
    profiler_obj.__exit__(None, None, None)
            
print(profiler_obj.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))