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
import sys

AGENT_TORCH_PATH = '/u/ayushc/projects/GradABM/MacroEcon/AgentTorch'

sys.path.insert(0, AGENT_TORCH_PATH)

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

# profiler_obj = profiler.profile(record_shapes=True, use_cuda=torch.cuda.is_available())
# profiler_started = False
# if not profiler_started:
#     profiler_obj.__enter__()
#     profiler_started = True

config = read_config(config_file)
registry = get_registry()
runner = get_runner(config, registry)

device = torch.device(runner.config['simulation_metadata']['device'])
num_episodes = runner.config['simulation_metadata']['num_episodes']
num_steps_per_episode = runner.config['simulation_metadata']['num_steps_per_episode']

runner.init()

opt = optim.Adam(runner.parameters(), 
                lr=runner.config['simulation_metadata']['learning_params']['lr'], 
                betas=runner.config['simulation_metadata']['learning_params']['betas'])

scheduler = optim.lr_scheduler.ExponentialLR(opt, 
                runner.config['simulation_metadata']['learning_params']['lr_gamma'])

for episode in range(num_episodes):
    opt.zero_grad()
    runner.step(num_steps_per_episode)
    
    traj = runner.trajectory[-1][-1]
    daily_infections_arr = traj['environment']['daily_infections']
    
    loss_val = daily_infections_arr.sum() # test loss for now. will be replaced after
    loss_val.backward()
    
    opt.step()
    
    pdb.set_trace()
    
    runner.reset()
    
# if profiler_started:
#     profiler_obj.__exit__(None, None, None)
            
# print(profiler_obj.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))