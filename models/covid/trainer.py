'''Command: python trainer.py --c config.yaml'''

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import torch.optim as optim
import sys
import torch.nn as nn

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

CALIB_MODE = 'NN'

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

named_params_learnable = [(name, param) for (name, param) in runner.named_parameters() if param.requires_grad]
print("named learnable_params: ", named_params_learnable)

learning_rate = runner.config['simulation_metadata']['learning_params']['lr']
betas = runner.config['simulation_metadata']['learning_params']['betas']

if CALIB_MODE == 'i':
    learnable_params = [param for param in runner.parameters() if param.requires_grad]
    opt = optim.Adam(learnable_params, lr=learning_rate,betas=betas)
else:
    R = nn.Parameter(torch.tensor([4.10]))
    opt = optim.Adam([R], lr=learning_rate,betas=betas)
    # breakpoint()
    runner.initializer.transition_function['0']['new_transmission'].learnable_args.R2 = R
    # runner.initializer.transition_function['0']['new_transmission'].learnable_args.R2.copy_(R)

for episode in range(num_episodes):
    opt.zero_grad()
    runner.step(num_steps_per_episode)
    
    traj = runner.state_trajectory[-1][-1]
    daily_infections_arr = traj['environment']['daily_infected'] #()
    
    loss_val = daily_infections_arr.sum() # test loss for now. will be replaced after
    loss_val.backward()

    # Check the gradients for all parameters in the optimizer
    for param_group in opt.param_groups:
        for param in param_group['params']:
            print(f"Parameter: {param.data}, Gradient: {param.grad}")

    breakpoint()
    
    opt.step()
    runner.reset()