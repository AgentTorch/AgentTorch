import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import os

from simulator import SimulationRunner, simulation_registry
from AgentTorch.helpers import read_config

'''Command: python trainer.py --c config.yaml'''

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
if args:
    config_file = args.config
else:
#     config_file = "/Users/shashankkumar/Documents/GitHub/MacroEcon/models/macro_economics/config.yaml"
    config_file = os.path.join(os.getcwd(), 'config.yaml')
    print("Config file path: ", config_file)
    
config = read_config(config_file)
registry = simulation_registry()

runner = SimulationRunner(config, registry)
device = torch.device(runner.config['simulation_metadata']['device'])

print("Initializing runner..")
runner.init()
print(" initialized!!!!!!!")
print("---"*8)

loss_log = []

optimizer = optim.Adam(runner.parameters(), 
                lr=runner.config['simulation_metadata']['learning_params']['lr'], 
                betas=runner.config['simulation_metadata']['learning_params']['betas'])
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 
                runner.config['simulation_metadata']['learning_params']['lr_gamma'])

num_steps_per_episode = runner.config["simulation_metadata"]["num_steps_per_episode"]

# execute all simulation episodes with a utility function in OpDynRunner
runner.execute()

torch.save(runner.state_dict(), runner.config['simulation_metadata']['learning_params']['model_path'])

print("Execution complete")