from __future__ import annotations
'''Command: python trainer.py --c yamls/config_opt_llm.yaml'''
import warnings
warnings.filterwarnings("ignore")

import argparse
from epiweeks import Week
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from utils.misc import week_num_to_epiweek

# AGENT_TORCH_PATH = '/u/ngkuru/ship/MacroEcon/AgentTorch'
AGENT_TORCH_PATH = '/u/ayushc/projects/GradABM/MacroEcon/AgentTorch'

import sys
sys.path.insert(0, AGENT_TORCH_PATH)

from simulator import get_registry, get_runner
from AgentTorch.helpers import read_config
from calibnn import CalibNN, LearnableParams

from utils.data import NN_INPUT_WEEKS, get_dataloader, get_labels
from utils.feature import Feature
from utils.misc import name_to_neighborhood
from utils.neighborhood import Neighborhood

from AgentTorch.helpers import memory_checkpoint

# *************************************************************************
# Parsing command line arguments
parser = argparse.ArgumentParser(
    description="AgentTorch: million-scale, differentiable agent-based models"
)
parser.add_argument(
    "-c", "--config", default="config_opt_llm.yaml", help="Name of the yaml config file with the parameters."
)
# *************************************************************************

args = parser.parse_args()
config_file = args.config
print("Running experiment with config file: ", config_file)

CALIB_MODE = 'calibNN' # i -> internal_param; external_param -> nn.Parameter; learnable_param -> learnable_parameters; nn -> CalibNN

config = read_config(config_file)
registry = get_registry()
runner = get_runner(config, registry)

device = torch.device(runner.config["simulation_metadata"]["device"])
num_episodes = runner.config["simulation_metadata"]["num_episodes"]
NUM_STEPS_PER_EPISODE = runner.config["simulation_metadata"]["num_steps_per_episode"]

runner.init()

named_params_learnable = [(name, param) for (name, param) in runner.named_parameters() if param.requires_grad]
print("named learnable_params: ", named_params_learnable)

learning_rate = runner.config['simulation_metadata']['learning_params']['lr']
betas = runner.config['simulation_metadata']['learning_params']['betas']

if CALIB_MODE == 'internal_param':
    learnable_params = [param for param in runner.parameters() if param.requires_grad]
    opt = optim.Adam(learnable_params, lr=learning_rate,betas=betas)
elif CALIB_MODE == 'external_param':
    R = nn.Parameter(torch.tensor([4.10]))
    opt = optim.Adam([R], lr=learning_rate,betas=betas)
    runner.initializer.transition_function['0']['new_transmission'].learnable_args.R2 = R
elif CALIB_MODE == 'learnable_param':
    learn_model = LearnableParams(num_params=1, device=device)
    opt = optim.Adam(learn_model.parameters(), lr=learning_rate, betas=betas)
elif CALIB_MODE == "calibNN":
    # set the epiweeks to simulate
    EPIWEEK_START: Week = week_num_to_epiweek(runner.config["simulation_metadata"]["START_WEEK"])
    NUM_WEEKS: int = runner.config["simulation_metadata"]["num_steps_per_episode"] // 7

    # set up variables
    FEATURE_LIST = [
        Feature.RETAIL_CHANGE,
        Feature.GROCERY_CHANGE,
        Feature.PARKS_CHANGE,
        Feature.TRANSIT_CHANGE,
        Feature.WORK_CHANGE,
        Feature.RESIDENTIAL_CHANGE,
        Feature.CASES,
    ]
    LABEL_FEATURE = Feature.CASES
    NEIGHBORHOOD = name_to_neighborhood(config["simulation_metadata"]["NEIGHBORHOOD"])

    # set up model
    learn_model = CalibNN(
        metas_train_dim=len(Neighborhood),
        X_train_dim=len(FEATURE_LIST),
        device=device,
        training_weeks=NN_INPUT_WEEKS,
        out_dim=1,
        scale_output="abm-covid",
    ).to(device)

    # set up loss function and optimizer
    loss_function = torch.nn.MSELoss().to(device)
    opt = optim.Adam(learn_model.parameters(), lr=learning_rate, betas=betas)

def _get_parameters(CALIB_MODE):
    if CALIB_MODE == "learnable_param":
        new_R = learn_model()
        print("R shape: ", new_R.shape)
        return new_R

    if CALIB_MODE == "calibNN":
        # get R values for the epiweeks
        dataloader: DataLoader = get_dataloader(
            NEIGHBORHOOD,
            EPIWEEK_START,
            NUM_WEEKS,
            FEATURE_LIST,
        )
        for metadata, features in dataloader:
            r0_values = learn_model(features, metadata)[:, 0, 0]

        return r0_values

def _set_parameters(new_R):
    # print("SET PARAMETERS ONLY WORKS FOR R0 for now!")
    '''Only sets R value for now..'''
    runner.initializer.transition_function['0']['new_transmission'].external_R = new_R

for episode in range(num_episodes):
    print(f"\nrunning episode {episode}...")

    # get the r0 predictions for the episode
    r0_values = _get_parameters(CALIB_MODE)
    _set_parameters(r0_values)
    print(f"r0 values: {r0_values}")
    
    if episode >=1:
        runner.reset()

    allocated1, reserved1 = memory_checkpoint(name="1")

    # run the simulation        
    opt.zero_grad()
    runner.step(NUM_STEPS_PER_EPISODE)

    allocated2, reserved2 = memory_checkpoint(name="2")

    # get daily number of infections
    traj = runner.state_trajectory[-1][-1]
    daily_infections_arr = traj["environment"]["daily_infected"].to(device)

    # get weekly number of infections from daily number of infections
    predicted_weekly_cases = (
        daily_infections_arr.reshape(-1, 7).sum(axis=1).to(dtype=torch.float32)
    )
    target_weekly_cases = get_labels(NEIGHBORHOOD, EPIWEEK_START, NUM_WEEKS, LABEL_FEATURE)
    target_weekly_cases = target_weekly_cases.to(device)

    # calculate the loss from the target cases
    loss_val = loss_function(predicted_weekly_cases, target_weekly_cases)
    loss_val.backward()
    print(f"predicted number of cases: {predicted_weekly_cases}, actual number of cases: {target_weekly_cases}, loss: {loss_val}")

    allocated3, reserved3 = memory_checkpoint(name="3")

    # run the optimization step, and clear simulation
    opt.step()
    print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()
    print("---------------------------------")
