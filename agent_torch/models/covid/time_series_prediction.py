from __future__ import annotations

from sympy import N
from zmq import device
'''Command: python trainer.py --c yamls/config_opt_llm.yaml'''
import warnings
warnings.filterwarnings("ignore")

import argparse
from epiweeks import Week
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
# AGENT_TORCH_PATH = '/u/ngkuru/ship/MacroEcon/AgentTorch'
AGENT_TORCH_PATH = '/Users/shashankkumar/Documents/GitHub/AgentTorchMain/agent_torch/models/covid'

import sys
sys.path.insert(0, AGENT_TORCH_PATH)
from torch.utils.data import DataLoader
from calibration.utils.misc import week_num_to_epiweek



from calibration.calibnn import CalibNN, ImprovedCovidPredictor,SimpleCovidPredictor

from calibration.utils.data import NN_INPUT_WEEKS, get_dataloader, get_labels
from calibration.utils.feature import Feature
from calibration.utils.misc import name_to_neighborhood
from calibration.utils.neighborhood import Neighborhood

# Data normalization function
def normalize_data(data):
    return (data - data.mean()) / data.std()

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
config_file = "/Users/shashankkumar/Documents/GitHub/MacroEcon/models/covid/yamls/config_opt_llm.yaml"
print("Running experiment with config file: ", config_file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
betas = (0.9, 0.999)


EPIWEEK_START: Week = week_num_to_epiweek(202212)
NUM_WEEKS: int =  28 // 7

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
NEIGHBORHOOD = name_to_neighborhood("Astoria")

# set up model
learn_model_calibnn = CalibNN(
    metas_train_dim=len(Neighborhood),
    X_train_dim=len(FEATURE_LIST),
    device=device,
    training_weeks=NN_INPUT_WEEKS,
    out_dim=1,
    scale_output="abm-covid",
).to(device)

learn_model_simple = SimpleCovidPredictor(input_dim=len(FEATURE_LIST))

improved_model = ImprovedCovidPredictor(input_dim=len(FEATURE_LIST),meta_dim=len(Neighborhood))
model = improved_model
    # set up loss function and optimizer
# loss_function = torch.nn.MSELoss().to(device)
loss_function  = nn.HuberLoss().to(device)
opt = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for episode in range(10):    
        # reset gradients from previous iteration
        print(f"\nrunning episode {episode}...")
        opt.zero_grad()

        # get weekly number of infections from daily number of infections
        dataloader: DataLoader = get_dataloader(
                NEIGHBORHOOD,
                EPIWEEK_START + episode,
                NUM_WEEKS,
                FEATURE_LIST,
            )
        for metadata, features in dataloader:
            r0_values = model(features, metadata)
        r0_values = r0_values
        target_weekly_cases = get_labels(NEIGHBORHOOD, EPIWEEK_START + episode, 4, LABEL_FEATURE)
        target_weekly_cases = target_weekly_cases.to(device)
        target_weekly_cases = normalize_data(target_weekly_cases)
        # calculate the loss from the target cases
        loss_val = loss_function(r0_values, target_weekly_cases)
        loss_val.backward()
        print(f"predicted number of cases: {r0_values}, actual number of cases: {target_weekly_cases}, loss: {loss_val}")


        # run the optimization step, and clear simulation
        # allocated3, reserved3 = memory_checkpoint(name="3")
        opt.step()
        # print(torch.cuda.memory_summary())
        # print("---------------------------------")
        torch.cuda.empty_cache()
        total_loss += loss_val.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)
