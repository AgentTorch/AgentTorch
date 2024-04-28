MODEL_PATH = '/Users/shashankkumar/Documents/GitHub/MacroEcon/models'
AGENT_TORCH_PATH = '/Users/shashankkumar/Documents/GitHub/MacroEcon/AgentTorch'

import pickle
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, AGENT_TORCH_PATH)
sys.path.append(MODEL_PATH)
import argparse
from epiweeks import Week
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader

from AgentTorch import Runner
from AgentTorch.helpers import read_config
from simulator import SimulationRunner, simulation_registry

from macro_economics.calibnn import CalibNN, LearnableParams

from macro_economics.utils.data import NN_INPUT_WEEKS, get_dataloader, get_labels
from macro_economics.utils.feature import Feature
from macro_economics.utils.misc import name_to_neighborhood
from macro_economics.utils.neighborhood import Neighborhood
from macro_economics.utils.misc import week_num_to_epiweek

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

config_file = '/Users/shashankkumar/Documents/GitHub/MacroEcon/models/macro_economics/yamls/config.yaml'
config = read_config(config_file)
registry = simulation_registry()

runner = Runner(config, registry)

device = torch.device(runner.config['simulation_metadata']['device'])
CALIB_MODE = 'calibNN' # i -> internal_param; external_param -> nn.Parameter; learnable_param -> learnable_parameters; nn -> CalibNN
num_episodes = runner.config["simulation_metadata"]["num_episodes"]
NUM_STEPS_PER_EPISODE = runner.config["simulation_metadata"]["num_steps_per_episode"]

runner.init()
loss_log = []

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
    NUM_WEEKS: int = runner.config["simulation_metadata"]["num_steps_per_episode"]*4

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
        out_dim=5,
        scale_output="abm-covid",
    ).to(device)

    # set up loss function and optimizer
    # loss_function = torch.nn.MSELoss().to(device)
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
            calib_values = learn_model(features, metadata)

        return calib_values

def _set_parameters(new_values):
    # print("SET PARAMETERS ONLY WORKS FOR UAC in labor-market for now!")
    '''Only sets UAC value for now..'''
    runner.initializer.transition_function['2']['update_macro_rates'].external_UAC = new_values

def _get_unemployment_labels(num_steps_per_episode=1):
    data_path = '/Users/shashankkumar/Documents/GitHub/MacroEcon/simulator_data/NYC/brooklyn_unemp.csv'
    df = pd.read_csv(data_path)
    df.sort_values(by=['year','month'],ascending=True,inplace=True)
    arr = df['unemployment_rate'].values
    unemployment_test_dataset = torch.from_numpy(arr).to(device)
    unemployment_test_dataset = unemployment_test_dataset
    unemployment_test_dataset = unemployment_test_dataset[:num_steps_per_episode].float().squeeze()

    return unemployment_test_dataset
state_data_dict = {}
NUM_TEST_STEPS = 3
for episode in range(num_episodes):    
    # reset gradients from previous iteration
    print(f"\nrunning episode {episode}...")
    opt.zero_grad()
    if episode >=1:
        runner.reset()

    # get the r0 predictions for the episode
    calib_values = _get_parameters(CALIB_MODE)
    avg_month_value = calib_values.reshape(-1, 4,5).mean(dim=1)
    _set_parameters(avg_month_value)
    print(f"calib values: {calib_values}")

    runner.step(NUM_STEPS_PER_EPISODE)

    predicted_month_unemployment_rate = runner.state_trajectory[-1][-1]['environment']['U'].squeeze()
    predicted_month_unemployment_rate_train = predicted_month_unemployment_rate[:-NUM_TEST_STEPS]
    target_month_unemployment_rate = _get_unemployment_labels(NUM_STEPS_PER_EPISODE)
    target_month_unemployment_rate_train = target_month_unemployment_rate[:-NUM_TEST_STEPS]

    # calculate the loss from the target cases
    loss_val = loss_function(predicted_month_unemployment_rate, target_month_unemployment_rate)
    loss_val.backward()
    print(f"predicted rate: {predicted_month_unemployment_rate_train}, actual rate: {target_month_unemployment_rate_train}, loss: {loss_val}")

    opt.step()
    
    #testing
    with torch.no_grad():
        predicted_month_unemployment_rate_test = predicted_month_unemployment_rate[-NUM_TEST_STEPS:]
        target_month_unemployment_rate_test = target_month_unemployment_rate[-NUM_TEST_STEPS:]
        loss_test = loss_function(predicted_month_unemployment_rate_test, target_month_unemployment_rate_test)
        print(f"predicted rate test: {predicted_month_unemployment_rate_test}, actual rate test: {target_month_unemployment_rate_test}, loss_test: {loss_test}")
        current_episode_state_data_dict = {
                "environment": {id : state_traj['environment'] for id,state_traj in enumerate(runner.state_trajectory[-1][::NUM_STEPS_PER_EPISODE])},
                "agents": {id : state_traj['agents'] for id,state_traj in enumerate(runner.state_trajectory[-1][::NUM_STEPS_PER_EPISODE])}
                }
        state_data_dict[episode] = current_episode_state_data_dict
    torch.cuda.empty_cache()

# save the state trace
with open('state_data_dict.pkl', 'wb') as f:
            pickle.dump(state_data_dict, f)

print("Training complete!")