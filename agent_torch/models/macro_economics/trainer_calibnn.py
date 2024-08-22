MODEL_PATH = "/Users/shashankkumar/Documents/GitHub/MacroEcon/models"
AGENT_TORCH_PATH = "/Users/shashankkumar/Documents/GitHub/MacroEcon/AgentTorch"

import sys

sys.path.insert(0, AGENT_TORCH_PATH)
sys.path.append(MODEL_PATH)
import os

os.environ["DSP_CACHEBOOL"] = "False"

import random
import numpy as np

random.seed(42)
np.random.seed(42)

import pickle
import warnings

from AgentTorch.utils import initialise_wandb

warnings.filterwarnings("ignore")

import argparse
from epiweeks import Week
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader

from AgentTorch import Runner
from AgentTorch.helpers import read_config
from macro_economics.simulator import SimulationRunner, simulation_registry

from macro_economics.calibnn import CalibNN, LearnableParams

from macro_economics.utils.data import NN_INPUT_WEEKS, get_dataloader, get_labels
from macro_economics.utils.feature import Feature
from macro_economics.utils.misc import name_to_neighborhood
from macro_economics.utils.neighborhood import Neighborhood
from macro_economics.utils.misc import week_num_to_epiweek

import wandb
import matplotlib.pyplot as plt

"""Command: python trainer.py --c config.yaml"""

# *************************************************************************
# Parsing command line arguments
parser = argparse.ArgumentParser(
    description="AgentTorch: design, simulate and optimize agent-based models"
)
parser.add_argument(
    "-c", "--config", help="Name of the yaml config file with the parameters."
)
# *************************************************************************


def wandb_log(name, value):
    wandb.log({name: value})


def plot_predictions(predicted, target, title):
    plt.plot(predicted, label="Predicted")
    plt.plot(target, label="Actual")
    plt.title(title)
    plt.legend()
    # plt.show()
    wandb.log({title: plt})


def plot_labor_force(labor_force_data):
    labor_force_data = labor_force_data.tolist()
    labor_force_pct_change = [
        (labor_force_data[i] - labor_force_data[i - 1]) / labor_force_data[i]
        for i in range(1, len(labor_force_data))
    ]
    plt.plot(labor_force_pct_change)
    plt.title("Labor Force")
    wandb.log({"Labor Force": plt})


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, interm_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, interm_dim)
        self.linear2 = nn.Linear(interm_dim, output_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = nn.ReLU()(out)
        out = self.linear2(out)
        out = nn.Sigmoid()(out)
        return out


def get_labor_step_nn(input_dim, interm_dim, output_dim):
    nn = LinearRegressionModel(input_dim, interm_dim, output_dim)


class Calibrator:
    def __init__(self, runner):
        self.runner = runner

        self.CALIB_MODE = "calibNN"
        self.NUM_TEST_STEPS = 5
        self.num_episodes = self.runner.config["simulation_metadata"]["num_episodes"]
        self.NUM_STEPS_PER_EPISODE = self.runner.config["simulation_metadata"][
            "num_steps_per_episode"
        ]
        self.learning_rate = self.runner.config["simulation_metadata"][
            "learning_params"
        ]["lr"]
        self.betas = self.runner.config["simulation_metadata"]["learning_params"][
            "betas"
        ]
        self.device = self.runner.config["simulation_metadata"]["device"]
        self.population_dir = self.runner.config["simulation_metadata"][
            "population_dir"
        ]
        # self.unemp_csv_path = self.runner.config["simulation_metadata"]["unemp_csv_path"]

        self.brooklyn_unemp_csv_path = self.runner.config["simulation_metadata"][
            "brooklyn_unemp_path"
        ]
        self.queens_unemp_csv_path = self.runner.config["simulation_metadata"][
            "queens_unemp_path"
        ]
        self.manhattan_unemp_csv_path = self.runner.config["simulation_metadata"][
            "manhattan_unemp_path"
        ]
        self.bronx_unemp_csv_path = self.runner.config["simulation_metadata"][
            "bronx_unemp_path"
        ]
        self.staten_island_unemp_csv_path = self.runner.config["simulation_metadata"][
            "staten_island_unemp_path"
        ]

        self.expt_mode = self.runner.config["simulation_metadata"]["expt_mode"]
        num_agents = self.runner.config["simulation_metadata"]["num_agents"]
        exp_name = (
            "GENPOP"
            + "_"
            + self.expt_mode
            + "_"
            + str(num_agents)
            + "_agents"
            + "_"
            + "without_stimulus"
        )
        initialise_wandb(
            entity="blankpoint",
            project="GENPOP",
            name=f"{exp_name}",
            config={
                "learning_rate": self.runner.config["simulation_metadata"][
                    "learning_params"
                ]["lr"],
                "epochs": self.runner.config["simulation_metadata"]["num_episodes"],
                "experiment_mode": self.expt_mode,
            },
        )

        self.setup_model()

    def setup_model(self):

        if self.CALIB_MODE == "internal_param":
            self.learnable_params = [
                param for param in self.runner.parameters() if param.requires_grad
            ]
            self.opt = optim.Adam(
                self.learnable_params, lr=self.learning_rate, betas=self.betas
            )
        elif self.CALIB_MODE == "external_param":
            R = torch.nn.Parameter(torch.tensor([4.10]))
            self.opt = optim.Adam([R], lr=self.learning_rate, betas=self.betas)
            self.runner.initializer.transition_function["0"][
                "new_transmission"
            ].learnable_args.R2 = R
        elif self.CALIB_MODE == "learnable_param":
            self.learn_model = LearnableParams(num_params=1, device=self.device)
            self.opt = optim.Adam(
                self.learn_model.parameters(), lr=self.learning_rate, betas=self.betas
            )
        elif self.CALIB_MODE == "calibNN":
            NUM_WEEKS = (
                self.runner.config["simulation_metadata"]["num_steps_per_episode"] * 4
            )
            FEATURE_LIST = [
                Feature.RETAIL_CHANGE,
                Feature.GROCERY_CHANGE,
                Feature.PARKS_CHANGE,
                Feature.TRANSIT_CHANGE,
                Feature.WORK_CHANGE,
                Feature.RESIDENTIAL_CHANGE,
                Feature.CASES,
            ]

            self.learn_model = CalibNN(
                metas_train_dim=len(Neighborhood),
                X_train_dim=len(FEATURE_LIST),
                device=self.device,
                training_weeks=NUM_WEEKS,
                out_dim=15,
                scale_output="abm-covid",
            ).to(self.device)
            self.initial_claims_weight = torch.nn.Parameter(torch.tensor([0.2]))
            self.labor_force_participation_rate_weight = torch.nn.Parameter(
                torch.tensor([0.02233])
            )
            self.loss_function = torch.nn.MSELoss().to(self.device)
            self.opt = optim.Adam(
                list(self.learn_model.parameters()),
                lr=self.learning_rate,
                betas=self.betas,
            )
            # self.opt = optim.Adam(list(self.learn_model.parameters()) + [self.initial_claims_weight,self.labor_force_participation_rate_weight], lr=self.learning_rate, betas=self.betas)
            if self.expt_mode == "LLM_WITH_SIM_INPUTS":
                self.load_calibnn()

    def save_learned_model(self):
        torch.save(
            self.learn_model.state_dict(),
            self.runner.config["simulation_metadata"]["learned_model_path"],
        )

    def load_learned_model(self):
        self.learn_model.load_state_dict(
            torch.load(self.runner.config["simulation_metadata"]["learned_model_path"])
        )
        self.learn_model.eval()

    def _get_parameters(self):
        if self.CALIB_MODE == "learnable_param":
            new_R = self.learn_model()
            ##print("R shape: ", new_R.shape)
            return new_R

        if self.CALIB_MODE == "calibNN":
            NEIGHBORHOOD = name_to_neighborhood(
                self.runner.config["simulation_metadata"]["NEIGHBORHOOD"]
            )
            EPIWEEK_START = week_num_to_epiweek(
                self.runner.config["simulation_metadata"]["START_WEEK"]
            )
            NUM_WEEKS = (
                self.runner.config["simulation_metadata"]["num_steps_per_episode"] * 4
            )
            FEATURE_LIST = [
                Feature.RETAIL_CHANGE,
                Feature.GROCERY_CHANGE,
                Feature.PARKS_CHANGE,
                Feature.TRANSIT_CHANGE,
                Feature.WORK_CHANGE,
                Feature.RESIDENTIAL_CHANGE,
                Feature.CASES,
            ]

            dataloader = get_dataloader(
                NEIGHBORHOOD, EPIWEEK_START, NUM_WEEKS, FEATURE_LIST
            )
            for metadata, features in dataloader:
                calib_values = self.learn_model(features, metadata)

            return (
                calib_values,
                self.initial_claims_weight,
                self.labor_force_participation_rate_weight,
            )

    def _set_parameters(
        self, external_UAC, initial_claims_weight, labor_force_participation_rate_weight
    ):
        self.runner.initializer.transition_function["2"][
            "update_macro_rates"
        ].external_UAC = external_UAC
        self.runner.initializer.transition_function["2"][
            "update_macro_rates"
        ].initial_claims_weight = initial_claims_weight
        self.runner.initializer.transition_function["2"][
            "update_macro_rates"
        ].labor_force_participation_rate_weight = labor_force_participation_rate_weight

    def _load_unemployment_labels(self, csv_path, num_steps_per_episode=1):
        data_path = csv_path
        df = pd.read_csv(data_path)
        df.sort_values(by=["year", "month"], ascending=True, inplace=True)
        arr = df["unemployment_rate"].values
        unemployment_test_dataset = torch.from_numpy(arr).to(self.device)
        unemployment_test_dataset = (
            unemployment_test_dataset[:num_steps_per_episode].float().squeeze()
        )

        return unemployment_test_dataset

    def get_unemployment_labels(self, num_steps_per_episode=1):
        brooklyn_unemp = self._load_unemployment_labels(
            self.brooklyn_unemp_csv_path, num_steps_per_episode
        )
        queens_unemp = self._load_unemployment_labels(
            self.queens_unemp_csv_path, num_steps_per_episode
        )
        manhattan_unemp = self._load_unemployment_labels(
            self.manhattan_unemp_csv_path, num_steps_per_episode
        )
        bronx_unemp = self._load_unemployment_labels(
            self.bronx_unemp_csv_path, num_steps_per_episode
        )
        staten_island_unemp = self._load_unemployment_labels(
            self.staten_island_unemp_csv_path, num_steps_per_episode
        )
        return (
            brooklyn_unemp,
            queens_unemp,
            manhattan_unemp,
            bronx_unemp,
            staten_island_unemp,
        )

    def run_episode(self, episode, NUM_TEST_STEPS):
        print(f"\nrunning episode {episode}...")
        self.opt.zero_grad()

        calib_values, initial_claims_weight, labor_force_participation_rate_weight = (
            self._get_parameters()
        )
        avg_month_value = calib_values.reshape(-1, 4, 15).mean(dim=1)
        self._set_parameters(
            avg_month_value,
            initial_claims_weight,
            labor_force_participation_rate_weight,
        )
        print(
            f"calib values: {calib_values}, initial_claims_weight: {initial_claims_weight}, labor_force_participation_rate_weight: {labor_force_participation_rate_weight}"
        )

        self.runner.step(self.NUM_STEPS_PER_EPISODE)

        # predicted_month_unemployment_rate = self.runner.state_trajectory[-1][-1]['environment']['U'].squeeze()
        # predicted_month_unemployment_rate_train = predicted_month_unemployment_rate[:-NUM_TEST_STEPS]

        predicted_brooklyn_unemp = self.runner.state_trajectory[-1][-1]["environment"][
            "Unemployment_Rate_Brooklyn"
        ].squeeze()
        predicted_bronx_unemp = self.runner.state_trajectory[-1][-1]["environment"][
            "Unemployment_Rate_Bronx"
        ].squeeze()
        predicted_queens_unemp = self.runner.state_trajectory[-1][-1]["environment"][
            "Unemployment_Rate_Queens"
        ].squeeze()
        predicted_manhattan_unemp = self.runner.state_trajectory[-1][-1]["environment"][
            "Unemployment_Rate_Manhattan"
        ].squeeze()
        predicted_staten_island_unemp = self.runner.state_trajectory[-1][-1][
            "environment"
        ]["Unemployment_Rate_Staten_Island"].squeeze()

        predicted_brooklyn_unemp_train = predicted_brooklyn_unemp[:-NUM_TEST_STEPS]
        predicted_queens_unemp_train = predicted_queens_unemp[:-NUM_TEST_STEPS]
        predicted_manhattan_unemp_train = predicted_manhattan_unemp[:-NUM_TEST_STEPS]
        predicted_bronx_unemp_train = predicted_bronx_unemp[:-NUM_TEST_STEPS]
        predicted_staten_island_unemp_train = predicted_staten_island_unemp[
            :-NUM_TEST_STEPS
        ]
        predicted_month_unemployment_rate_train = torch.stack(
            [
                predicted_brooklyn_unemp_train,
                predicted_queens_unemp_train,
                predicted_manhattan_unemp_train,
                predicted_bronx_unemp_train,
                predicted_staten_island_unemp_train,
            ],
            dim=1,
        )

        (
            brooklyn_unemp,
            queens_unemp,
            manhattan_unemp,
            bronx_unemp,
            staten_island_unemp,
        ) = self.get_unemployment_labels(self.NUM_STEPS_PER_EPISODE)
        brooklyn_unemp_train = brooklyn_unemp[:-NUM_TEST_STEPS]
        queens_unemp_train = queens_unemp[:-NUM_TEST_STEPS]
        manhattan_unemp_train = manhattan_unemp[:-NUM_TEST_STEPS]
        bronx_unemp_train = bronx_unemp[:-NUM_TEST_STEPS]
        staten_island_unemp_train = staten_island_unemp[:-NUM_TEST_STEPS]
        # target_month_unemployment_rate_train = target_month_unemployment_rate[:-NUM_TEST_STEPS]
        target_month_unemployment_rate_train = torch.stack(
            [
                brooklyn_unemp_train,
                queens_unemp_train,
                manhattan_unemp_train,
                bronx_unemp_train,
                staten_island_unemp_train,
            ],
            dim=1,
        )

        loss_val = self.loss_function(
            predicted_month_unemployment_rate_train,
            target_month_unemployment_rate_train,
        )  # + self.loss_fn_bound_calib_values
        loss_val.backward()
        self.opt.step()
        wandb_log("loss", loss_val.item())

        predicted_brooklyn_unemp_test = predicted_brooklyn_unemp[-NUM_TEST_STEPS:]
        predicted_queens_unemp_test = predicted_queens_unemp[-NUM_TEST_STEPS:]
        predicted_manhattan_unemp_test = predicted_manhattan_unemp[-NUM_TEST_STEPS:]
        predicted_bronx_unemp_test = predicted_bronx_unemp[-NUM_TEST_STEPS:]
        predicted_staten_island_unemp_test = predicted_staten_island_unemp[
            -NUM_TEST_STEPS:
        ]
        predicted_unemp_test = torch.stack(
            [
                predicted_brooklyn_unemp_test,
                predicted_queens_unemp_test,
                predicted_manhattan_unemp_test,
                predicted_bronx_unemp_test,
                predicted_staten_island_unemp_test,
            ],
            dim=1,
        )

        brooklyn_unemp_test = brooklyn_unemp[-NUM_TEST_STEPS:]
        queens_unemp_test = queens_unemp[-NUM_TEST_STEPS:]
        manhattan_unemp_test = manhattan_unemp[-NUM_TEST_STEPS:]
        bronx_unemp_test = bronx_unemp[-NUM_TEST_STEPS:]
        staten_island_unemp_test = staten_island_unemp[-NUM_TEST_STEPS:]
        target_month_unemployment_rate_test = torch.stack(
            [
                brooklyn_unemp_test,
                queens_unemp_test,
                manhattan_unemp_test,
                bronx_unemp_test,
                staten_island_unemp_test,
            ],
            dim=1,
        )

        labor_force_data = self.runner.state_trajectory[-1][-1]["environment"][
            "labor_force"
        ].squeeze()
        labor_force_data = labor_force_data.detach().numpy()
        plot_labor_force(labor_force_data)

        plot_predictions(
            predicted_month_unemployment_rate_train.detach().numpy(),
            target_month_unemployment_rate_train.detach().numpy(),
            "Training" + str(self.episode),
        )
        print(
            f"predicted rate: {predicted_month_unemployment_rate_train}, actual rate: {target_month_unemployment_rate_train}, loss: {loss_val.item()}"
        )
        self.log_file.write(
            f"predicted rate: {predicted_month_unemployment_rate_train}, actual rate: {target_month_unemployment_rate_train}, loss: {loss_val.item()}"
        )

        self.export_and_save_parameters(self.runner.state_trajectory)
        return (
            predicted_unemp_test,
            target_month_unemployment_rate_test,
            loss_val.item(),
        )
        # return predicted_month_unemployment_rate, target_month_unemployment_rate, loss_val.item()

    def test_episode(
        self,
        predicted_month_unemployment_rate,
        target_month_unemployment_rate,
        NUM_TEST_STEPS,
    ):
        with torch.no_grad():
            predicted_month_unemployment_rate_test = predicted_month_unemployment_rate[
                -NUM_TEST_STEPS:
            ]
            target_month_unemployment_rate_test = target_month_unemployment_rate[
                -NUM_TEST_STEPS:
            ]
            loss_test = self.loss_function(
                predicted_month_unemployment_rate_test,
                target_month_unemployment_rate_test,
            )
            print(
                f"predicted rate test: {predicted_month_unemployment_rate_test}, actual rate test: {target_month_unemployment_rate_test}, loss_test: {loss_test.item()}"
            )
            wandb_log("loss_test", loss_test.item())
            plot_predictions(
                predicted_month_unemployment_rate_test.detach().numpy(),
                target_month_unemployment_rate_test.detach().numpy(),
                "Testing" + str(self.episode),
            )
            self.log_file.write(
                f"predicted rate test: {predicted_month_unemployment_rate_test}, actual rate test: {target_month_unemployment_rate_test}, loss_test: {loss_test.item()}"
            )
            return loss_test.item()

    def get_state_trace(self):
        current_episode_state_trace = {
            "environment": {
                id: state_traj["environment"]
                for id, state_traj in enumerate(
                    self.runner.state_trajectory[-1][:: self.NUM_STEPS_PER_EPISODE]
                )
            },
            "agents": {
                id: state_traj["agents"]
                for id, state_traj in enumerate(
                    self.runner.state_trajectory[-1][:: self.NUM_STEPS_PER_EPISODE]
                )
            },
        }
        return current_episode_state_trace

    def save_state_trace(self, state_trace_dict):
        save_path = os.path.join(self.population_dir, "state_data_dict.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(state_trace_dict, f)

    def export_and_save_parameters(self, state_trajectory):
        labor_force_data = self.runner.state_trajectory[-1][-1]["environment"][
            "labor_force"
        ].squeeze()
        labor_force_data = labor_force_data.detach().numpy()
        save_path = os.path.join(self.population_dir, "labor_force.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(labor_force_data, f)

    def run(self):
        state_trace_dict = {}
        loss_val_list = []
        loss_test_list = []
        min_loss = 1e9
        self.log_file = open("LLM_PEER_LOG.txt", "a")

        for episode in range(self.num_episodes):
            self.episode = episode
            if episode == 1:
                print("Loading learned model")
            self.runner.reset()
            (
                predicted_month_unemployment_rate,
                target_month_unemployment_rate,
                loss_val,
            ) = self.run_episode(episode, self.NUM_TEST_STEPS)
            loss_test = self.test_episode(
                predicted_month_unemployment_rate,
                target_month_unemployment_rate,
                self.NUM_TEST_STEPS,
            )
            if loss_val < min_loss:
                min_loss = loss_val
                self.save_learned_model()
            state_trace_dict[episode] = self.get_state_trace()
            loss_val_list.append(loss_val)
            loss_test_list.append(loss_test)

            # Write the values to the log file
            self.log_file.write(
                f"Episode: {episode}, Loss Val: {loss_val}, Loss Test: {loss_test}\n"
            )

        # Close the log file
        self.log_file.close()

        torch.cuda.empty_cache()
        self.save_state_trace(state_trace_dict)

        print("Simulation Ended.")

    def run_llm_with_sim_feedback(self):
        state_trace_dict = {}
        # Open the log file in append mode
        self.log_file = open("LLM_PEER_LOG.txt", "a")
        save_path = os.path.join(self.population_dir, "labor_force.pkl")
        for episode in range(self.num_episodes):
            self.episode = episode
            self.runner.reset()
            print(f"\nrunning episode {episode}...")

            (
                calib_values,
                initial_claims_weight,
                labor_force_participation_rate_weight,
            ) = self._get_parameters()
            avg_month_value = calib_values.reshape(-1, 4, 1)
            self._set_parameters(
                avg_month_value,
                initial_claims_weight,
                labor_force_participation_rate_weight,
            )
            print(
                f"calib values: {calib_values}, initial_claims_weight: {initial_claims_weight}, labor_force_participation_rate_weight: {labor_force_participation_rate_weight}"
            )

            self.runner.step(self.NUM_STEPS_PER_EPISODE)
            labor_force_data = self.runner.state_trajectory[-1][-1]["environment"][
                "labor_force"
            ].squeeze()
            plot_labor_force(labor_force_data)
            with open(save_path, "wb") as f:
                pickle.dump(labor_force_data, f)

        # Close the log file
        self.log_file.close()
        torch.cuda.empty_cache()

        print("Simulation Ended.")


if __name__ == "__main__":
    config_file = "/Users/shashankkumar/Documents/GitHub/MacroEcon/models/macro_economics/yamls/config_nyc_100_agents.yaml"
    # config_file = '/Users/shashankkumar/Documents/GitHub/MacroEcon/models/macro_economics/yamls/config.yaml'
    # config_file = '/Users/shashankkumar/Documents/GitHub/MacroEcon/models/macro_economics/yamls/config_nyc_all.yaml' # NYC complete population
    config = read_config(config_file)
    registry = simulation_registry()
    runner = SimulationRunner(config=config, registry=registry)
    runner.init()
    calibrator = Calibrator(runner)
    calibrator.run()
    # calibrator.run_llm_with_sim_feedback()
