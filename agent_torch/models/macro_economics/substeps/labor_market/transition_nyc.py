import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal
import sys

sys.path.append("/Users/shashankkumar/Documents/GitHub/MacroEcon/AgentTorch")
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
from torch.nn import functional as F
import re


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


class UpdateMacroRates(SubstepTransition):
    """Macro quantities relevant to labor markets - hourly wage, unemployment rate, price of goods"""

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.num_timesteps = self.config["simulation_metadata"]["num_steps_per_episode"]
        self.max_rate_change = self.config["simulation_metadata"][
            "maximum_rate_of_change_of_wage"
        ]
        self.num_agents = self.config["simulation_metadata"]["num_agents"]
        self.external_UAC = torch.tensor(
            self.learnable_args["unemployment_adaptation_coefficient"],
            requires_grad=True,
        )
        self.initial_claims_weight = torch.tensor(
            self.learnable_args["initial_claims_weight"], requires_grad=True
        )
        self.labor_force_participation_rate_weight = torch.tensor(
            self.learnable_args["labor_force_participation_rate_weight"],
            requires_grad=True,
        )

        brooklyn_claims_path = self.config["simulation_metadata"][
            "brooklyn_claims_path"
        ]
        bronx_claims_path = self.config["simulation_metadata"]["bronx_claims_path"]
        manhattan_claims_path = self.config["simulation_metadata"][
            "manhattan_claims_path"
        ]
        queens_claims_path = self.config["simulation_metadata"]["queens_claims_path"]
        staten_island_claims_path = self.config["simulation_metadata"][
            "staten_island_claims_path"
        ]

        brooklyn_claims_df = pd.read_csv(brooklyn_claims_path)
        bronx_claims_df = pd.read_csv(bronx_claims_path)
        manhattan_claims_df = pd.read_csv(manhattan_claims_path)
        queens_claims_df = pd.read_csv(queens_claims_path)
        staten_island_claims_df = pd.read_csv(staten_island_claims_path)

        self.brooklyn_claims = torch.tensor(brooklyn_claims_df["beneficiaries"].values)
        self.bronx_claims = torch.tensor(bronx_claims_df["beneficiaries"].values)
        self.manhattan_claims = torch.tensor(
            manhattan_claims_df["beneficiaries"].values
        )
        self.queens_claims = torch.tensor(queens_claims_df["beneficiaries"].values)
        self.staten_island_claims = torch.tensor(
            staten_island_claims_df["beneficiaries"].values
        )

        self.model = LinearRegressionModel(2, 4, 1)

    def calculateNumberOfAgentsNotWorking(self, working_status):
        agents_not_working = torch.sum((1 - working_status))
        return agents_not_working

    def calculateNumberOfAgentsNotWorkingForCounty(self, working_status, county_mask):
        total_population = torch.sum(county_mask)
        agents_working = torch.sum(working_status)
        agents_not_working = total_population - agents_working

        return agents_not_working

    def updateHourlyWage(self, hourly_wage, imbalance):
        omega = imbalance.float()
        r1, r2 = self.max_rate_change * omega, 0

        sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2

        new_hourly_wage = hourly_wage + hourly_wage * sampled_omega
        return new_hourly_wage

    def legacy_calculateHourlyWage(self, hourly_wage, imbalance):
        # Calculate hourly wage
        w = hourly_wage
        omega = imbalance.float()

        if omega > 0:
            max_rate_change = self.config["simulation_metadata"][
                "maximum_rate_of_change_of_wage"
            ]
            r2 = max_rate_change * omega
            r1 = 0
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            new_hourly_wage = w * (1 + sampled_omega)
        else:
            max_rate_change = self.config["simulation_metadata"][
                "maximum_rate_of_change_of_wage"
            ]
            r1 = max_rate_change * omega
            r2 = 0
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            # sampled_omega = torch.tensor.uniform((max_rate_change * omega),0)
            new_hourly_wage = w * (1 + sampled_omega)
        return new_hourly_wage

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        timestep_tensor = torch.tensor([timestep])
        one_hot_tensor = F.one_hot(timestep_tensor, num_classes=num_timesteps)

        return one_hot_tensor.to(self.device)

    def create_county_masks(self, county):
        county_masks = []
        for i in range(5):
            mask = torch.tensor([True] * self.num_agents)
            mask = torch.logical_and(mask, county == i)
            mask = mask.unsqueeze(1)
            county_masks.append(mask)
        return county_masks

    def calculateUnemploymentRate(
        self, unemployment_adaptation_coefficient, labor_force, initial_claims
    ):
        # unemployment_adaptation_coefficient = unemployment_adaptation_coefficient.reshape(
        current_total_unemployment_rate = (
            unemployment_adaptation_coefficient[0] * torch.log(labor_force)
            + unemployment_adaptation_coefficient[1] * torch.log(initial_claims)
            + unemployment_adaptation_coefficient[2]
        )  # + self.labor_force_participation_rate_weight*(labor_force_participation_rate)
        return current_total_unemployment_rate

    def calculateUnemploymentRateUsingNN(self, labor_force, initial_claims):
        output = self.model(
            torch.tensor([torch.log(labor_force), torch.log(initial_claims)])
        )

        return output

    def forward(self, state, action):
        print("Executing Substep: Labor Market")
        month_id = state["current_step"]
        t = int(month_id)
        time_step_one_hot = self._generate_one_hot_tensor(t, self.num_timesteps)
        working_status = get_by_path(
            state, re.split("/", self.input_variables["will_work"])
        )
        imbalance = get_by_path(state, re.split("/", self.input_variables["imbalance"]))
        hourly_wage = get_by_path(
            state, re.split("/", self.input_variables["hourly_wage"])
        )

        unemployment_rate = get_by_path(
            state, re.split("/", self.input_variables["unemployment_rate"])
        )
        unemployment_rate_bronx = get_by_path(
            state, re.split("/", self.input_variables["unemployment_rate_bronx"])
        )
        unemployment_rate_brooklyn = get_by_path(
            state, re.split("/", self.input_variables["unemployment_rate_brooklyn"])
        )
        unemployment_rate_manhattan = get_by_path(
            state, re.split("/", self.input_variables["unemployment_rate_manhattan"])
        )
        unemployment_rate_queens = get_by_path(
            state, re.split("/", self.input_variables["unemployment_rate_queens"])
        )
        unemployment_rate_staten_island = get_by_path(
            state,
            re.split("/", self.input_variables["unemployment_rate_staten_island"]),
        )

        county = get_by_path(state, re.split("/", self.input_variables["region"]))
        labor_force = get_by_path(
            state, re.split("/", self.input_variables["labor_force"])
        )

        total_labor_force = torch.sum(working_status)
        unemployment_adaptation_coefficient_all = torch.matmul(
            time_step_one_hot.float().unsqueeze(dim=0), self.external_UAC
        ).squeeze([0, 1])
        unemployment_adaptation_coefficient_all = (
            unemployment_adaptation_coefficient_all.reshape(5, 3)
        )
        county_masks = self.create_county_masks(county)

        population_county_bronx = torch.sum(county_masks[0])
        population_county_brooklyn = torch.sum(county_masks[1])
        population_county_manhattan = torch.sum(county_masks[2])
        population_county_queens = torch.sum(county_masks[3])
        population_county_staten_island = torch.sum(county_masks[4])

        working_status_county_bronx = working_status * county_masks[0]
        working_status_county_brooklyn = working_status * county_masks[1]
        working_status_county_manhattan = working_status * county_masks[2]
        working_status_county_queens = working_status * county_masks[3]
        working_status_county_staten_island = working_status * county_masks[4]

        labor_force_bronx = torch.clamp(torch.sum(working_status_county_bronx), min=1)
        labor_force_brooklyn = torch.clamp(
            torch.sum(working_status_county_brooklyn), min=1
        )
        labor_force_manhattan = torch.clamp(
            torch.sum(working_status_county_manhattan), min=1
        )
        labor_force_queens = torch.clamp(torch.sum(working_status_county_queens), min=1)
        labor_force_staten_island = torch.clamp(
            torch.sum(working_status_county_staten_island), min=1
        )

        unemployment_adaptation_coefficient_bronx = (
            unemployment_adaptation_coefficient_all[0]
        )
        unemployment_adaptation_coefficient_brooklyn = (
            unemployment_adaptation_coefficient_all[1]
        )
        unemployment_adaptation_coefficient_manhattan = (
            unemployment_adaptation_coefficient_all[2]
        )
        unemployment_adaptation_coefficient_queens = (
            unemployment_adaptation_coefficient_all[3]
        )
        unemployment_adaptation_coefficient_staten_island = (
            unemployment_adaptation_coefficient_all[4]
        )

        current_month_initial_claims_bronx = self.bronx_claims[t]
        current_month_initial_claims_brooklyn = self.brooklyn_claims[t]
        current_month_initial_claims_manhattan = self.manhattan_claims[t]
        current_month_initial_claims_queens = self.queens_claims[t]
        current_month_initial_claims_staten_island = self.staten_island_claims[t]

        current_unemployment_rate_county_bronx = self.calculateUnemploymentRate(
            unemployment_adaptation_coefficient_bronx,
            labor_force_bronx,
            current_month_initial_claims_bronx,
        )
        current_unemployment_rate_county_brooklyn = self.calculateUnemploymentRate(
            unemployment_adaptation_coefficient_brooklyn,
            labor_force_brooklyn,
            current_month_initial_claims_brooklyn,
        )
        current_unemployment_rate_county_manhattan = self.calculateUnemploymentRate(
            unemployment_adaptation_coefficient_manhattan,
            labor_force_manhattan,
            current_month_initial_claims_manhattan,
        )
        current_unemployment_rate_county_queens = self.calculateUnemploymentRate(
            unemployment_adaptation_coefficient_queens,
            labor_force_queens,
            current_month_initial_claims_queens,
        )
        current_unemployment_rate_county_staten_island = self.calculateUnemploymentRate(
            unemployment_adaptation_coefficient_staten_island,
            labor_force_staten_island,
            current_month_initial_claims_staten_island,
        )
        current_nyc_unemployment_rate = (
            current_unemployment_rate_county_bronx
            + current_unemployment_rate_county_brooklyn
            + current_unemployment_rate_county_manhattan
            + current_unemployment_rate_county_queens
            + current_unemployment_rate_county_staten_island
        ) / 5

        unemployment_rate_bronx = unemployment_rate_bronx + (
            current_unemployment_rate_county_bronx * time_step_one_hot
        )
        unemployment_rate_brooklyn = unemployment_rate_brooklyn + (
            current_unemployment_rate_county_brooklyn * time_step_one_hot
        )
        unemployment_rate_manhattan = unemployment_rate_manhattan + (
            current_unemployment_rate_county_manhattan * time_step_one_hot
        )
        unemployment_rate_queens = unemployment_rate_queens + (
            current_unemployment_rate_county_queens * time_step_one_hot
        )
        unemployment_rate_staten_island = unemployment_rate_staten_island + (
            current_unemployment_rate_county_staten_island * time_step_one_hot
        )
        unemployment_rate = unemployment_rate + (
            current_nyc_unemployment_rate * time_step_one_hot
        )

        # update labor force data
        labor_force = labor_force + (total_labor_force * time_step_one_hot)

        # hourly wages
        new_hourly_wages = self.updateHourlyWage(
            hourly_wage, imbalance
        )  # self.calculateHourlyWage() to revert

        return {
            self.output_variables[0]: new_hourly_wages,
            self.output_variables[1]: unemployment_rate_bronx,
            self.output_variables[2]: unemployment_rate_brooklyn,
            self.output_variables[3]: unemployment_rate_manhattan,
            self.output_variables[4]: unemployment_rate_queens,
            self.output_variables[5]: unemployment_rate_staten_island,
            self.output_variables[6]: unemployment_rate,
            self.output_variables[7]: labor_force,
        }
