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
        initial_claims_path = self.config["simulation_metadata"]["initial_claims_path"]
        initial_claims_df = pd.read_pickle(initial_claims_path)
        self.initial_claims = torch.tensor(
            initial_claims_df["beneficiaries"].values
        )  # load initial claims data for each county separately. Current is kings county
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

        county = get_by_path(state, re.split("/", self.input_variables["region"]))
        labor_force = get_by_path(
            state, re.split("/", self.input_variables["labor_force"])
        )
        unemployment_adaptation_coefficient_all = torch.matmul(
            time_step_one_hot.float().unsqueeze(dim=0), self.external_UAC
        ).squeeze([0, 1])
        current_month_initial_claims = self.initial_claims[t]

        total_labor_force = torch.sum(
            working_status
        )  # total number of agents willing to work
        labor_force_participation_rate = total_labor_force / self.num_agents

        ## EQ 2
        current_total_unemployment_rate = (
            unemployment_adaptation_coefficient_all[0] * torch.log(total_labor_force)
            + unemployment_adaptation_coefficient_all[1]
            * torch.log(current_month_initial_claims)
            + unemployment_adaptation_coefficient_all[2]
        )  # + self.labor_force_participation_rate_weight*(labor_force_participation_rate)
        # current_total_unemployment_rate = unemployment_adaptation_coefficient_all[0]*torch.log(total_labor_force) + self.initial_claims_weight*torch.log(current_month_initial_claims) #+ self.labor_force_participation_rate_weight*(labor_force_participation_rate)
        ## EQ 3
        # current_total_unemployment_rate = unemployment_adaptation_coefficient_all[0]*torch.log(labor_force_participation_rate) + self.initial_claims_weight*torch.log(current_month_initial_claims)

        ## Update unemployment rate for this month
        unemployment_rate = unemployment_rate + (
            current_total_unemployment_rate * time_step_one_hot
        )

        # update labor force data -
        labor_force = labor_force + (total_labor_force * time_step_one_hot)

        # hourly wages
        new_hourly_wages = self.updateHourlyWage(
            hourly_wage, imbalance
        )  # self.calculateHourlyWage() to revert

        return {
            self.output_variables[0]: new_hourly_wages,
            self.output_variables[1]: unemployment_rate,
            self.output_variables[2]: labor_force,
        }
