import asyncio
import json
import os
import pandas as pd
import torch
import re
from AgentTorch.LLM.llm_agent import LLMAgent
from AgentTorch.helpers.distributions import StraightThroughBernoulli

from AgentTorch.LLM.llm_agent import LLMAgent, BasicQAEcon, COT
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import get_by_path

from macro_economics.prompt import agent_profile
import itertools
import time
from AgentTorch.LLM.refactor import LLMInitializer, LLMArchetype, BasicQAEcon, COT
from AgentTorch.LLM.behavior import Behavior
from populations import NYC


class WorkConsumptionPropensity(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        OPENAI_API_KEY = self.config["simulation_metadata"]["OPENAI_API_KEY"]
        self.month_mapping = self.config["simulation_metadata"]["month_mapping"]
        self.year_mapping = self.config["simulation_metadata"]["year_mapping"]
        self.mode = self.config["simulation_metadata"]["execution_mode"]
        self.save_memory_dir = self.config["simulation_metadata"]["memory_dir"]
        self.num_steps_per_episode = self.config["simulation_metadata"][
            "num_steps_per_episode"
        ]
        self.num_agents = self.config["simulation_metadata"]["num_agents"]
        self.covid_cases_path = self.config["simulation_metadata"]["covid_cases_path"]
        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.prompt = self.config["simulation_metadata"]["EARNING_ACTION_PROMPT"]
        self.expt_mode = self.config["simulation_metadata"]["expt_mode"]
        self.st_bernoulli = StraightThroughBernoulli.apply
        self.covid_cases = pd.read_csv(self.covid_cases_path)
        self.covid_cases = torch.tensor(self.covid_cases.values)  # add to device

        llm = LLMInitializer(
            backend="dspy", qa=BasicQAEcon, cot=COT, openai_api_key=OPENAI_API_KEY
        )
        llm.initialize_llm()

        archetype = LLMArchetype(llm=llm, user_prompt=self.prompt, num_agents=12)
        self.earning_behavior = Behavior(archetype=archetype, region=NYC)

    def forward(self, state, observation):
        print("Substep Action: Earning decision")

        number_of_months = (
            state["current_step"] + 1 + 7
        )  # 1 indexed, also since we are starting from 8th month add 7 here
        current_year = (
            number_of_months // 12 + 1 + 1
        )  # 1 indexed, also since we are starting from 2020 add 1 here
        year = self.year_mapping[current_year]
        month = self.month_mapping[number_of_months % 12]
        covid_cases = self.covid_cases[number_of_months % 12].item()

        kwargs = {
            "month": month,
            "year": year,
            "covid_cases": covid_cases,
            "device": "cpu",
            "current_memory_dir": self.save_memory_dir,
            "price_of_goods": 0.05,
        }

        output = self.earning_behavior.sample(kwargs)

        return {
            self.output_variables[0]: will_work,
            self.output_variables[1]: consumption_propensity,
        }
