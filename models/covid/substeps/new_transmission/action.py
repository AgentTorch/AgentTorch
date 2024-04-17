# AGENT_TORCH_PATH = '/u/ayushc/projects/GradABM/MacroEcon/AgentTorch'
# MODEL_PATH = '/u/ayushc/projects/GradABM/MacroEcon/models'

import torch
import numpy as np
import re
import time

# sys.path.append(MODEL_PATH)
# sys.path.insert(0, AGENT_TORCH_PATH)

from AgentTorch.helpers import get_by_path
from AgentTorch.substep import SubstepAction
from AgentTorch.LLM.llm_agent import LLMAgent

from utils.data import get_data
from utils.feature import Feature
from utils.llm import AgeGroup, SYSTEM_PROMPT, construct_user_prompt
from utils.misc import week_num_to_epiweek, name_to_neighborhood

class MakeIsolationDecision(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set values from config
        OPENAI_API_KEY = self.config['simulation_metadata']['OPENAI_API_KEY']
        self.device = torch.device(self.config['simulation_metadata']['device'])
        self.mode = self.config['simulation_metadata']['EXECUTION_MODE']
        self.num_agents = self.config['simulation_metadata']['num_agents']
        self.epiweek_start = week_num_to_epiweek(
            self.config["simulation_metadata"]["START_WEEK"]
        )
        self.num_weeks = self.config["simulation_metadata"]["NUM_WEEKS"]
        self.neighborhood = name_to_neighborhood(
            self.config["simulation_metadata"]["NEIGHBORHOOD"]
        )
        self.include_week_count = self.config["simulation_metadata"]["INCLUDE_WEEK_COUNT"]

        # set up llm agent
        self.agent = LLMAgent(agent_profile=SYSTEM_PROMPT, openai_api_key=OPENAI_API_KEY)

        # retrieve data
        data = get_data(
            self.neighborhood,
            self.epiweek_start,
            self.num_weeks,
            [Feature.CASES, Feature.CASES_4WK_AVG],
        )
        self.cases_week = data[:, 0]
        self.cases_4_week_avg = data[:, 1]

    def string_to_number(self, string):
        if string.lower() == "yes":
            return 1
        else:
            return 0

    def change_text(self, change_amount):
        change_amount = int(change_amount)
        if change_amount >= 1:
            return f"a {change_amount}% increase from last week"
        elif change_amount <= -1:
            return f"a {abs(change_amount)}% decrease from last week"
        else:
            return "the same as last week"

    async def forward(self, state, observation):
        '''
            LLMAgent class has three functions: a) mask sub-groups, b) format_prompt, c) invoke LLM, d) aggregate response 
        '''
        t = int(state['current_step'])
        week_index = t//7
        input_variables = self.input_variables


        agent_age = get_by_path(state, re.split("/", input_variables['age']))

        if self.mode == 'debug':
            will_isolate = torch.rand(self.num_agents, 1)
            return {self.output_variables[0]: will_isolate}

        # prompts are segregated based on agent age
        masks = []
        for age_group in AgeGroup:
            # agent subgroups for each prompt
            age_mask = (agent_age == age_group.value)
            masks.append(age_mask.float())

        # generate the prompt list
        prompt_list = [
            {
                "age": age_group.text,
                "location": self.neighborhood.text,
                "user_prompt": construct_user_prompt(
                    self.include_week_count,
                    self.epiweek_start,
                    week_index,
                    self.cases_week[week_index],
                    self.cases_4_week_avg[week_index],
                ),
            }
            for age_group in AgeGroup
        ]

        # time.sleep(1)
        # execute prompts from LLMAgent and compile response
        agent_output = await self.agent(prompt_list)

        # assign prompt response to agents
        will_isolate = torch.zeros((self.num_agents, 1)).to(self.device)

        for en, output_value in enumerate(agent_output):
            output_response = output_value['text']
            decision = output_response.split('.')[0]
            reasoning = output_response.split('.')[1] # reasoning to be saved for RAG later
            isolation_response = self.string_to_number(decision)
            will_isolate = will_isolate + masks[en]*isolation_response
        
        return {self.output_variables[0]: will_isolate}
