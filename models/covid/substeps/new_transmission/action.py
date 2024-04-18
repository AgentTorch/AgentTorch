# AGENT_TORCH_PATH = '/u/ayushc/projects/GradABM/MacroEcon/AgentTorch'
# MODEL_PATH = '/u/ayushc/projects/GradABM/MacroEcon/models'

import torch
import numpy as np
import re
import time
import os

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
        # if in debug mode, return random values for isolation
        if self.mode == 'debug':
            will_isolate = torch.rand(self.num_agents, 1).to(self.device)
            return {self.output_variables[0]: will_isolate}

        # figure out time step
        time_step = int(state['current_step'])

        # if beginning of the week, recalculate isolation probabilities
        if time_step % 7 == 0:
            # figure out week index
            week_index = time_step//7
            print(f"sampling isolation probabilities for week {week_index}")

            # prompts are segregated based on agent age
            masks = []
            agent_age = get_by_path(state, re.split("/", self.input_variables["age"]))
            for age_group in AgeGroup:
                # agent subgroups for each prompt
                age_mask = agent_age == age_group.value
                masks.append(age_mask.float())

            # generate the prompt list, prompt 7 times for each age group for each week to get
            # probabilities
            prompt_list = [
                {
                    "age": age_group.text,
                    "location": self.neighborhood.text,
                    "user_prompt": construct_user_prompt(
                        self.include_week_count,
                        self.epiweek_start,
                        week_index,
                        # this is a bug. case data is in float format in csv. should get it in int
                        # to avoid further confusion.
                        int(self.cases_week[week_index]),
                        int(self.cases_4_week_avg[week_index]),
                    ),
                }
                for age_group in AgeGroup
            ] * 7

            # use this as additional context to the prompt
            episode_history_file = "/tmp/history_predicted_weekly_cases.npy"
            if os.path.exists(episode_history_file):
                cases_past_episode = np.load(episode_history_file)

            # time.sleep(1)
            # execute prompts from LLMAgent and compile response
            agent_output = await self.agent(prompt_list)

            # this is a bug. this probably belongs in the init function, but I don't know if this
            # class gets initialized at each episode. if not, this would cause the week 0 running
            # average to include the last episode's last week.
            if time_step == 0:
                self.isolation_probabilities = (
                    torch.ones((self.num_agents, 1)).to(self.device) * 1 / 2
                )

            # assign prompt response to agents
            self.last_isolation_probabilities = self.isolation_probabilities
            self.isolation_probabilities = torch.zeros((self.num_agents, 1)).to(self.device)
            for en, output_value in enumerate(agent_output):
                output_response = output_value["text"]
                decision = output_response.split(".")[0]
                # reasoning to be saved for RAG later
                reasoning = output_response.split(".")[1]
                isolation_response = self.string_to_number(decision)
                # this is a bug. it is a bit hacky, currently prompt_list[0], prompt_list[6], 12
                # etc. belong to the age_group 0, while 1, 7 etc. belong to the age_group 1, and
                # there are 7 responses for each group so we add each one of them with 1/7 weight
                self.isolation_probabilities += masks[en % len(AgeGroup)] * isolation_response * 1/7

        # sample isolation decision from probabilities
        will_isolate = torch.bernoulli(
            self.last_isolation_probabilities * 1 / 2
            + self.isolation_probabilities * 1 / 2
        )
        print(f"day {time_step}, number of isolating agents {sum(will_isolate).item()}")
        return {self.output_variables[0]: will_isolate}
