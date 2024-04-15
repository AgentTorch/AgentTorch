AGENT_TORCH_PATH = '/u/ayushc/projects/GradABM/MacroEcon/AgentTorch'
MODEL_PATH = '/u/ayushc/projects/GradABM/MacroEcon/models'

import torch
import numpy as np
import json
import sys
import re
import pdb
import time

sys.path.append(MODEL_PATH)
sys.path.insert(0, AGENT_TORCH_PATH)

from AgentTorch.helpers import get_by_path
from AgentTorch.substep import SubstepAction
from AgentTorch.LLM.llm_agent import LLMAgent
from prompt import prompt_template_var, system_prompt

from llm_utils import get_answer, CaseProvider, Neighborhood

class MakeIsolationDecision(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        OPENAI_API_KEY = self.config['simulation_metadata']['OPENAI_API_KEY']
        self.device = self.config['simulation_metadata']['device']
        self.mode = self.config['simulation_metadata']['EXECUTION_MODE']

        self.agent = LLMAgent(agent_profile = system_prompt, openai_api_key=OPENAI_API_KEY)

        self.provider = CaseProvider()

        # index to age string mapping - for prompt formatting
        self.age_mapping = {0: "under 19 years old", 1: "between 20-29 years old", 
                            2: "between 30-39 years old",
                            3: "between 40-49 years old",
                            4: "between 50-64 years old",
                            5: "above 65 years old"}
        
        self.dates = self.provider.get_dates()
        self.case_numbers = self.provider.get_case_numbers()
        
        self.num_agents = self.config['simulation_metadata']['num_agents']
    
    def string_to_number(self, string):
        if 'false' in string.lower():
            return 0
        else:
            return 1
    
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
        input_variables = self.input_variables

        week_id = int(t/7) + 1
                
        past_week_num = self.case_numbers[week_id-1]
        curr_week_num = self.case_numbers[week_id]
        week_i_change = (curr_week_num/past_week_num - 1)*100
        
        agent_age = get_by_path(state, re.split("/", input_variables['age']))

        if self.mode == 'debug':
            will_isolate = torch.rand(self.num_agents, 1)
            return {self.output_variables[0]: will_isolate}

        masks = []
        prompt_list = []
                
        # prompts are segregated based on agent age
        for value in self.age_mapping.keys():
            # agent subgroups for each prompt
            age_mask = (agent_age == value)
            masks.append(age_mask.float())
        
        for value in self.age_mapping.keys():
            # formatting prompt for each group
            prompt = prompt_template_var.format(age=self.age_mapping[value], week_i_num=curr_week_num, change_text=self.change_text(week_i_change))
            prompt_list.append(prompt)

        # execute prompts from LLMAgent and compile response
        time.sleep(1)
        agent_output = await self.agent(prompt_list)

        # assign prompt response to agents
        will_isolate = torch.zeros((self.num_agents, 1))

        for en, output_value in enumerate(agent_output):
            output_response = output_value['text']
            decision = output_response.split('.')[0]
            reasoning = output_response.split('.')[1] # reasoning to be saved for RAG later
            isolation_response = self.string_to_number(decision)
            will_isolate = will_isolate + masks[en]*isolation_response
        
        return {self.output_variables[0]: will_isolate}