import torch
import torch.nn as nn
import numpy as np
import re
from AgentTorch.AgentTorch.LLM.llm_agent import LLMAgent
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import get_by_path

class CalculateWorkPropensity(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        self.agent = LLMAgent(config)
        self.agent_list = []
    
    def forward(self, state, observation):
        agent_ids = get_by_path(state, re.split("/", self.input_variables['ID']))
        
        prompt = self.config['simulation_metadata']['work_prompt']
        age_mapping = self.config['simulation_metadata']['age_mapping']
        gender_mapping = self.config['simulation_metadata']['gender_mapping']
        ethnicity_mapping = self.config['simulation_metadata']['ethnicity_mapping']
        get_agent_age = "U19"
        get_agent_gender = "Male"
        get_agent_ethnicity = "White"
        prompt = prompt.format(prompt,get_agent_gender,get_agent_age,get_agent_ethnicity)
        agent = self.agent_list[agent_ids[0]]
        prompt_output = agent(prompt)
        
        # work_propensity = self.agent(input = prompt)
        work_propensity = torch.rand(16573530,1)
        whether_to_work = torch.bernoulli(work_propensity)
        return {self.output_variables[0] : whether_to_work, self.output_variables[1] : work_propensity}
        
class CalculateConsumptionPropensity(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        self.agent = LLMAgent(config)
        
    def forward(self, state, observation):
        prompt = self.config['simulation_metadata']['consumption_prompt']
        #TODO format prompt with current state properties
        # consumption_propensity = self.agent(input = prompt)
        consumption_propensity = torch.rand(16573530,1)
        return {self.output_variables[0] : consumption_propensity}
