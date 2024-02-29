import torch
import torch.nn as nn
import numpy as np
import re
from AgentTorch.LLM.llm_agent import LLMAgent
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import get_by_path
from AgentTorch.models.macro_economics.prompt import prompt_template

class CalculateWorkPropensity(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        self.agent = LLMAgent(config)
        self.agent_list = []
    
    def forward(self, state, observation):
        agent_id = 12
        prompt_template = "You are {male} of age {age} and are {ethnicity}. You live in {area}. Give your willingness to work in {industry}, denote the willingness by giving a value between 0 and 1, with 0 being not willing at all and 1 being completely willing."
        gender_mapping = {1:"male", 2:"female"}
        gender_tensor = get_by_path(state, re.split("/", self.input_variables['Gender']))
        gender = gender_tensor[agent_id].item()
        gender = gender_mapping[gender]
        get_agent_age = "U19"
        get_agent_gender = gender
        get_agent_ethnicity = "White"
        area = "Manhattan"
        industry = "Finance"
        prompt = prompt_template.format(male = get_agent_gender,age = get_agent_age,ethnicity=get_agent_ethnicity,industry=industry,area=area)
        prompt_output = self.agent(prompt)
        
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
