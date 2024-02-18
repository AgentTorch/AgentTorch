import torch
import torch.nn as nn
import numpy as np
import re
from AgentTorch.llm_agent import LLMAgent
from AgentTorch.substep import SubstepAction

class CalculateWorkPropensity(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        self.agent = LLMAgent(config)
        
    def forward(self, state, observation):
        prompt = self.config['simulation_metadata']['work_prompt']
        #TODO format prompt with current state properties
        #TODO agent.format_prompt_with_state_properties(state)
        # work_propensity = self.agent(input = prompt)
        work_propensity = torch.rand(6400,1)
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
        consumption_propensity = torch.rand(6400,1)
        return {self.output_variables[0] : consumption_propensity}
