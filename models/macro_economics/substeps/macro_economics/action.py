import json
import torch
import torch.nn as nn
import numpy as np
import re
import sys
sys.path.append('/Users/shashankkumar/Documents/GitHub/MacroEcon/AgentTorch')
sys.path.append('/Users/shashankkumar/Documents/GitHub/MacroEcon/models')
from AgentTorch.LLM.llm_agent import LLMAgent
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import get_by_path
from macro_economics.prompt import prompt_template_var,agent_profile
import itertools
OPENAI_API_KEY = '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'
class CalculateWorkPropensity(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        self.agent = LLMAgent(agent_profile = agent_profile,openai_api_key = OPENAI_API_KEY )
        self.mapping = self.load_mapping(self.config['simulation_metadata']['mapping_path'])
        self.variables = self.get_variables(prompt_template_var)
        self.filtered_mapping = self.filter_mapping(self.mapping,self.variables)
        self.combinations_of_prompt_variables, self.combinations_of_prompt_variables_with_index = self.get_combinations_of_prompt_variables(self.filtered_mapping)
        
    def augment_mapping_with_index(self,mapping):
        return {k: {v: i for i, v in enumerate(values)} for k, values in mapping.items()}

    def filter_mapping(self,mapping,variables):
        return {k: mapping[k] for k in variables}
        
    def forward(self, state, observation):
        gender = get_by_path(state, re.split("/", self.input_variables['Gender']))
        age = get_by_path(state,re.split("/", self.input_variables['Age']))
        consumption_propensity = get_by_path(state,re.split("/", self.input_variables['Consumption_Propensity']))
        work_propensity = get_by_path(state,re.split("/", self.input_variables['Work_Propensity']))
        masks = []
        output_values = []
        
        for target_values in self.combinations_of_prompt_variables_with_index:
            masks_for_dict = [globals()[key] == value for key, value in target_values.items()]
            mask = torch.all(torch.stack(masks_for_dict), dim=0).float()
            masks.append(mask)
        
        for en,_ in enumerate(self.combinations_of_prompt_variables_with_index.keys()):
            age = self.combinations_of_prompt_variables[en]['age']
            gender = self.combinations_of_prompt_variables[en]['gender']
            prompt = prompt_template_var.format(age = age,gender = gender)
            output_value = self.agent(prompt)
            output_values.append(output_value)
        
        for en,output_value in enumerate(output_values):
            output_value = json.loads(output_value)
            group_work_propensity = output_value['work']
            group_consumption_propensity = output_value['consumption']
            consumption_propensity = consumption_propensity + (masks[en]*group_consumption_propensity)
            work_propensity = work_propensity + (mask[en]*group_work_propensity)

        # work_propensity = torch.rand(16573530,1)
        whether_to_work = torch.bernoulli(work_propensity)
        return {self.output_variables[0] : whether_to_work, self.output_variables[1] : work_propensity}
        
    def get_variables(self,prompt_template):
        variables = re.findall(r'\{(.+?)\}', prompt_template)
        return variables
    
    def load_mapping(self,path):
        with open(path, 'r') as f:
            mapping = json.load(f)
        return mapping
    
    def get_combinations_of_prompt_variables(self,mapping):
        combinations = list(itertools.product(*mapping.values()))
        dict_combinations = [dict(zip(mapping.keys(), combination)) for combination in combinations]
        index_combinations = [{k: mapping[k].index(v) for k, v in combination.items()} for combination in dict_combinations]
        return dict_combinations, index_combinations
        
        
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
