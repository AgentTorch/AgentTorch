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
class CalculateWorkAndConsumptionPropensity(SubstepAction):
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
            gender_mask = gender == target_values['gender']
            age_mask = age == target_values['age']
            mask = torch.logical_and(gender_mask, age_mask)
            float_mask = mask.float()
            masks.append(float_mask)
        
        for en,_ in enumerate(self.combinations_of_prompt_variables_with_index):
            age = self.combinations_of_prompt_variables[en]['age']
            gender = self.combinations_of_prompt_variables[en]['gender']
            prompt = prompt_template_var.format(age = age,gender = gender)
            output_value = self.agent(prompt)
            output_values.append(output_value)
        
        for en,output_value in enumerate(output_values):
            # output_value = re.search(r'\{(.+?)\}', output_value, re.DOTALL)
            output_value = json.loads(output_value)
            group_work_propensity = output_value['work']
            group_consumption_propensity = output_value['consumption']
            consumption_propensity_for_group = masks[en]*group_consumption_propensity
            consumption_propensity = torch.add(consumption_propensity,consumption_propensity_for_group)
            work_propensity_for_group = masks[en]*group_work_propensity
            work_propensity = torch.add(work_propensity,work_propensity_for_group)

        # work_propensity = torch.rand(16573530,1)
        whether_to_work = torch.bernoulli(work_propensity)
        return {self.output_variables[0] : whether_to_work, self.output_variables[1] : work_propensity, self.output_variables[2] : consumption_propensity}
        
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
        
        
