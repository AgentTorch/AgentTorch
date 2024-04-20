# AGENT_TORCH_PATH = '/Users/shashankkumar/Documents/GitHub/MacroEcon/AgentTorch'
# MODEL_PATH = '/Users/shashankkumar/Documents/GitHub/MacroEcon/models'
AGENT_TORCH_PATH = '/u/ayushc/projects/GradABM/MacroEcon/AgentTorch'
MODEL_PATH = '/u/ayushc/projects/GradABM/MacroEcon/models'

# OPENAI_API_KEY = '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'

import json
import os
import torch
import re
import sys
import pdb
from AgentTorch.helpers.distributions import StraightThroughBernoulli

sys.path.append(MODEL_PATH)
# sys.path.insert(0, AGENT_TORCH_PATH)
from AgentTorch.LLM.llm_agent import LLMAgent
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import get_by_path

sys.path.append(MODEL_PATH)
from macro_economics.prompt import prompt_template_var,agent_profile
import itertools

class WorkConsumptionPropensity(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        OPENAI_API_KEY = self.config['simulation_metadata']['OPENAI_API_KEY']
        
        self.mode = self.config['simulation_metadata']['execution_mode']        
        self.mapping = self.load_mapping(self.config['simulation_metadata']['mapping_path'])
        self.variables = self.get_variables(prompt_template_var)
        self.filtered_mapping = self.filter_mapping(self.mapping,self.variables)
        self.combinations_of_prompt_variables, self.combinations_of_prompt_variables_with_index = self.get_combinations_of_prompt_variables(self.filtered_mapping)
        self.num_llm_agents = len(self.combinations_of_prompt_variables)
        self.agent = LLMAgent(agent_profile = agent_profile,openai_api_key = OPENAI_API_KEY,num_agents = self.num_llm_agents)
        self.save_memory_dir = self.config['simulation_metadata']['memory_dir']
        self.num_steps_per_episode = self.config['simulation_metadata']['num_steps_per_episode']

        self.st_bernoulli = StraightThroughBernoulli.apply
        
    
    async def forward(self, state, observation):        
        print("Substep Action: Earning decision")
        num_agents = self.config['simulation_metadata']['num_agents']
        number_of_months = state['current_step'] + 1
        current_episode = 1
        
        gender = get_by_path(state, re.split("/", self.input_variables['gender']))
        age = get_by_path(state,re.split("/", self.input_variables['age']))
        county = get_by_path(state,re.split("/", self.input_variables['county']))
        
        
        consumption_propensity = get_by_path(state,re.split("/", self.input_variables['consumption_propensity']))
        work_propensity = get_by_path(state,re.split("/", self.input_variables['work_propensity']))
        
        if self.mode == 'simple':
            print("Simple mode expts")
            work_propensity = torch.rand(num_agents,1)
            whether_to_work = torch.bernoulli(work_propensity)
            
            return {self.output_variables[0] : whether_to_work, 
                    self.output_variables[1] : work_propensity, 
                    self.output_variables[2] : consumption_propensity}
        
        print("LLM benchmark expts")
        masks = []
        output_values = []
        
        for target_values in self.combinations_of_prompt_variables_with_index:
            mask = torch.tensor([True]*len(gender))  # Initialize mask as tensor of True values
            for key, value in target_values.items():
                if key in locals():  # Check if variable with this name exists
                    mask = torch.logical_and(mask, locals()[key] == value)
            mask = mask.unsqueeze(1)  # Ensure consistent adding later
            float_mask = mask.float()
            masks.append(float_mask)
        
        # for target_values in self.combinations_of_prompt_variables_with_index:
        #     gender_mask = (gender == target_values['gender'])
        #     age_mask = (age == target_values['age'])
        #     mask = torch.logical_and(gender_mask, age_mask).unsqueeze(1) # ayush fix -> to ensure consistent adding later
        #     float_mask = mask.float()
        #     masks.append(float_mask)
        
        prompt_list = []
        for en,_ in enumerate(self.combinations_of_prompt_variables_with_index):
            age = self.combinations_of_prompt_variables[en]['age']
            gender = self.combinations_of_prompt_variables[en]['gender']
            county = self.combinations_of_prompt_variables[en]['county']
            prompt = prompt_template_var.format(age = age,gender = gender,county = county)
            prompt_list.append(prompt)

        agent_output = await self.agent(prompt_list)
        for en,output_value in enumerate(agent_output):
            output_value = json.loads(output_value['text'])
            group_work_propensity = output_value['work']
            group_consumption_propensity = output_value['consumption']
            consumption_propensity_for_group = masks[en]*group_consumption_propensity
            consumption_propensity = torch.add(consumption_propensity,consumption_propensity_for_group)
            work_propensity_for_group = masks[en]*group_work_propensity
            work_propensity = torch.add(work_propensity,work_propensity_for_group)

        # work_propensity = torch.rand(16573530,1)
        will_work = self.st_bernoulli(work_propensity) #torch.bernoulli(work_propensity)
        
        if number_of_months == self.num_steps_per_episode:
            current_memory_dir = os.path.join(self.save_memory_dir ,str(current_episode), str(number_of_months))
            self.agent.export_memory_to_file(current_memory_dir)
            
        
        return {self.output_variables[0] : will_work, 
                self.output_variables[1] : consumption_propensity}
    
    def augment_mapping_with_index(self,mapping):
        return {k: {v: i for i, v in enumerate(values)} for k, values in mapping.items()}

    def filter_mapping(self,mapping,variables):
        return {k: mapping[k] for k in variables}
    
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
        
        
