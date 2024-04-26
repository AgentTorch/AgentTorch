import asyncio
import json
import os
import torch
import re
import sys
import pdb
from AgentTorch.helpers.distributions import StraightThroughBernoulli

# sys.path.insert(0, AGENT_TORCH_PATH)
from AgentTorch.LLM.llm_agent import LLMAgent
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import get_by_path

from macro_economics.prompt import prompt_template_var,agent_profile
import itertools

class WorkConsumptionPropensity(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        OPENAI_API_KEY = self.config['simulation_metadata']['OPENAI_API_KEY']
        self.month_mapping = self.config['simulation_metadata']['month_mapping']
        self.year_mapping = self.config['simulation_metadata']['year_mapping']
        self.mode = self.config['simulation_metadata']['execution_mode']        
        self.save_memory_dir = self.config['simulation_metadata']['memory_dir']
        self.num_steps_per_episode = self.config['simulation_metadata']['num_steps_per_episode']
        self.num_agents = self.config['simulation_metadata']['num_agents']
        
        self.mapping = self.load_mapping(self.config['simulation_metadata']['mapping_path'])
        self.variables = self.get_variables(prompt_template_var)
        self.filtered_mapping = self.filter_mapping(self.mapping,self.variables)
        self.combinations_of_prompt_variables, self.combinations_of_prompt_variables_with_index = self.get_combinations_of_prompt_variables(self.filtered_mapping)
        self.num_llm_agents = len(self.combinations_of_prompt_variables)
        self.agent = LLMAgent(agent_profile = agent_profile,openai_api_key = OPENAI_API_KEY,num_agents = self.num_llm_agents)

        self.st_bernoulli = StraightThroughBernoulli.apply
        
    
    async def forward(self, state, observation):        
        print("Substep Action: Earning decision")
        number_of_months = state['current_step'] + 1
        current_year = number_of_months // 12 + 1 # 1 indexed
        year = self.year_mapping[current_year]
        consumption_propensity = get_by_path(state,re.split("/", self.input_variables['consumption_propensity']))
        work_propensity = get_by_path(state,re.split("/", self.input_variables['work_propensity']))
        
        prompt_variables_dict = self.get_prompt_variables_dict(state)
        masks = self.get_masks_for_each_group(prompt_variables_dict)
        prompt_list = self.get_prompt_list(prompt_variables_dict)
        # await asyncio.sleep(10)
        agent_output = self.agent(prompt_list,last_k=1)
        consumption_propensity, work_propensity = self.get_propensity_values(consumption_propensity, work_propensity, masks, agent_output)
        will_work = self.st_bernoulli(work_propensity) #torch.bernoulli(work_propensity)
        
        if number_of_months == self.num_steps_per_episode:
            current_memory_dir = os.path.join(self.save_memory_dir ,str(current_year), str(number_of_months))
            self.agent.export_memory_to_file(current_memory_dir)
            
        return {self.output_variables[0] : will_work, 
                self.output_variables[1] : consumption_propensity}

    def get_propensity_values(self, consumption_propensity, work_propensity, masks, agent_output):
        for en,output_value in enumerate(agent_output):
            output_value = json.loads(output_value)
            # group_work_propensity = output_value['work']
            # group_consumption_propensity = output_value['consumption']
            group_work_propensity = output_value[0] # changed for dspy compatibility
            group_consumption_propensity = output_value[1] # changed for dspy compatibility
            consumption_propensity_for_group = masks[en]*group_consumption_propensity
            consumption_propensity = torch.add(consumption_propensity,consumption_propensity_for_group)
            work_propensity_for_group = masks[en]*group_work_propensity
            work_propensity = torch.add(work_propensity,work_propensity_for_group)
        return consumption_propensity,work_propensity

    def get_prompt_list(self, variables):
        prompt_list = []
        for en,_ in enumerate(self.combinations_of_prompt_variables_with_index):
            prompt_values = self.combinations_of_prompt_variables[en]
            for key, value in variables.items():
                if isinstance(value, (int, str, float, bool)):
                    prompt_values[key] = value
            prompt = prompt_template_var.format(**prompt_values)
            prompt_list.append(prompt)
        return prompt_list

    def get_masks_for_each_group(self, variables):
        print("LLM benchmark expts")
        masks = []
        output_values = []
        
        for target_values in self.combinations_of_prompt_variables_with_index:
            mask = torch.tensor([True]*self.num_agents)  # Initialize mask as tensor of True values
            for key, value in target_values.items():
                if key in variables:  # Check if variable with this name exists
                    mask = torch.logical_and(mask, variables[key] == value)
            mask = mask.unsqueeze(1)  # Ensure consistent adding later
            float_mask = mask.float()
            masks.append(float_mask)
        return masks

    def get_prompt_variables_dict(self, state):      
        number_of_months = state['current_step'] + 1
        current_month = number_of_months % 12
        month = self.month_mapping[current_month]
        
        current_year = number_of_months // 12 + 1 # 1 indexed
        year = self.year_mapping[current_year]
        variables = {}
        for key in self.variables:
            variables[key] = get_by_path(state, re.split("/", self.input_variables[key])) if key in self.input_variables else locals()[key]

        variables['inflation_rate'] = variables['inflation_rate'][-1].item()
        variables['interest_rate'] = variables['interest_rate'][-1][-1].item()
        variables['unemployment_rate'] = variables['unemployment_rate'][-1][-1].item()
        variables['price_of_goods'] = variables['price_of_goods'][-1][-1].item()
        return variables
    
    def augment_mapping_with_index(self,mapping):
        return {k: {v: i for i, v in enumerate(values)} for k, values in mapping.items()}

    def filter_mapping(self,mapping,variables):
        return {k: mapping[k] for k in variables if k in mapping}
    
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
        
        
