import asyncio
import json
import os
import pandas as pd
import torch
import re
from AgentTorch.LLM.llm_agent import LLMAgent
from AgentTorch.helpers.distributions import StraightThroughBernoulli

from AgentTorch.LLM.llm_agent import LLMAgent, BasicQAEcon, COT
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import get_by_path

from macro_economics.prompt import agent_profile
import itertools
import time

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
        self.covid_cases_path = self.config['simulation_metadata']['covid_cases_path']
        self.mapping = self.load_mapping(self.config['simulation_metadata']['mapping_path'])
        self.device = torch.device(self.config['simulation_metadata']['device'])
        self.prompt = self.config['simulation_metadata']['EARNING_ACTION_PROMPT']
        self.variables = self.get_variables(self.prompt)
        self.filtered_mapping = self.filter_mapping(self.mapping,self.variables)
        self.combinations_of_prompt_variables, self.combinations_of_prompt_variables_with_index = self.get_combinations_of_prompt_variables(self.filtered_mapping)
        self.num_llm_agents = len(self.combinations_of_prompt_variables)
        self.agent = LLMAgent(BasicQAEcon,COT,openai_api_key = OPENAI_API_KEY,num_agents = self.num_llm_agents)
        self.expt_mode = self.config['simulation_metadata']['expt_mode']
        self.st_bernoulli = StraightThroughBernoulli.apply
        self.covid_cases = pd.read_csv(self.covid_cases_path)
        self.covid_cases = torch.tensor(self.covid_cases.values) # add to device
        
    
    def forward(self, state, observation):        
        # TODO: Improve retry logic when LLM's output doesn't match the expected format
        print("Substep Action: Earning decision")
        number_of_months = state['current_step'] + 1 + 7 # 1 indexed, also since we are starting from 8th month add 7 here
        current_year = number_of_months // 12 + 1 + 1 # 1 indexed, also since we are starting from 2020 add 1 here
        year = self.year_mapping[current_year]
        # consumption_propensity = get_by_path(state,re.split("/", self.input_variables['consumption_propensity']))
        # work_propensity = get_by_path(state,re.split("/", self.input_variables['work_propensity']))
        consumption_propensity = torch.zeros(self.num_agents,1).to(self.device)
        work_propensity = torch.zeros(self.num_agents,1).to(self.device)
        if self.expt_mode == 'LLM_PEER' or self.expt_mode == 'PER_AGENT_LLM' or self.expt_mode == "LLM_WITH_SIM_INPUTS":
            prompt_variables_dict = self.get_prompt_variables_dict(state)
            masks = self.get_masks_for_each_group(prompt_variables_dict)
            prompt_list = self.get_prompt_list(prompt_variables_dict)
            
            for num_retries in range(10):  # Retry up to 10 times
                try:
                    start_time = time.time()
                    agent_output = self.agent(prompt_list, last_k=3)
                    end_time = time.time()

                    execution_time = end_time - start_time
                    print(f"Execution time: {execution_time} seconds")
                    consumption_propensity, work_propensity = self.get_propensity_values(consumption_propensity, work_propensity, masks, agent_output)
                    break  # If successful, break out of the loop
                except Exception as e:
                    print(f"Error in getting propensity values: {e}")
                    print("retrying")
                    continue
            else:  # If we've exhausted all retries, re-raise the last exception
                    raise
            current_memory_dir = os.path.join(self.save_memory_dir ,str(year), str(number_of_months))
            self.agent.export_memory_to_file(current_memory_dir)
            self.agent.inspect_history(len(prompt_list),current_memory_dir)
        
        else:
            work_propensity = torch.rand(self.num_agents,1)
            consumption_propensity = torch.rand(self.num_agents,1)
        will_work = self.st_bernoulli(work_propensity) 
        
        # self.agent.show_reasoning(3)
        return {self.output_variables[0] : will_work, 
                self.output_variables[1] : consumption_propensity}
    
    # def per_agent_prompt(self,variables):
    #     current_step = state['current_step']    
    #     number_of_months = current_step + 1 + 7 # 1 indexed, also since we are starting from 8th month add 7 here
    #     current_year = number_of_months // 12 + 1 + 1 # 1 indexed, also since we are starting from 2020 add 1 here 
    #     current_month = number_of_months % 12 + 1
    #     month = self.month_mapping[current_month]
    #     covid_cases = self.covid_cases[current_step] #* 10
    #     year = self.year_mapping[current_year]
    #     gender = 
        
        
    #     prompt = self.prompt
    #     for key, value in variables.items():
    #         if isinstance(value, (int, str, float, bool)):
    #             prompt = prompt.replace("{" + key + "}", str(value))
    #     return prompt
    
    def get_propensity_values(self, consumption_propensity, work_propensity, masks, agent_output):
        for en,output_value in enumerate(agent_output):
            output_value = json.loads(output_value)
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
            prompt = self.prompt.format(**prompt_values)
            prompt_list.append(prompt)
        return prompt_list

    def get_masks_for_each_group(self, variables):
        print("LLM benchmark expts")
        masks = []
        output_values = []
        keys_to_remove = []
        for en,target_values in enumerate(self.combinations_of_prompt_variables_with_index):
            
            mask = torch.tensor([True]*self.num_agents)  # Initialize mask as tensor of True values
            for key, value in target_values.items():
                if key in variables:  # Check if variable with this name exists
                    if key == 'age' and value == 0:
                        mask = torch.zeros_like(mask)
                    else:
                        mask = torch.logical_and(mask, variables[key] == value)
            mask = mask.unsqueeze(1)  # Ensure consistent adding later
            float_mask = mask.float()
            masks.append(float_mask)
        
            # if float_mask.sum() == 0:
            #     keys_to_remove.append(en)
            # else:
            #     masks.append(float_mask)
        
        # Remove peers which have no agents. A change done for "one llm agent per agent"
        # for key in reversed(keys_to_remove):
        #     try:
        #         del self.combinations_of_prompt_variables[key]
        #         del self.combinations_of_prompt_variables_with_index[key]
        #     except Exception as e:
        #         print(e)
        #         pass
        return masks

    def get_prompt_variables_dict(self, state): 
        current_step = state['current_step']    
        number_of_months = current_step + 1 + 7 # 1 indexed, also since we are starting from 8th month add 7 here
        current_year = number_of_months // 12 + 1 + 1 # 1 indexed, also since we are starting from 2020 add 1 here 
        current_month = number_of_months % 12 + 1
        month = self.month_mapping[current_month]
        covid_cases = self.covid_cases[current_step] #* 10
        year = self.year_mapping[current_year]
        payment = 0
        if month == 'December' and year == 2020:
            payment = 1200
        elif month == 'March' and year == 2021:
            payment = 1200
        else:
            payment = 0
        
        variables = {}
        for key in self.variables:
            variables[key] = get_by_path(state, re.split("/", self.input_variables[key])) if key in self.input_variables else locals()[key]

        # variables['inflation_rate'] = variables['inflation_rate'][-1].item()
        if 'interest_rate' in variables.keys():
            variables['interest_rate'] = variables['interest_rate'][-1][-1].item()
        if 'unemployment_rate' in variables.keys():
            variables['unemployment_rate'] = variables['unemployment_rate'][-1][-1].item()
        if 'price_of_goods' in variables.keys():
            variables['price_of_goods'] = variables['price_of_goods'][-1][-1].item()
        if 'covid_cases' in variables.keys():
            variables['covid_cases'] = covid_cases.item()
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
        
        
