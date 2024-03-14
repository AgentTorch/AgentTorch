import torch
import torch.nn as nn
from torch.distributions import Normal
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
import re
import pdb

class UpdateAssets(SubstepTransition):
    def __init__(self,config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
    
    def forward(self, state, action):
        assets,monthly_income = self.calculateAssets(state, action)
        return {self.output_variables[0] : assets, self.output_variables[1] : monthly_income}
    
    def calculateAssets(self, state, action,assets = None):
        if assets is None:
            assets = get_by_path(state, re.split("/", self.input_variables['assets']))
        number_of_months = get_by_path(state, re.split("/", self.input_variables['Month_Counter']))
        if number_of_months % 12 == 0:
            assets = self.increaseAssetsAnnualy(state,action)
            
        monthly_income,total_tax = self.calculatePostTaxIncome(state,action)
        post_distribution_income = self.distributeTax(state,action,total_tax,monthly_income)
        
        total_assets = assets + post_distribution_income
        return total_assets, post_distribution_income
    
    def calculateMonthlyIncome(self, state, action):
        hourly_wage = get_by_path(state, re.split("/", self.input_variables['hourly_wage']))
        l = action['consumers']['Whether_to_Work']
        hours_worked = self.config['simulation_metadata']['hours_worked']
        monthly_income = get_by_path(state, re.split("/", self.input_variables['monthly_income']))
        monthly_income_per_agent = hourly_wage * hours_worked
        monthly_income = (monthly_income*l) + monthly_income_per_agent
        
        return monthly_income
        
    def calculatePostTaxIncome(self, state, action):
        brackets = self.config['simulation_metadata']['tax_brackets']
        brackets = torch.tensor(brackets)
        rates = self.config['simulation_metadata']['tax_rates']
        rates = torch.tensor(rates)
        zi = self.calculateMonthlyIncome(state, action)
        tax_per_agent = torch.zeros_like(zi)
        zi_copy = zi.clone().detach()
        if len(brackets) != len(rates):
            raise ValueError("Number of brackets and rates must be equal.")
        num_brackets = len(brackets)
        bin_indices = torch.bucketize(zi, brackets, right=True)
        min_bin_index = torch.clamp(bin_indices, 0, num_brackets - 1)
        tax_within_bracket = []
        for i in range(num_brackets):
            mask = (min_bin_index == i+1)
            mask_float = mask.float()
            base_income = brackets[i]
            tax_per_agent = tax_per_agent + ((((zi*mask_float) - base_income)*rates[i]))*mask_float
            bin_income = torch.masked_select(zi_copy, mask)
            tax_within_bracket.append(
                (bin_income - base_income) * rates[i].masked_select(mask)
            )
        total_tax = torch.cat(tax_within_bracket).sum(dim=0)
        zi  = zi - tax_per_agent
        return zi , total_tax
    
    def distributeTax(self, state, action, total_tax,zi):
        # Distribute tax
        num_agents = self.config['simulation_metadata']['num_agents']
        tax_distribution = total_tax / num_agents
        zi = zi + tax_distribution
        return zi
        
    def increaseAssetsAnnualy(self, state, action):
        # Calculate new assets
        s = get_by_path(state, re.split("/", self.input_variables['assets']))
        r = get_by_path(state, re.split("/", self.input_variables['interest_rate']))
        new_assets = s * (1+r)
        return new_assets

class WriteToState(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
    
    def forward(self, state, action):
        consumption_propensity = action['consumers']['Consumption_Propensity']
        weather_to_work = action['consumers']['Whether_to_Work']
        return {self.output_variables[0] : consumption_propensity, self.output_variables[1] : weather_to_work}
