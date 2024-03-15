import torch
import torch.nn as nn
from torch.distributions import Normal
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
import re
import pdb

class UpdateMacroRates(SubstepTransition):
    '''Macro quantities relevant to labor markets - hourly wage, unemployment rate, price of goods'''
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
            
    def calculateUnemploymentRate(self, working_status):
        l = working_status
        agg_l = torch.sum(torch.sum((1-l),dim=1),dim=0)
        
        unemployment_rate = agg_l / (l.size(0) * 12.0)
        return unemployment_rate
    
    def calculateHourlyWage(self, hourly_wage, imbalance):
        # Calculate hourly wage
        w = hourly_wage
        omega = imbalance.float()
        
        if omega > 0:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_wage']
            r2 = (max_rate_change * omega)
            r1 = 0
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            new_hourly_wage = w * (1 + sampled_omega)
        else:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_wage']
            r1 = (max_rate_change * omega)
            r2 = 0
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            # sampled_omega = torch.tensor.uniform((max_rate_change * omega),0)
            new_hourly_wage = w * (1 + sampled_omega)
        return new_hourly_wage

    def forward(self, state, action):
        print("Executing Substep: Labor Market")
        month_id = state['current_step']
        
        working_status = get_by_path(state, re.split("/", self.input_variables['will_work']))
        imbalance = get_by_path(state, re.split("/", self.input_variables['imbalance']))
        hourly_wage = get_by_path(state, re.split("/", self.input_variables['hourly_wage']))
                
        # unemployment rate
        unemployment_rate = self.calculateUnemploymentRate(working_status)
                
        # hourly wages
        new_hourly_wages = self.calculateHourlyWage(hourly_wage, imbalance)
                
        return {self.output_variables[0]: new_hourly_wages, 
                self.output_variables[1]: unemployment_rate}
