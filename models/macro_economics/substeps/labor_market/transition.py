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
            
    # def calculateUnemploymentRate(self, working_status):
    #     l = working_status
    #     agg_l = torch.sum(torch.sum((1-l),dim=1),dim=0)
        
    #     unemployment_rate = agg_l / (l.size(0) * 12.0)
    #     return unemployment_rate
    
    def calculateNumberOfAgentsNotWorking(self, working_status):
        l = working_status
        agents_not_working = torch.sum(torch.sum((1-l),dim=1),dim=0)
        return agents_not_working
    
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
    
    def calculateUnemploymentRate(self, Labor_Force_Participation_Rate,Gender_Ratio,Imbalance):
        coeff_0,coeff_1,coeff_2,coeff_3,error_coeff = self.args['coeff_0'],self.args['coeff_1'], self.args['coeff_2'], self.args['coeff_3'], self.args['error_coeff']
        unemployment_rate = coeff_0 + (coeff_1 * Imbalance + coeff_2 * Labor_Force_Participation_Rate + coeff_3 * Gender_Ratio) + error_coeff
        return unemployment_rate

    def forward(self, state, action):
        print("Executing Substep: Labor Market")
        month_id = state['current_step']
        
        working_status = get_by_path(state, re.split("/", self.input_variables['will_work']))
        imbalance = get_by_path(state, re.split("/", self.input_variables['imbalance']))
        hourly_wage = get_by_path(state, re.split("/", self.input_variables['hourly_wage']))
                
        # unemployment rate
        agents_not_working = self.calculateNumberOfAgentsNotWorking(working_status)
        agent_willing_to_work = working_status.shape[0] - agents_not_working
        Labor_Force_Participation_Rate = agent_willing_to_work / working_status.shape[0]
        
        unemployment_adaptation_coefficient = self.args['unemployment_adaptation_coefficient']
        # unemployment_rate = B0 + B1(GDP Growth Rate) + B2(Job Creation Rate) + β₃(Labor Force Participation Rate) + β₄(Education Level) + β₅(Industry Shift) + β₆(Government Spending) + ε
        unemployed_agents = torch.ceil(agents_not_working * unemployment_adaptation_coefficient)
        num_agents = working_status.shape[0]
        unemployment_rate = unemployed_agents / num_agents
                
        # hourly wages
        new_hourly_wages = self.calculateHourlyWage(hourly_wage, imbalance)
                
        return {self.output_variables[0]: new_hourly_wages, 
                self.output_variables[1]: unemployment_rate}
