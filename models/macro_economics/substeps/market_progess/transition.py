import torch
import torch.nn as nn
from torch.distributions import Normal
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
import re
import pdb

class UpdateMacroRates(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
    
    def forward(self, state, action):
        number_of_months = get_by_path(state, re.split("/", self.input_variables['Month_Counter']))
        cummulative_price_of_goods = get_by_path(state, re.split("/", self.input_variables['Cummulative_Price_of_Goods']))
        imbalance = self.calculateImbalance(state, action)
        avg_price_of_goods = (cummulative_price_of_goods + price_of_goods) / number_of_months
        interest_rate = self.calculateInterestRate(state, action)
        unemployment_rate = self.calculateUnemploymentRate(state, action)
        return {self.output_variables[0] : interest_rate, self.output_variables[1] : unemployment_rate}
    
    

    def calculateHourlyWage(self, state,action,imbalance):
        # Calculate hourly wage
        w = get_by_path(state, re.split("/", self.input_variables['Hourly_Wage']))
        omega = imbalance.float()
        
        if omega > 0:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_wage']
            r2 = (max_rate_change * omega)
            r1 = 0
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            hourly_wage = w * (1 + sampled_omega)
        else:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_wage']
            r1 = (max_rate_change * omega)
            r2 = 0
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            hourly_wage = w * (1 + sampled_omega)
        return hourly_wage
    
    def calculateGoodsPrice(self, state,action,imbalance):
        P = get_by_path(state, re.split("/", self.input_variables['Price_of_Goods']))
        omega = imbalance.float()
        if omega > 0:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_price']
            r2 = (max_rate_change * omega)
            r1 = 0
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            price_of_goods = P * (1 + sampled_omega)
        else:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_price']
            r1 = (max_rate_change * omega)
            r2 = 0
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            price_of_goods = P * (1 + sampled_omega)
        return price_of_goods
    
    def calculateInflationRate(self, state, action, avg_price_of_goods, price_of_goods):
        Pn = price_of_goods
        Pm = avg_price_of_goods
        inflation_rate = (Pn - Pm) / Pm
        return inflation_rate


class UpdateFinancialMarket(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
    
    def forward(self, state, action):
        interest_rate = self.calculateInterestRate(state, action)
        number_of_months = get_by_path(state, re.split("/", self.input_variables['Month_Counter']))
        if number_of_months % 12 == 0:
            unemployment_rate = self.calculateUnemploymentRate(state, action)
            return {self.output_variables[0] : interest_rate, self.output_variables[1] : unemployment_rate}
        else:
            unemployment_rate = get_by_path(state, re.split("/", self.input_variables['Unemployment_Rate']))
            return {self.output_variables[0] : interest_rate, self.output_variables[1] : unemployment_rate}
    
    def calculateInterestRate(self,state,action):
        rn = self.config['simulation_metadata']['natural_interest_rate']
        un = self.config['simulation_metadata']['natural_unemployment_rate']
        pit = self.config['simulation_metadata']['target_inflation_rate']
        a_i = self.config['simulation_metadata']['inflation_adaptation_coefficient']
        a_u = self.config['simulation_metadata']['unemployment_adaptation_coefficient']
        pi = get_by_path(state, re.split("/", self.input_variables['Inflation_Rate']))
        u = get_by_path(state, re.split("/", self.input_variables['Unemployment_Rate']))
        interest_rate = rn + pit + a_i * (pi - pit) + a_u * (un - u)
        interest_rate = torch.max(interest_rate, torch.zeros_like(interest_rate))
        return interest_rate
    
    
    

