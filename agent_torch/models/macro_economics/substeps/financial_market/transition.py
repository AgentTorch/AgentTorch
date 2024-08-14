import torch
import torch.nn as nn
from torch.distributions import Normal
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
import re
import pdb

class UpdateFinancialMarket(SubstepTransition):
    '''macro quantities in the financial market - interest rate, inflation rate, price of goods'''
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        
    def calculateInflationRate(self, price_of_goods, avg_price_of_goods):
        Pn = price_of_goods
        Pm = avg_price_of_goods
        
        inflation_rate = (Pn - Pm) / Pm
        return inflation_rate
        
    def calculateInterestRate(self, inflation_rate, unemployment_rate):
        rn = self.config['simulation_metadata']['natural_interest_rate']
        un = self.config['simulation_metadata']['natural_unemployment_rate']
        pit = self.config['simulation_metadata']['target_inflation_rate']
        a_i = self.config['simulation_metadata']['inflation_adaptation_coefficient']
        a_u = self.config['simulation_metadata']['unemployment_adaptation_macro']
        
        pi = inflation_rate
        u = unemployment_rate
        
        interest_rate = rn + pit + a_i * (pi - pit) + a_u * (un - u)
        interest_rate = torch.max(interest_rate, torch.zeros_like(interest_rate))
        return interest_rate
    
    def calculateGoodsPrice(self, price_of_goods, imbalance):
        P = price_of_goods
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
    
    def calculateCumulativeGoodsPrice(self, cumulative_price_of_goods, price_of_goods):
        # pdb.set_trace()
        cumulative_price_of_goods = cumulative_price_of_goods + price_of_goods.squeeze()
        return cumulative_price_of_goods

    def forward(self, state, action):
        number_of_months = state['current_step'] + 1
        print("Executing Substep: Financial Market")
        
        inflation_rate = get_by_path(state, re.split("/", self.input_variables['inflation_rate']))
        unemployment_rate = get_by_path(state, re.split("/", self.input_variables['unemployment_rate']))

        # interest rate
        new_interest_rate = self.calculateInterestRate(inflation_rate, unemployment_rate)
        
        # price of goods
        price_of_goods = get_by_path(state, re.split("/", self.input_variables['price_of_goods']))
        cumulative_price_of_goods = get_by_path(state, re.split("/", self.input_variables['cumulative_price_of_goods']))
        imbalance = get_by_path(state, re.split("/", self.input_variables["imbalance"]))
        
        new_price_of_goods = self.calculateGoodsPrice(price_of_goods, imbalance)
        avg_price_of_goods = (cumulative_price_of_goods + new_price_of_goods) / number_of_months
        
        new_cumulative_price_of_goods = self.calculateCumulativeGoodsPrice(cumulative_price_of_goods, new_price_of_goods)
        
        # inflation rate
        new_inflation_rate = self.calculateInflationRate(price_of_goods, avg_price_of_goods)
        
        return {self.output_variables[0]: new_interest_rate,
               self.output_variables[1]: new_price_of_goods,
               self.output_variables[2]: new_cumulative_price_of_goods,
               self.output_variables[3]: new_inflation_rate}
                
        
        

        
        
    

