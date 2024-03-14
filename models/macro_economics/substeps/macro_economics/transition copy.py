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
    
class UpdateAssetsGoods(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
    
    def forward(self, state, action):
        goods_inventory = self.calculateGoodsInventory(state, action)
        total_demand = self.calculateTotalDemand(state, action)
        imbalance = self.calculateImbalance(state, action, goods_inventory, total_demand)
        new_inventory,new_savings = self.consumeGoods(state, action, goods_inventory, total_demand)
    
        return {self.output_variables[0] : new_inventory, 
                self.output_variables[1] : new_savings, 
                self.output_variables[2] : total_demand,
                self.output_variables[3] : imbalance}
    
    def calculateGoodsInventory(self, state,action):
        # Calculate total production
        l = action['consumers']['work_propensity']
        A = self.config['simulation_metadata']['universal_productivity']
        G = get_by_path(state, re.split("/", self.input_variables['goods_inventory']))
        production = (l * 168 * A).sum()
        # Update inventory (assuming units are compatible)
        new_inventory = G + production
        return new_inventory
    
    def calculateIntendedConsumption(self, state, action):
        # Calculate intended consumption
        price_of_goods = get_by_path(state, re.split("/", self.input_variables['price_of_goods']))
        s = get_by_path(state, re.split("/", self.input_variables['savings']))
        l = get_by_path(state, re.split("/", self.input_variables['consumption_propensity']))
        intended_consumption = (s * l) / price_of_goods
        return intended_consumption
    
    def calculateTotalDemand(self, state, action):
        # Calculate total demand
        intended_consumption = self.calculateIntendedConsumption(state, action)
        total_demand = torch.sum(intended_consumption)
        return total_demand
    
    def calculateImbalance(self,state,action,goods_inventory,total_demand):
        # Calculate imbalance
        D = total_demand
        G = goods_inventory
        imbalance = (D - G)/torch.max(D, G)
        return imbalance

    def consumeGoods(self, state, action, goods_inventory, total_demand):
        # Consume goods
        D = total_demand
        G = goods_inventory
        savings = get_by_path(state, re.split("/", self.input_variables['savings']))
        new_inventory = torch.min((G - D), torch.zeros_like(G))
        new_savings = savings * torch.rand(1)
        return new_inventory, new_savings
    

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
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            hourly_wage = w * (1 + sampled_omega)
        else:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_wage']
            r1 = (max_rate_change * omega)
            r2 = 0
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            # sampled_omega = torch.tensor.uniform((max_rate_change * omega),0)
            hourly_wage = w * (1 + sampled_omega)
        return hourly_wage
    
    def calculateGoodsPrice(self, state,action,imbalance):
        P = get_by_path(state, re.split("/", self.input_variables['Price_of_Goods']))
        omega = imbalance.float()
        if omega > 0:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_price']
            r2 = (max_rate_change * omega)
            r1 = 0
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            # sampled_omega = torch.uniform(0, (max_rate_change * omega))
            price_of_goods = P * (1 + sampled_omega)
        else:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_price']
            r1 = (max_rate_change * omega)
            r2 = 0
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            price_of_goods = P * (1 + sampled_omega)
        return price_of_goods
    
    def calculateInflationRate(self, state, action, avg_price_of_goods, price_of_goods):
        Pn = price_of_goods
        Pm = avg_price_of_goods
        
        inflation_rate = (Pn - Pm) / Pm
        return inflation_rate
    
    
class UpdateMacroeconomics(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
    
    def forward(self, state, action):
        number_of_months = get_by_path(state, re.split("/", self.input_variables['Month_Counter']))
        #Produce goods
        goods_inventory = self.calculateGoodsInventory(state, action)
        #Calculate demand of goods
        total_demand = self.calculateTotalDemand(state, action)
        #Calculate imbalance
        imbalance = self.calculateImbalance(state, action,goods_inventory,total_demand)
        # imbalance = 0.3
        #Calculate hourly wage and Price of goods
        hourly_wage = self.calculateHourlyWage(state, action, imbalance)
        price_of_goods = self.calculateGoodsPrice(state, action, imbalance)
        cummulative_price_of_goods = get_by_path(state, re.split("/", self.input_variables['Cummulative_Price_of_Goods']))
        avg_price_of_goods = (cummulative_price_of_goods + price_of_goods) / number_of_months
        #Consume goods
        cummulative_price_of_goods += price_of_goods.squeeze()
        new_inventory = self.consumeGoods(state, action,goods_inventory,total_demand)
        inflation_rate = self.calculateInflationRate(state, action, avg_price_of_goods, price_of_goods)
        
        return {self.output_variables[0] : new_inventory, self.output_variables[1] : total_demand, self.output_variables[2] : imbalance, self.output_variables[3] : hourly_wage, self.output_variables[4] : price_of_goods, self.output_variables[5] : cummulative_price_of_goods, self.output_variables[6] : inflation_rate}

    def calculateGoodsInventory(self, state,action):
        # Calculate total production
        l = action['consumers']['Work_Propensity']
        A = self.config['simulation_metadata']['universal_productivity']
        G = get_by_path(state, re.split("/", self.input_variables['Goods_Inventory']))
        production = (l * 168 * A).sum()
        # Update inventory (assuming units are compatible)
        new_inventory = G + production
        return new_inventory
    
    def calculateIntendedConsumption(self, state, action):
        # Calculate intended consumption
        price_of_goods = get_by_path(state, re.split("/", self.input_variables['Price_of_Goods']))
        s = get_by_path(state, re.split("/", self.input_variables['Savings']))
        l = get_by_path(state, re.split("/", self.input_variables['Consumption_Propensity']))
        intended_consumption = (s * l) / price_of_goods
        return intended_consumption
    
    def calculateTotalDemand(self, state, action):
        # Calculate total demand
        intended_consumption = self.calculateIntendedConsumption(state, action)
        total_demand = torch.sum(intended_consumption)
        return total_demand
    
    def calculateImbalance(self,state,action,goods_inventory,total_demand):
        # Calculate imbalance
        D = total_demand
        G = goods_inventory
        imbalance = (D - G)/torch.max(D, G)
        return imbalance

    def calculateHourlyWage(self, state,action,imbalance):
        # Calculate hourly wage
        w = get_by_path(state, re.split("/", self.input_variables['Hourly_Wage']))
        omega = imbalance.float()
        
        if omega > 0:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_wage']
            r2 = (max_rate_change * omega)
            r1 = 0
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            hourly_wage = w * (1 + sampled_omega)
        else:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_wage']
            r1 = (max_rate_change * omega)
            r2 = 0
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            # sampled_omega = torch.tensor.uniform((max_rate_change * omega),0)
            hourly_wage = w * (1 + sampled_omega)
        return hourly_wage
    
    def calculateGoodsPrice(self, state,action,imbalance):
        P = get_by_path(state, re.split("/", self.input_variables['Price_of_Goods']))
        omega = imbalance.float()
        if omega > 0:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_price']
            r2 = (max_rate_change * omega)
            r1 = 0
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            # sampled_omega = torch.uniform(0, (max_rate_change * omega))
            price_of_goods = P * (1 + sampled_omega)
        else:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_price']
            r1 = (max_rate_change * omega)
            r2 = 0
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            price_of_goods = P * (1 + sampled_omega)
        return price_of_goods
    
    def calculateInflationRate(self, state, action, avg_price_of_goods, price_of_goods):
        Pn = price_of_goods
        Pm = avg_price_of_goods
        
        inflation_rate = (Pn - Pm) / Pm
        return inflation_rate
    
    def consumeGoods(self, state, action, goods_inventory, total_demand):
        # Consume goods
        D = total_demand
        G = goods_inventory
        new_inventory = torch.min((G - D), torch.zeros_like(G))
        return new_inventory
    
    def calculateUnemploymentRate(self, state, action):
        l = get_by_path(state, re.split("/", self.input_variables['weather_to_work']))
        agg_l = torch.sum(torch.sum((1-l),dim=1),dim=0)
        unemployment_rate = agg_l / (l.size(0) * 12.0)
        return unemployment_rate

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
    
    
    

