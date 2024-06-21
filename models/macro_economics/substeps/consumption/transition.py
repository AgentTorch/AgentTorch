import torch
import torch.nn as nn
from torch.distributions import Normal
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
import re
import pdb

class UpdateAssetsGoods(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
    
    def calculateGoodsInventory(self, state):
        # Calculate total production
        l = get_by_path(state, re.split("/", self.input_variables["work_propensity"]))
        A = self.config['simulation_metadata']['universal_productivity']
        G = get_by_path(state, re.split("/", self.input_variables['goods_inventory']))
        production_of_goods = (l * 168 * A).sum()
        # Update inventory (assuming units are compatible)
        updated_goods_inventory = G + production_of_goods
        return updated_goods_inventory
    
    def calculateIntendedConsumption(self, state):
        # Calculate intended consumption by agents for this step
        price_of_goods = get_by_path(state, re.split("/", self.input_variables['price_of_goods']))
        s = get_by_path(state, re.split("/", self.input_variables['assets']))
        l = get_by_path(state, re.split("/", self.input_variables['consumption_propensity']))
        
        intended_consumption = (s * l) / price_of_goods
        return intended_consumption
    
    def calculateTotalDemand(self, state, action):
        # Calculate total demand
        intended_consumption = self.calculateIntendedConsumption(state)
        total_demand = torch.sum(intended_consumption)
        return total_demand
    
    def calculateImbalance(self,goods_inventory,total_demand):
        # Calculate imbalance
        D = total_demand
        G = goods_inventory
        imbalance = (D - G)/torch.max(D, G)
        return imbalance

    def consumeGoods(self, state, goods_inventory, total_demand):
        # Consume goods
        D = total_demand
        G = goods_inventory
        assets = get_by_path(state, re.split("/", self.input_variables['assets']))
        good_inventory_after_consumption = torch.min((G - D), torch.zeros_like(G))
        assets_after_consumption = assets * torch.rand(1)
        return good_inventory_after_consumption, assets_after_consumption
    
    def forward(self, state, action):
        print("Substep: Agent Consumption")
        goods_inventory = self.calculateGoodsInventory(state)
        total_demand = self.calculateTotalDemand(state, action)
        imbalance = self.calculateImbalance(goods_inventory, total_demand)
        new_inventory,new_assets = self.consumeGoods(state,goods_inventory, total_demand)
    
        return {self.output_variables[0] : new_inventory, 
                self.output_variables[1] : new_assets, 
                self.output_variables[2] : total_demand,
                self.output_variables[3] : imbalance}
        

