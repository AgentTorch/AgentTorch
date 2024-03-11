import pandas as pd
import numpy as np 
import sys
sys.path.insert(0, '/Users/shashankkumar/Documents/GitHub/MacroEcon/AgentTorch')
import torch
import torch.optim as optim

from AgentTorch import Runner, Registry

def opdyn_registry():
    reg = Registry()

    from substeps.macro_economics.transition import UpdateMacroeconomics, UpdateFinancialMarket, UpdateMonthCounter, UpdateSavings
    reg.register(UpdateMacroeconomics, "UpdateMacroeconomics", key="transition")
    reg.register(UpdateFinancialMarket, "UpdateFinancialMarket", key="transition")
    reg.register(UpdateMonthCounter, "UpdateMonthCounter", key="transition")
    reg.register(UpdateSavings, "UpdateSavings", key="transition")

    from substeps.macro_economics.action import CalculateWorkAndConsumptionPropensity
    reg.register(CalculateWorkAndConsumptionPropensity, "CalculateWorkAndConsumptionPropensity", key="policy")

    from AgentTorch.helpers import zeros, random_normal, constant, grid_network
    reg.register(zeros, "zeros", key="initialization")
    reg.register(random_normal, "random_normal", key="initialization")
    reg.register(constant, "constant", key="initialization")
    reg.register(grid_network, "grid", key="network")

    from substeps.utils import random_normal_col_by_col, load_population_attribute,initialize_id
    reg.register(random_normal_col_by_col, "random_normal_col_by_col", key="initialization")
    reg.register(load_population_attribute, "load_population_attribute", key="initialization")
    reg.register(initialize_id, "initialize_id", key="initialization")


    return reg

class OpDynRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.optimizer = optim.Adam(self.parameters(), 
        #         lr=self.config['simulation_metadata']['learning_params']['lr'], 
        #         betas=self.config['simulation_metadata']['learning_params']['betas'])

    def forward(self):
        for episode in range(self.config['simulation_metadata']['num_episodes']):
            num_steps_per_episode = self.config["simulation_metadata"]["num_steps_per_episode"]
            self.reset()
            self.step(num_steps_per_episode)

            #self.controller.learn_after_episode(jax.tree_map(lambda x: x[-1], self.trajectory), self.initializer, self.optimizer)

    def execute(self):
        self.forward()
