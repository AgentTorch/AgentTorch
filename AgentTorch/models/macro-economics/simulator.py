import pandas as pd
import numpy as np 
import sys
sys.path.insert(0, '/Users/shashankkumar/Documents/AgentTorch_Official/AgentTorch')
import torch
import torch.optim as optim

from AgentTorch import Runner, Registry

def opdyn_registry():
    reg = Registry()

    from substeps.purchase_product.transition import UpdateMacroeconomics, UpdateFinancialMarket, UpdateMonthCounter, UpdateSavings
    reg.register(UpdateMacroeconomics, "UpdateMacroeconomics", key="transition")
    reg.register(UpdateFinancialMarket, "UpdateFinancialMarket", key="transition")
    reg.register(UpdateMonthCounter, "UpdateMonthCounter", key="transition")
    reg.register(UpdateSavings, "UpdateSavings", key="transition")

    from substeps.purchase_product.action import CalculateWorkPropensity, CalculateConsumptionPropensity
    reg.register(CalculateWorkPropensity, "CalculateWorkPropensity", key="policy")
    reg.register(CalculateConsumptionPropensity, "CalculateConsumptionPropensity", key="policy")

    from AgentTorch.helpers import zeros, random_normal, constant, grid_network
    reg.register(zeros, "zeros", key="initialization")
    reg.register(random_normal, "random_normal", key="initialization")
    reg.register(constant, "constant", key="initialization")
    reg.register(grid_network, "grid", key="network")

    from substeps.utils import random_normal_col_by_col
    reg.register(random_normal_col_by_col, "random_normal_col_by_col", key="initialization")

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
