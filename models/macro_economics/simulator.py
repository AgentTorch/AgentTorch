AGENT_TORCH_PATH = '/u/ayushc/projects/GradABM/MacroEcon/AgentTorch'

import pandas as pd
import numpy as np 
import sys
sys.path.insert(0, AGENT_TORCH_PATH)
import torch
import torch.optim as optim

from AgentTorch import Runner, Registry

def opdyn_registry():
    reg = Registry()

    # Agent earning behavior
    from substeps.earning.action import WorkConsumptionPropensity
    reg.register(WorkConsumptionPropensity, "get_work_consumption_decision", key="policy")
    
    from substeps.earning.transition import UpdateAssets, WriteActionToState
    reg.register(UpdateAssets, "update_assets", key="transition")
    reg.register(WriteActionToState, "write_action_to_state", key="transition")
    
    # Agents spending behavior
    from substeps.consumption.transition import UpdateAssetsGoods
    reg.register(UpdateAssetsGoods, "update_assets_and_goods", key="transition")
    
    # Market Evolves macro quantities - Labor Market
    from substeps.labor_market.transition import UpdateMacroRates
    reg.register(UpdateMacroRates, "update_macro_rates", key="transition")
    
    # Financial market evolves interest rate and tax brackets - FED
    from substeps.financial_market.transition import UpdateFinancialMarket
    reg.register(UpdateFinancialMarket, "update_financial_market", key="transition")
        
    from AgentTorch.helpers import zeros, random_normal, constant, grid_network
    reg.register(zeros, "zeros", key="initialization")
    reg.register(random_normal, "random_normal", key="initialization")
    reg.register(constant, "constant", key="initialization")
    reg.register(grid_network, "grid", key="network")

    from substeps.utils import random_normal_col_by_col, load_population_attribute, initialize_id
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
            print("episode: ", episode)
            num_steps_per_episode = self.config["simulation_metadata"]["num_steps_per_episode"]
            self.reset()
            self.step(num_steps_per_episode)

            #self.controller.learn_after_episode(jax.tree_map(lambda x: x[-1], self.trajectory), self.initializer, self.optimizer)

    def execute(self):
        self.forward()
