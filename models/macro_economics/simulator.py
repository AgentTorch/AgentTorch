# AGENT_TORCH_PATH = '/u/ayushc/projects/GradABM/MacroEcon/AgentTorch'
#'/Users/shashankkumar/Documents/GitHub/MacroEcon/AgentTorch'

import pickle
import pandas as pd
import numpy as np 
import sys
# sys.path.append(AGENT_TORCH_PATH)
import torch
import torch.optim as optim
import sys
# sys.path.append("/Users/shashankkumar/Documents/GitHub/MacroEcon/AgentTorch/AgentTorch")
from AgentTorch import Runner, Registry

def simulation_registry():
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

class SimulationRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_path = '/Users/shashankkumar/Documents/GitHub/MacroEcon/simulator_data/NYC/brooklyn_unemp.csv'
        df = pd.read_csv(data_path)
        df.sort_values(by=['year','month'],ascending=True,inplace=True)
        arr = df['unemployment_rate'].values
        tensor = torch.from_numpy(arr)
        self.unemployment_test_dataset = tensor.view(5,-1)
        self.mse_loss = torch.nn.MSELoss()
        self.state_data_dict = {}
        
    def forward(self):
        # for name, params in self.named_parameters():
        #     print(name)
        # for params in self.parameters():
        #     print(params)
        self.optimizer = optim.Adam(self.parameters(), 
                lr=self.config['simulation_metadata']['learning_params']['lr'], 
                betas=self.config['simulation_metadata']['learning_params']['betas'])
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 
                self.config['simulation_metadata']['learning_params']['lr_gamma'])

        for episode in range(self.config['simulation_metadata']['num_episodes']):
            print("episode: ", episode)
            num_steps_per_episode = self.config["simulation_metadata"]["num_steps_per_episode"]
            self.step(num_steps_per_episode)
            unemployment_rate = self.state_trajectory[-1][-1]['environment']['U'].squeeze()
            loss = unemployment_rate.sum()
            # test_set_for_episode = self.unemployment_test_dataset[episode][:num_steps_per_episode].float().squeeze()
            # loss =  self.mse_loss(unemployment_rate, test_set_for_episode)
            loss.backward()
            print([(p, p.grad) for p in self.parameters()])
            # breakpoint()
            # for param in self.parameters():
            #     print(param.grad)
            self.optimizer.step()

            self.optimizer.zero_grad()
            current_episode_state_data_dict = {
            "environment": {id : state_traj['environment'] for id,state_traj in enumerate(self.state_trajectory[-1][::self.config['simulation_metadata']['num_substeps_per_step']])},
            "agents": {id : state_traj['agents'] for id,state_traj in enumerate(self.state_trajectory[-1][::self.config['simulation_metadata']['num_substeps_per_step']])}
            }
            self.state_data_dict[episode] = current_episode_state_data_dict
            self.reset()
            #self.controller.learn_after_episode(jax.tree_map(lambda x: x[-1], self.trajectory), self.initializer, self.optimizer)
    def get_runner(config, registry):
        return Runner(config, registry)
    
    def execute(self):
        self.forward()
        with open('state_data_dict.pkl', 'wb') as f:
            pickle.dump(self.state_data_dict, f)
            
