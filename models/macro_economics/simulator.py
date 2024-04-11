import pandas as pd
import torch
import torch.optim as optim

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
        data_path = '/u/ayushc/projects/GradABM/MacroEcon/simulator_data/NYC/brooklyn_unemp.csv'
        df = pd.read_csv(data_path)
        df.sort_values(by=['year','month'],ascending=True,inplace=True)
        arr = df['unemployment_rate'].values
        tensor = torch.from_numpy(arr)
        self.unemployment_test_dataset = tensor.view(5,-1)
        self.mse_loss = torch.nn.MSELoss()
        
    def forward(self):
        self.optimizer = optim.Adam(self.parameters(), 
                lr=self.config['simulation_metadata']['learning_params']['lr'], 
                betas=self.config['simulation_metadata']['learning_params']['betas'])
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 
                self.config['simulation_metadata']['learning_params']['lr_gamma'])

        for episode in range(self.config['simulation_metadata']['num_episodes']):
            print("episode: ", episode)
            num_steps_per_episode = self.config["simulation_metadata"]["num_steps_per_episode"]
            self.step(num_steps_per_episode)

            breakpoint()

            unemployment_rate_list = [state['environment']['U'] for state in self.state_trajectory[-1] if state['current_substep'] == str(self.config['simulation_metadata']['num_substeps_per_step'] - 1)]
            unemployment_rate_tensor = torch.tensor(unemployment_rate_list,requires_grad=True).float()
            test_set_for_episode = self.unemployment_test_dataset[episode][:num_steps_per_episode].float()
            loss =  self.mse_loss(unemployment_rate_tensor, test_set_for_episode)
            loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            self.reset()

            #self.controller.learn_after_episode(jax.tree_map(lambda x: x[-1], self.trajectory), self.initializer, self.optimizer)

    def execute(self):
        self.forward()
