from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
import argparse

from initializer_final import Initializer
from registry import Registry
from controller import Controller

from utils.general import *

# *************************************************************************
# Parsing command line arguments
parser = argparse.ArgumentParser(
    description="TorchABM - framework for scalable, differentiable agent based models."
)
parser.add_argument(
    "-c", "--config", help="Name of the yaml config file with the parameters."
)
# *************************************************************************

print("Imports completed..")

def create_registry():
    
    reg = Registry()
    
    # transition
    from utils.transition.opinion import NewQExp, NewPurchasedBefore
    reg.register(NewQExp, "new_Q_exp", key="transition")
    reg.register(NewPurchasedBefore, "new_purchased_before", key="transition")
        
    # policy
    from utils.policy.opinion import PurchaseProduct
    reg.register(PurchaseProduct, "purchase_product", key="policy")
    
    # observation
    from utils.observation.opinion import GetFromState, GetNeighborsSum, GetNeighborsSumReduced
    reg.register(GetFromState, "get_from_state", key="observation")
    reg.register(GetNeighborsSum, "get_neighbors_sum", key="observation")
    reg.register(GetNeighborsSumReduced, "get_neighbors_sum_reduced", key="observation")
    
    # initialization and network
    from utils.initialization.opinion import zeros, random_normal, constant, random_normal_col_by_col, grid_network
    reg.register(zeros, "zeros", key="initialization")
    reg.register(random_normal, "random_normal", key="initialization")
    reg.register(constant, "constant", key="initialization")
    reg.register(random_normal_col_by_col, "random_normal_col_by_col", key="initialization")
        
    reg.register(grid_network, "grid", key="network")
    
    return reg
    
class Runner(nn.Module):
    def __init__(self, config_file, initial_state=None):
        super().__init__()
        
        self.config_file = config_file
        
        self.config = read_config(self.config_file)
        self.registry = create_registry()
                
        assert self.config["simulation_metadata"]["num_substeps_per_step"] == len(list(self.config['substeps'].keys()))
        
        self.initializer = Initializer(self.registry, self.config)
        self.state = self.initializer.state
        
        self.controller = Controller(self.config)
                
        self.trajectory = { 'states': deque(), 'observations': deque(), 'actions': deque(),'rewards': deque() }
                
    def execute(self):
        # run for all episodes
        for episode in range(self.config['simulation_metadata']['num_episodes']):
            for traj_var in self.trajectory.keys():
                self.trajectory[traj_var].append(deque())

            for step in range(self.config["simulation_metadata"]["num_steps_per_episode"]):
                self.state['current_step'] = step
                for traj_var in self.trajectory.keys():
                    self.trajectory[traj_var][-1].append(deque())
                
                for substep in self.config['substeps'].keys():

                    self.trajectory["states"][-1][-1].append(self.state)
                    observation_profile, action_profile = {}, {}

                    for agent_type in self.config['substeps'][substep]['active_agents']:
                        assert substep == self.state['current_substep']
                        assert step == self.state['current_step']
                        
                        observation_profile[agent_type] = self.controller.observe(self.state, self.initializer.observation_function, agent_type)
                        action_profile[agent_type] = self.controller.act(self.state, observation_profile[agent_type], self.initializer.policy_function, agent_type)
                                                
                    self.trajectory["observations"][-1][-1].append(observation_profile)
                    self.trajectory["actions"][-1][-1].append(action_profile)

                    next_state = self.controller.progress(self.state, action_profile, self.initializer.transition_function)
                    # next_state = self.controller.progress(self.config, self.state, action_profile, step, substep, tf, tf_params)
                    self.state = next_state
            
    def forward(self):
        return self.execute()
    
                    

if __name__ == '__main__':
    print("The runner file..")
    args = parser.parse_args()
    
    config_file = args.config

    # create runner object
    runner = Runner(config_file)    
    runner.execute()
    
    for name, param in runner.named_parameters(): 
        print(name, param.data) 
