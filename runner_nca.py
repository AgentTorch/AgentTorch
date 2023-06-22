from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
import argparse
from utils.general import *

from initializer_final import Initializer
from registry import Registry
from controller import Controller
from utils.initialization.nca import nca_initialize_state

print("Imports completed..")

def create_registry():
    reg = Registry()
    # transition
    from utils.transition.nca import NCAEvolve
    reg.register(NCAEvolve, "NCAEvolve", key="transition")
    # policy
    # observation
    # initialization and network
    from utils.initialization.nca import nca_initialize_state
    from utils.initialization.opinion import grid_network
    reg.register(nca_initialize_state, "nca_initialize_state", key="initialization")
    
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
        self.state_trajectory = []
                
    def execute(self):
        # run for one episode
        self.state_trajectory.append([self.state])
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
                self.state_trajectory[-1].append(self.state)
            
    def forward(self):
        return self.execute()
    
    def reset(self):
        shape = [5184, 16]
        params = {'n_channels': torch.tensor([16.]), 'batch_size': torch.tensor([8.]), 'device': 'cpu'}
        x0 = nca_initialize_state(shape, params)
        self.state = self.initializer.state
        self.state['agents']['automata']['cell_state'] = x0
    
