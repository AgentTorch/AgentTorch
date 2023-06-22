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
    description="AgentTorch - framework for scalable, differentiable agent based models."
)
parser.add_argument(
    "-c", "--config", help="Name of the yaml config file with the parameters."
)
# *************************************************************************

print("Imports completed..")

def create_registry():
    
    reg = Registry()
    
    # transition
    from utils.transition.covid import NewTransmission, SEIRMProgression
    reg.register(NewTransmission, "new_transmission", key="transition")
    reg.register(SEIRMProgression, "seirm_progression", key="transition")
    
    # policy
    # observation
    
    # initialization and network
    from utils.initialization.covid import network_from_file, read_from_file, get_lam_gamma_integrals, get_mean_agent_interactions, get_infected_time, get_next_stage_time
    reg.register(network_from_file, "network_from_file", key="network")
    reg.register(read_from_file, "read_from_file", key="initialization")
    reg.register(get_lam_gamma_integrals, "get_lam_gamma_integrals", key="initialization")
    reg.register(get_mean_agent_interactions, "get_mean_agent_interactions", key="initialization")
    reg.register(get_infected_time, "get_infected_time", key="initialization")
    reg.register(get_next_stage_time, "get_next_stage_time", key="initialization")
    
    return reg
    
class Runner(nn.Module):
    def __init__(self, config_file):
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

    import ipdb; ipdb.set_trace()    
