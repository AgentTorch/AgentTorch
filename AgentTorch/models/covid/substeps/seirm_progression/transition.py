import torch
from torch import distributions, nn
import torch.nn.functional as F
import re

from AgentTorch.substep import SubstepTransition

class SEIRMProgression(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
        self.device = self.config['simulation_metadata']['device']
        
        self.SUSCEPTIBLE_VAR = self.config['simulation_metadata']['SUSCEPTIBLE_VAR']
        self.EXPOSED_VAR = self.config['simulation_metadata']['EXPOSED_VAR']
        self.INFECTED_VAR = self.config['simulation_metadata']['INFECTED_VAR']
        self.RECOVERED_VAR = self.config['simulation_metadata']['RECOVERED_VAR']
        self.MORTALITY_VAR = self.config['simulation_metadata']['MORTALITY_VAR']
        
        self.INFINITY_TIME = self.config['simulation_metadata']['num_steps_per_episode'] + 20
    
    def update_stages(self, t, current_stages, current_transition_times):                        
        transition_to_infected = self.INFECTED_VAR*(current_transition_times <= t) + self.EXPOSED_VAR*(current_transition_times > t)
        transition_to_mortality_or_recovered = self.RECOVERED_VAR*(current_transition_times <= t) + self.INFECTED_VAR*(current_transition_times > t)
        
        next_stage = (current_stages == self.SUSCEPTIBLE_VAR)*self.SUSCEPTIBLE_VAR + (current_stages == self.RECOVERED_VAR)*self.RECOVERED_VAR + (current_stages == self.MORTALITY_VAR)*self.MORTALITY_VAR + (current_stages == self.EXPOSED_VAR)*transition_to_infected + (current_stages == self.INFECTED_VAR)*transition_to_mortality_or_recovered
        
        return next_stage
        
    def update_times(self, t, next_stages, current_transition_times, exposed_to_infected_time, infected_to_recovered_time):   
        new_transition_times = torch.clone(current_transition_times)
        stages = torch.clone(next_stages).long()
        
        new_transition_times[(stages==self.INFECTED_VAR)*(current_transition_times == t)] = self.INFINITY_TIME
        new_transition_times[(stages==self.EXPOSED_VAR)*(current_transition_times == t)] = t + exposed_to_infected_time
        
        return new_transition_times
        
    def forward(self, state, action=None):
        '''Update stage and transition times for already infected agents'''
        input_variables = self.input_variables
        t = state['current_step']
        
        current_stages = get_by_path(state, re.split("/", input_variables['disease_stage']))
        current_transition_times = get_by_path(state, re.split("/", input_variables['next_stage_time']))
        exposed_to_infected_time = get_by_path(state, re.split("/", input_variables['exposed_to_infected_time']))
        infected_to_recovered_time = get_by_path(state, re.split("/", input_variables['infected_to_recovered_time']))

        new_stages = self.update_stages(t, current_stages, current_transition_times)
        new_transition_times = self.update_times(t, new_stages, current_transition_times, exposed_to_infected_time, infected_to_recovered_time)
        
        return {self.output_variables[0]: new_stages, self.output_variables[1]: new_transition_times}