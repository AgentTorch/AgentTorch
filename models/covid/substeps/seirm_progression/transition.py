import torch
from torch import distributions, nn
import torch.nn.functional as F
import re

from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path, logical_and, logical_or

class SEIRMProgression(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = self.config['simulation_metadata']['device']
        
        self.SUSCEPTIBLE_VAR = self.config['simulation_metadata']['SUSCEPTIBLE_VAR']
        self.EXPOSED_VAR = self.config['simulation_metadata']['EXPOSED_VAR']
        self.INFECTED_VAR = self.config['simulation_metadata']['INFECTED_VAR']
        self.RECOVERED_VAR = self.config['simulation_metadata']['RECOVERED_VAR']
        self.MORTALITY_VAR = self.config['simulation_metadata']['MORTALITY_VAR']

        self.STAGE_SAME_VAR = 0
        self.STAGE_UPDATE_VAR = 1
        
        self.INFINITY_TIME = self.config['simulation_metadata']['num_steps_per_episode'] + 20
        self.INFECTED_TO_RECOVERED_TIME = self.config['simulation_metadata']['INFECTED_TO_RECOVERED_TIME']

    def update_current_stages(self, t, current_stages, current_transition_times):
        transition_eligible_agents = (current_transition_times <= t)
        
        stage_transition = transition_eligible_agents*torch.logical_or(current_stages == self.EXPOSED_VAR, current_stages == self.INFECTED_VAR)
        
        new_stages = current_stages + stage_transition
        return new_stages
    
    def update_next_transition_times(self, t, current_stages, current_transition_times):
        time_transition = torch.logical_and(current_stages==self.INFECTED_VAR, current_transition_times == t)*self.INFINITY_TIME + torch.logical_and(current_stages==self.EXPOSED_VAR, current_transition_times==t)*self.INFECTED_TO_RECOVERED_TIME
        
        new_transition_times = current_transition_times + time_transition
        return new_transition_times
            
    def forward(self, state, action):
        '''Update stage and transition times for already infected agents'''
        input_variables = self.input_variables
        t = state['current_step']
        
        print("Substep: SEIRM progression!")
        
        current_stages = get_by_path(state, re.split("/", input_variables['disease_stage']))
        current_transition_times = get_by_path(state, re.split("/", input_variables['next_stage_time']))

        new_stages = self.update_current_stages(t, current_stages, current_transition_times)        
        new_transition_times = self.update_next_transition_times(t, new_stages, current_transition_times)
        
        return {self.output_variables[0]: new_stages, self.output_variables[1]: new_transition_times}
        