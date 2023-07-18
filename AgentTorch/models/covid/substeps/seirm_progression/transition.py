import torch
from torch import distributions, nn
import torch.nn.functional as F
import re

from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path

class SEIRMProgression(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print("To fix this class!!!!!")
            
        self.device = self.config['simulation_metadata']['device']
        
        self.SUSCEPTIBLE_VAR = self.config['simulation_metadata']['SUSCEPTIBLE_VAR']
        self.EXPOSED_VAR = self.config['simulation_metadata']['EXPOSED_VAR']
        self.INFECTED_VAR = self.config['simulation_metadata']['INFECTED_VAR']
        self.RECOVERED_VAR = self.config['simulation_metadata']['RECOVERED_VAR']
        self.MORTALITY_VAR = self.config['simulation_metadata']['MORTALITY_VAR']

        self.STAGE_SAME_VAR = 0
        self.STAGE_UPDATE_VAR = 1
        
        self.INFINITY_TIME = self.config['simulation_metadata']['num_steps_per_episode'] + 20

    def update_current_stages(self, t, current_stages, current_transition_times):
        transit_agents = (current_transition_times > t)*self.STAGE_SAME_VAR + (current_transition_times<= t)*self.STAGE_UPDATE_VAR

        stage_transition = (current_stages == self.EXPOSED_VAR)*transit_agents + (current_stages == self.INFECTED_VAR)*transit_agents

        new_stages = current_stages + stage_transition

        return new_stages
    
    def update_times(self, t, next_stages, current_transition_times, infected_to_recovered_time):
        stages = torch.clone(next_stages).long()
        time_transition = (stages==self.INFECTED_VAR)*(current_transition_times == t)*self.INFINITY_TIME + (stages == self.EXPOSED_VAR)*(current_transition_times == t)*(infected_to_recovered_time)

        new_transition_times = current_transition_times + time_transition

        return new_transition_times

    def update_times_legacy(self, t, next_stages, current_transition_times, exposed_to_infected_time, infected_to_recovered_time):   
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
        infected_to_recovered_time = get_by_path(state, re.split("/", input_variables['infected_to_recovered_time']))

        new_stages = self.update_current_stages(t, current_stages, current_transition_times)
        new_transition_times = self.update_times(t, new_stages, current_transition_times, infected_to_recovered_time)
        
        return {self.output_variables[0]: new_stages, self.output_variables[1]: new_transition_times}