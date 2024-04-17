import torch
import torch.nn.functional as F
import re

from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path

class SEIRMProgression(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = torch.device(self.config['simulation_metadata']['device'])
        self.num_timesteps = self.config['simulation_metadata']['num_steps_per_episode']
        
        self.SUSCEPTIBLE_VAR = self.config['simulation_metadata']['SUSCEPTIBLE_VAR']
        self.EXPOSED_VAR = self.config['simulation_metadata']['EXPOSED_VAR']
        self.INFECTED_VAR = self.config['simulation_metadata']['INFECTED_VAR']
        self.RECOVERED_VAR = self.config['simulation_metadata']['RECOVERED_VAR']
        self.MORTALITY_VAR = self.config['simulation_metadata']['MORTALITY_VAR']
        self.RESET_VAR = -1*self.RECOVERED_VAR

        self.STAGE_SAME_VAR = 0
        self.STAGE_UPDATE_VAR = 1

        self.external_M = torch.tensor(self.learnable_args['M'], requires_grad=True) # we use this if calibration is external
        
        self.INFINITY_TIME = self.config['simulation_metadata']['num_steps_per_episode'] + 20
        self.INFECTED_TO_RECOVERED_TIME = self.config['simulation_metadata']['INFECTED_TO_RECOVERED_TIME']

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        timestep_tensor = torch.tensor([timestep])
        one_hot_tensor = F.one_hot(timestep_tensor, num_classes=num_timesteps)

        return one_hot_tensor.to(self.device)

    def update_daily_deaths(self, t, daily_dead, current_stages, current_transition_times):
        # recovered or dead agents
        recovered_and_dead_mask = (current_stages == self.INFECTED_VAR)*(current_transition_times <= t)
        new_death_recovered_today = current_stages*recovered_and_dead_mask / self.INFECTED_VAR

        num_dead_today = new_death_recovered_today.sum()*self.external_M

        daily_dead = daily_dead + self._generate_one_hot_tensor(t, self.num_timesteps)*num_dead_today
        return daily_dead

    def update_current_stages(self, t, current_stages, current_transition_times):        
        transit_agents = (current_transition_times <= t)*self.STAGE_UPDATE_VAR
        stage_transition = (current_stages == self.EXPOSED_VAR)*transit_agents + (current_stages == self.INFECTED_VAR)*transit_agents
        
        new_stages = current_stages + stage_transition
        return new_stages
    
    def update_next_transition_times(self, t, current_stages, current_transition_times):
        new_transition_times = torch.clone(current_transition_times).to(current_transition_times.device)
        curr_stages = torch.clone(current_stages).to(current_stages.device)
                
        new_transition_times[(curr_stages==self.INFECTED_VAR)*(current_transition_times == t)] = self.INFINITY_TIME
        new_transition_times[(curr_stages==self.EXPOSED_VAR)*(current_transition_times == t)] = (t + self.INFECTED_TO_RECOVERED_TIME)
        
        return new_transition_times
        
#         time_transition = torch.logical_and(current_stages==self.INFECTED_VAR, current_transition_times == t)*self.INFINITY_TIME + torch.logical_and(current_stages==self.EXPOSED_VAR, current_transition_times==t)*self.INFECTED_TO_RECOVERED_TIME
#         new_transition_times = current_transition_times + time_transition
#         return new_transition_times
            
    def forward(self, state, action):
        '''Update stage and transition times for already infected agents'''
        input_variables = self.input_variables
        t = state['current_step']
        # print("Substep: SEIRM progression!")
        
        current_stages = get_by_path(state, re.split("/", input_variables['disease_stage']))
        current_transition_times = get_by_path(state, re.split("/", input_variables['next_stage_time']))
        daily_deaths = get_by_path(state, re.split("/", input_variables['daily_deaths']))

        new_stages = self.update_current_stages(t, current_stages, current_transition_times)        
        new_transition_times = self.update_next_transition_times(t, current_stages, current_transition_times)

        new_daily_deaths = self.update_daily_deaths(t, daily_deaths, current_stages, current_transition_times)
        
        return {self.output_variables[0]: new_stages, self.output_variables[1]: new_transition_times, self.output_variables[2]: new_daily_deaths}
        