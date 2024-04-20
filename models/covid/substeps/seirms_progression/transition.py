import torch
import torch.nn.functional as F
import re

from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
from AgentTorch.helpers.distributions import StraightThroughBernoulli

class SEIRMSProgression(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = self.config['simulation_metadata']['device']
        self.num_timesteps = self.config['simulation_metadata']['num_steps_per_episode']
        
        self.SUSCEPTIBLE_VAR = self.config['simulation_metadata']['SUSCEPTIBLE_VAR']
        self.EXPOSED_VAR = self.config['simulation_metadata']['EXPOSED_VAR']
        self.INFECTED_VAR = self.config['simulation_metadata']['INFECTED_VAR']
        self.RECOVERED_VAR = self.config['simulation_metadata']['RECOVERED_VAR']
        self.MORTALITY_VAR = self.config['simulation_metadata']['MORTALITY_VAR']
        self.RESET_RECOVERED_VAR = -1*self.RECOVERED_VAR

        self.STAGE_SAME_VAR = 0
        self.STAGE_UPDATE_VAR = 1

        self.INFINITY_TIME = self.config['simulation_metadata']['num_steps_per_episode'] + 20
        self.INFECTED_TO_RECOVERED_TIME = self.config['simulation_metadata']['INFECTED_TO_RECOVERED_TIME']
        self.RECOVERED_TO_SUSCEPTIBLE_TIME = self.config['simulation_metadata']['RECOVERED_TO_SUSCEPTIBLE_TIME'] # agents become eligible for re-infection

        self.external_M = torch.tensor(self.learnable_args['M'], requires_grad=True) # we use this if calibration is external
        self.st_bernoulli = StraightThroughBernoulli.apply

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        timestep_tensor = torch.tensor([timestep])
        one_hot_tensor = F.one_hot(timestep_tensor, num_classes=num_timesteps)

        return one_hot_tensor

    def update_daily_deaths(self, t, daily_death_count, current_stages, current_transition_times):
        # recovered or dead agents
        recovered_or_dead_agents = (current_stages == self.INFECTED_VAR)*(current_transition_times <= t)
        
        new_death_recovered_today = current_stages*recovered_or_dead_agents / self.INFECTED_VAR
        num_dead_today = new_death_recovered_today.sum()*self.external_M

        daily_death_count = daily_death_count + self._generate_one_hot_tensor(t, self.num_timesteps)*num_dead_today

        agent_death_prob = self.external_M*recovered_or_dead_agents # probability agent dies for all num_agents. It is only non-zero for recovered_or_dead_mask        
        dead_agents = self.st_bernoulli(agent_death_prob)
        recovered_agents = new_death_recovered_today - dead_agents

        return daily_death_count, dead_agents, recovered_agents
    
    def update_current_stages(self, t, current_stages, current_transition_times, newly_recovered_agents, newly_dead_agents):
        '''exposed move to infected; infected move to recovered or die; recovered move to susceptible'''

        transit_to_infected = (current_stages == self.EXPOSED_VAR)*(current_transition_times <=t)*self.STAGE_UPDATE_VAR
        transit_to_susceptible = (current_stages == self.RECOVERED_VAR)*(current_transition_times <=t)*self.RESET_RECOVERED_VAR

        transit_to_dead = newly_dead_agents*(self.STAGE_UPDATE_VAR*2) # increase by two levels
        transit_to_recovered =  newly_recovered_agents*self.STAGE_UPDATE_VAR

        stage_transition = (current_stages/self.EXPOSED_VAR)*transit_to_infected + (current_stages/self.INFECTED_VAR)*transit_to_dead + (current_stages/self.INFECTED_VAR)*transit_to_recovered + (current_stages/self.RECOVERED_VAR)*transit_to_susceptible
        new_stages = current_stages + stage_transition

        return new_stages

    # def update_current_stages(self, t, current_stages, current_transition_times):
    #     '''infected agents can recover or die; recovered agents become susceptible again'''       
    #     transit_agents = (current_transition_times <= t)*self.STAGE_UPDATE_VAR
    #     stage_transition = (current_stages == self.EXPOSED_VAR)*transit_agents + (current_stages == self.INFECTED_VAR)*transit_agents
        
    #     new_stages = current_stages + stage_transition
    #     return new_stages
    
    def update_next_transition_times(self, t, current_stages, current_transition_times, newly_recovered_agents, newly_dead_agents):
        new_transition_times = torch.clone(current_transition_times).to(current_transition_times.device)
        curr_stages = torch.clone(current_stages).to(current_stages.device)
        
        # exposed agents get infected. next time their state will change for recovery
        new_transition_times[(curr_stages==self.EXPOSED_VAR)*(current_transition_times == t)] = (t + self.INFECTED_TO_RECOVERED_TIME)

        # dead agents go out of simulation
        new_transition_times[newly_dead_agents.long()] = self.INFINITY_TIME

        # recovered agents will change back to susceptible in future
        new_transition_times[newly_recovered_agents.long()] = self.RECOVERED_TO_SUSCEPTIBLE_TIME
        
        return new_transition_times
        
#         time_transition = torch.logical_and(current_stages==self.INFECTED_VAR, current_transition_times == t)*self.INFINITY_TIME + torch.logical_and(current_stages==self.EXPOSED_VAR, current_transition_times==t)*self.INFECTED_TO_RECOVERED_TIME
#         new_transition_times = current_transition_times + time_transition
#         return new_transition_times
            
    def forward(self, state, action):
        '''Update stage and transition times for already infected agents'''
        input_variables = self.input_variables
        t = state['current_step']
        print("Substep: SEIRM progression!")
        
        current_stages = get_by_path(state, re.split("/", input_variables['disease_stage']))
        current_transition_times = get_by_path(state, re.split("/", input_variables['next_stage_time']))
        daily_deaths = get_by_path(state, re.split("/", input_variables['daily_deaths']))

        new_daily_deaths, recovered_agents, dead_agents = self.update_daily_deaths(t, daily_deaths, current_stages, current_transition_times)

        new_stages = self.update_current_stages(t, current_stages, current_transition_times, recovered_agents, dead_agents)        
        new_transition_times = self.update_next_transition_times(t, current_stages, current_transition_times, recovered_agents, dead_agents)
        
        return {self.output_variables[0]: new_stages, self.output_variables[1]: new_transition_times, self.output_variables[2]: new_daily_deaths}
        