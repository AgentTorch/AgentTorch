import torch
import torch.nn as nn
from torch.distributions import Normal
import sys
sys.path.append("/Users/shashankkumar/Documents/GitHub/MacroEcon/AgentTorch")
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
from torch.nn import functional as F
import re

class UpdateMacroRates(SubstepTransition):
    '''Macro quantities relevant to labor markets - hourly wage, unemployment rate, price of goods'''
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
            
        self.device = torch.device(self.config['simulation_metadata']['device'])
        self.num_timesteps = self.config['simulation_metadata']['num_steps_per_episode']
        self.max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_wage']
        self.num_agents = self.config['simulation_metadata']['num_agents']

        self.external_UAC = torch.tensor(self.learnable_args['unemployment_adaptation_coefficient'], requires_grad=True)
    
    def calculateNumberOfAgentsNotWorking(self, working_status):
        agents_not_working = torch.sum(torch.sum((1-working_status),dim=1),dim=0)
        return agents_not_working
    
    def updateHourlyWage(self, hourly_wage, imbalance):
        omega = imbalance.float()
        r1, r2 = self.max_rate_change*omega, 0

        sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2

        new_hourly_wage = hourly_wage + hourly_wage*sampled_omega
        return new_hourly_wage

    def legacy_calculateHourlyWage(self, hourly_wage, imbalance):
        # Calculate hourly wage
        w = hourly_wage
        omega = imbalance.float()
        
        if omega > 0:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_wage']
            r2 = (max_rate_change * omega)
            r1 = 0
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            new_hourly_wage = w * (1 + sampled_omega)
        else:
            max_rate_change = self.config['simulation_metadata']['maximum_rate_of_change_of_wage']
            r1 = (max_rate_change * omega)
            r2 = 0
            # sampled_omega = torch.FloatTensor(1,1).uniform_((max_rate_change * omega),0)
            sampled_omega = (r1 - r2) * torch.rand(1, 1) + r2
            # sampled_omega = torch.tensor.uniform((max_rate_change * omega),0)
            new_hourly_wage = w * (1 + sampled_omega)
        return new_hourly_wage
    
    def calculateUnemploymentRate(self, Labor_Force_Participation_Rate,Gender_Ratio,Imbalance):
        coeff_0,coeff_1,coeff_2,coeff_3,error_coeff = self.args['coeff_0'],self.args['coeff_1'], self.args['coeff_2'], self.args['coeff_3'], self.args['error_coeff']
        unemployment_rate = coeff_0 + (coeff_1 * Imbalance + coeff_2 * Labor_Force_Participation_Rate + coeff_3 * Gender_Ratio) + error_coeff
        return unemployment_rate
    
    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        timestep_tensor = torch.tensor([timestep])
        one_hot_tensor = F.one_hot(timestep_tensor, num_classes=num_timesteps)

        return one_hot_tensor.to(self.device)

    def forward(self, state, action):
        print("Executing Substep: Labor Market")
        month_id = state['current_step']
        t = int(month_id)
        time_step_one_hot = self._generate_one_hot_tensor(t, self.num_timesteps)
        
        working_status = get_by_path(state, re.split("/", self.input_variables['will_work']))
        imbalance = get_by_path(state, re.split("/", self.input_variables['imbalance']))
        hourly_wage = get_by_path(state, re.split("/", self.input_variables['hourly_wage']))
        unemployment_rate = get_by_path(state, re.split("/", self.input_variables['unemployment_rate']))

        unemployment_adaptation_coefficient = (self.external_UAC*time_step_one_hot).sum()

        # unemployment rate
        agents_not_working = self.calculateNumberOfAgentsNotWorking(working_status)
        time_step_one_hot = self._generate_one_hot_tensor(t, self.num_timesteps)
        
        unemployed_agents = agents_not_working * unemployment_adaptation_coefficient
        current_unemployment_rate = unemployed_agents / self.num_agents
        unemployment_rate = unemployment_rate + (current_unemployment_rate*time_step_one_hot)

        # hourly wages
        new_hourly_wages = self.updateHourlyWage(hourly_wage, imbalance) # self.calculateHourlyWage() to revert
                
        return {self.output_variables[0]: new_hourly_wages, 
                self.output_variables[1]: unemployment_rate}
