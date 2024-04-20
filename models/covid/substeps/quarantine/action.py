import torch
import torch.nn as nn
import numpy as np
import re
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import discrete_sample, get_by_path, logical_and, logical_not, logical_or
import pdb

class StartCompliance(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        
        self.output_variables = output_variables
        self.num_agents = self.config['simulation_metadata']['num_agents']
        self.device = self.config['simulation_metadata']['device']
        
        self.EXPOSED_VAR = self.config['simulation_metadata']['EXPOSED_VAR']
        self.INFECTED_VAR = self.config['simulation_metadata']['INFECTED_VAR']
        
    def forward(self, state, observation):
        quarantine_start_prob = observation['quarantine_start_prob']
        is_quarantined = observation['is_quarantined']
        disease_stage = observation['disease_stage']
                
        exposed_infected_agents = torch.logical_or(disease_stage==self.EXPOSED_VAR, disease_stage==self.INFECTED_VAR)
                
        quarantine_start_decision = discrete_sample(quarantine_start_prob, size=self.num_agents, device=self.device).unsqueeze(1)
        quarantine_start_decision = torch.logical_and(quarantine_start_decision, logical_not(is_quarantined))
        quarantine_start_decision = torch.logical_and(quarantine_start_decision, exposed_infected_agents)
        
        return {self.output_variables[0]: quarantine_start_decision}
    
class BreakCompliance(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.num_agents = self.config['simulation_metadata']['num_agents']
        self.device = torch.device(self.config['simulation_metadata']['device'])
        
    def forward(self, state, observation):
        quarantine_break_prob = observation['quarantine_break_prob']
        is_quarantined = observation['is_quarantined']
        
        quarantine_break_decision = discrete_sample(quarantine_break_prob, size=self.num_agents, device=self.device).unsqueeze(1)
        quarantine_break_decision = torch.logical_and(is_quarantined, quarantine_break_decision)
        
        return {self.output_variables[0]: quarantine_break_decision}
        
        