import torch
import torch.nn as nn
import numpy as np
import re
from AgentTorch.substep import SubstepAction
from AgentTorch.helpers import discrete_sample, get_by_path, logical_and, logical_or, logical_not
import pdb

class AcceptTest(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.arguments = arguments
        
        self.num_agents = self.config['simulation_metadata']['num_agents']
        self.SUSCEPTIBLE_VAR = self.config['simulation_metadata']['SUSCEPTIBLE_VAR']
        self.RECOVERED_VAR = self.config['simulation_metadata']['RECOVERED_VAR']
        self.device = torch.device(self.config['simulation_metadata']['device'])
        
    def forward(self, state, observation):
        print("Executing Substep Policy: Accept Test!")        
        agent_is_quarantined = get_by_path(state, re.split("/", self.input_variables["is_quarantined"]))
        agent_disease_stage = get_by_path(state, re.split("/", self.input_variables["disease_stage"]))
        test_compliance_prob = get_by_path(state, re.split("/", self.input_variables["test_compliance_prob"]))
                
        not_susceptible = (agent_disease_stage > self.SUSCEPTIBLE_VAR).long()
        not_recovered = (agent_disease_stage < self.RECOVERED_VAR).long()
        
        exposed_infected = logical_or(not_susceptible, not_recovered)
        agent_is_eligible = logical_and(exposed_infected, logical_not(agent_is_quarantined).long())
        
        agent_test_compliance = discrete_sample(sample_prob=test_compliance_prob, size=(self.num_agents,), device=self.device).unsqueeze(1)
        agent_test_action = logical_and(agent_is_eligible, agent_test_compliance)
        
        return {self.output_variables[0]: agent_test_action}
    
