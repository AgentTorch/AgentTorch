import torch
import torch.nn as nn
import re

from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path

class TransferStimulus(SubstepTransition):
    '''Same stimulus payments received to all eligible citizen, at specific time intervals'''
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = self.config['simulation_metadata']['device']

    def update_agent_assets(self, t, current_assets, stimulus_payments, agent_eligibility):
        new_assets = current_assets + agent_eligibility*stimulus_payments[t].to_dense()

        return new_assets
        
    def forward(self, state, action):
        t = state['current_step']
        input_variables = self.input_variables

        current_assets = get_by_path(state, re.split("/", input_variables['agent_assets']))
        stimulus_payments = get_by_path(state, re.split("/", input_variables['stimulus_amounts']))
        agent_eligibility = get_by_path(state, re.split("/", input_variables['stimulus_eligibility']))

        new_agent_assets = self.update_agent_assets(t, current_assets, stimulus_payments, agent_eligibility)

        return {self.output_variables[0]: new_agent_assets}