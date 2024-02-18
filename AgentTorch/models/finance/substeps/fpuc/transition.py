'''
Federal Pandemic Unemployment Compensation
    - $600/week from 4/5/2020 to 7/26/2020
    - $300/week from 1/3/2021 to 3/14/2021
'''

import torch
import torch.nn as nn
import re

from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path

class FPUC(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = self.config['simulation_metadata']['device']
        self.UNEMPLOYED_FLAG = self.config['simulation_metadata']['EMPLOYMENT_FLAG_UNEMPLOYED']

    def update_agent_assets(self, t, current_assets, eligible_agents, compensation_for_eligible):
        '''revisit: can we store even granular metadata'''
        new_assets = current_assets + eligible_agents*compensation_for_eligible[t].to_dense()

        return new_assets

    def forward(self, state, action):
        t = state['current_step']
        input_variables = self.input_variables

        current_assets = get_by_path(state, re.split("/", input_variables['agent_assets']))

        agents_occupation_status = get_by_path(state, re.split("/", input_variables['occupation_status']))
        eligible_agents = (agents_occupation_status == self.UNEMPLOYED_FLAG)
        
        compensation_for_eligible = get_by_path(state, re.split("/", input_variables['fpuc_payments']))

        new_agent_assets = self.update_agent_assets(t, current_assets, eligible_agents, compensation_for_eligible)

        return {self.output_variables[0]: new_agent_assets}