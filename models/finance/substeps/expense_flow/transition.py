import torch
from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path
import re

class RegularExpenses(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = torch.device(self.config['simulation_metadata']['device'])

    def update_assets(self, current_assets, total_earnings, total_expenses, weekly_rate):
        '''Done at household level every week'''
        new_assets = current_assets*(1 + weekly_rate) + total_earnings - total_expenses

        return new_assets

    def forward(self, state, action):
        t = state['current_step']
        input_variables = self.input_variables

        current_assets = get_by_path(state, re.split('/', input_variables['current_assets']))

        total_earnings = get_by_path(state, re.split('/', input_variables['total_earnings'])) # compute recurring income flow: sum all individual incomes
        total_expenses = get_by_path(state, re.split('/', input_variables['total_expenses']))

        weekly_rate = get_by_path(state, re.split('/', input_variables['weekly_rate']))

        current_assets = self.update_assets(current_assets, total_earnings, total_expenses, weekly_rate)

        return {self.output_variables[0]: current_assets}
