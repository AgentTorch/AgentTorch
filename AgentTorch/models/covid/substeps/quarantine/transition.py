import torch
import torch.nn as nn
import re

from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import discrete_sample, logical_and, logical_not, get_by_path

class Quarantine(SubstepTransition):
    '''Logic: exposed or infected agents can start quarantine'''
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = self.config['simulation_metadata']['device']
        self.num_agents = self.config['simulation_metatdata']['num_citizens']
        self.quarantine_days = self.config['simulation_metadata']['quarantine_days']
        self.num_steps = self.config['simulation_metadata']['num_steps']

        self.SUSCEPTIBLE_VAR = self.config['simulation_metadata']['SUSCEPTIBLE_VAR']
        self.EXPOSED_VAR = self.config['simulation_metadata']['EXPOSED_VAR']
        self.INFECTED_VAR = self.config['simulation_metadata']['INFECTED_VAR']
        self.RECOVERED_VAR = self.config['simulation_metadata']['RECOVERED_VAR']

        self.END_QUARANTINE_VAR = -1
        self.START_QUARANTINE_VAR = 1
        self.BREAK_QUARANTINE_VAR = -1

    def _end_quarantine(self, t, is_quarantined, quarantine_start_date):
        agents_quarantine_end_date = quarantine_start_date + self.quarantine_days
        agent_quarantine_ends =  (t>= agents_quarantine_end_date).long()

        is_quarantined += agent_quarantine_ends*self.END_QUARANTINE_VAR
        quarantine_start_date = quarantine_start_date*(1 - agent_quarantine_ends) + (self.num_steps + 1)*agent_quarantine_ends

        return is_quarantined, quarantine_start_date

    def _start_quarantine(self, t, is_quarantined, exposed_infected_agents, quarantine_start_date, quarantine_start_prob):
        agents_quarantine_start = discrete_sample(quarantine_start_prob,size=self.num_agents, device=self.device)
        agents_quarantine_start = logical_and(agents_quarantine_start, exposed_infected_agents)
        agents_quarantine_start = logical_and(logical_not(is_quarantined), agents_quarantine_start)

        is_quarantined += agents_quarantine_start*self.START_QUARANTINE_VAR
        quarantine_start_date = quarantine_start_date*(1 - agents_quarantine_start) + (agents_quarantine_start)*t

        return is_quarantined, quarantine_start_date

    def _break_quarantine(self, t, is_quarantined, quarantine_start_date, quarantine_break_prob):
        agents_quarantine_break = discrete_sample(quarantine_break_prob, size=self.num_agents, device=self.device)
        agents_quarantine_break = logical_and(is_quarantined[t], agents_quarantine_break)

        is_quarantined += agents_quarantine_break*self.BREAK_QUARANTINE_VAR
        quarantine_start_date = quarantine_start_date*(1 - agents_quarantine_break) + (self.num_steps + 1)*agents_quarantine_break

        return is_quarantined, quarantine_start_date

    def update_quarantine_status(self, t, is_quarantined, infected_agents, quarantine_start_date, quarantine_start_prob, quarantine_break_prob):
        is_quarantined, quarantine_start_date = self._end_quarantine(t, is_quarantined, quarantine_start_date)
        is_quarantined, quarantine_start_date = self._start_quarantine(t, is_quarantined, infected_agents, quarantine_start_date, quarantine_start_prob)
        is_quarantined, quarantine_start_date = self._break_quarantine(t, is_quarantined, quarantine_start_date, quarantine_break_prob)

        return is_quarantined, quarantine_start_date
    
    def forward(self, state, action=None):
        input_variables = self.input_variables
        t = state['current_step']

        is_quarantined = get_by_path(state, re.split("/", input_variables['is_quarantined']))
        quarantine_start_date = get_by_path(state, re.split("/", input_variables['quarantine_start_date']))
        disease_stage = get_by_path(state, re.split("/", input_variables['disease_stage']))

        quarantine_start_prob = get_by_path(state, re.split("/", input_variables['quarantine_start_prob']))
        quarantine_break_prob = get_by_path(state, re.split("/", input_variables['quarantine_break_prob']))

        not_susceptible = (disease_stage > self.SUSCEPTIBLE_VAR).long()
        not_recovered = (disease_stage < self.RECOVERED_VAR).long()

        infected_agents = logical_and(not_susceptible, not_recovered)

        new_is_quarantined, new_quarantine_start_date = self.update_quarantine_status(self, t, is_quarantined, infected_agents, quarantine_start_date, quarantine_start_prob, quarantine_break_prob)

        return {self.output_variables[0]: new_is_quarantined, self.output_variables[1]: new_quarantine_start_date}