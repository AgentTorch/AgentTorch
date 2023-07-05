import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import *


class PublicHealthSafety(nn.Module):

    def __init__(self, params, device) -> None:
        super().__init__()

        self.params = params
        self.device = device

        self.is_quarantined = torch.zeros(self.params['num_steps'],
                                          self.params['num_agents']).to(
                                              self.device)
        self.quarantine_start_date = (
            self.params['num_steps'] + 1) * torch.ones(
                self.params['num_agents']).to(self.device)

        self.params['quarantine_days'] = 10  # TODO: moves to params file

        self.is_masked = torch.zeros(self.params['num_steps'],
                                     self.params['num_agents']).to(self.device)

        # self.break_quarantine_dist = torch.distributions.Categorical(probs = torch.tensor([1 - self.params['quarantine_break_prob'], self.params['quarantine_break_prob']]))
        # self.start_quarantine_dist = torch.distributions.Categorical(probs = torch.tensor([1 - self.params['quarantine_start_prob'], self.params['quarantine_start_prob']]))

    def end_quarantine(self, t, learnable_params):
        # end quarantine
        agents_quarantine_end_date = self.quarantine_start_date + self.params[
            'quarantine_days']
        agent_quarantine_ends = t >= agents_quarantine_end_date

        if agent_quarantine_ends.sum() >= 0:
            self.is_quarantined[t, agent_quarantine_ends.bool()] = 0
            self.quarantine_start_date[
                agent_quarantine_ends.bool()] = self.params['num_steps'] + 1

    def start_quarantine(self, t, learnable_params):
        # start quarantine
        # agents_quarantine_start = self.start_quarantine_dist.sample((self.params['num_agents'],))
        agents_quarantine_start = diff_sample(
            learnable_params['quarantine_start_prob'],
            size=self.params['num_agents']).to(self.device)
        agents_quarantine_start = torch.logical_and(
            torch.logical_not(self.is_quarantined[t]), agents_quarantine_start)
        if agents_quarantine_start.sum() >= 0:
            self.is_quarantined[t, agents_quarantine_start.bool()] = 1
            self.quarantine_start_date[agents_quarantine_start.bool()] = t

    def break_quarantine(self, t, learnable_params):
        agents_quarantine_break = diff_sample(
            learnable_params['quarantine_break_prob'],
            size=self.params['num_agents']).to(self.device)
        agents_quarantine_break = torch.logical_and(self.is_quarantined[t],
                                                    agents_quarantine_break)
        if agents_quarantine_break.sum() >= 0:
            self.is_quarantined[t, agents_quarantine_break.bool()] = 0
            self.quarantine_start_date[
                agents_quarantine_break.bool()] = self.params['num_steps'] + 1

    def step(self, t, learnable_params):
        self.end_quarantine(t, learnable_params)
        self.start_quarantine(t, learnable_params)
        self.break_quarantine(t, learnable_params)

    def forward(self, t, learnable_params):
        self.step(t, learnable_params)