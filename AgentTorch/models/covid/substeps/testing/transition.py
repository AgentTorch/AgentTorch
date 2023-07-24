import torch
import torch.nn as nn
import re

from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path, discrete_sample, logical_and, logical_not

class COVIDTesting(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = self.config['simulation_metadata']['device']
        self.SUSCEPTIBLE_VAR = self.config['simulation_metadata']['SUSCEPTIBLE_VAR']
        self.RECOVERED_VAR = self.config['simulation_metadata']['RECOVERED_VAR']
        self.EXPOSED_VAR = self.config['simulation_metadata']['EXPOSED_VAR']

        self.AWAITING_RESULT_VAR = 1

    def get_test_result(self, t, agents_test_results, agent_result_dates, current_stages, true_positive_prob, false_positive_prob):
        '''Agents receive test result'''
        print("To check differentiability...")

        agents_result_expected_today = (agent_result_dates == t).long()
        agent_result_dates[agents_result_expected_today] = t # result date for current agents
        agents_awaiting_results[agents_result_expected_today] = 0 # these agents are no longer waiting for result

        not_susceptible = (current_stages > self.SUSCEPTIBLE_VAR)
        not_recovered = (current_stages < self.RECOVERED_VAR)
        infected_exposed_agents = logical_and(not_susceptible, not_recovered)

        positive_result_candidates = logical_and(infected_exposed_agents, agents_result_expected_today)
        negative_result_candidates = logical_not(positive_result_candidates)

        positive_test_result = discrete_sample(sample=true_positive_prob, size=positive_result_candidates.sum())
        negative_test_result = discrete_sample(sample=false_positive_prob, size=negative_result_candidates.sum())
        
        agents_test_results[negative_result_candidates] = negative_test_result
        agents_test_results[positive_result_candidates] = positive_test_result

        return agents_test_results

    def get_result_date(self, t, agents_result_dates, current_stages, is_quarantined):
        '''Agents get themselves tested'''
        print("To check differentiability")

        not_susceptible = (current_stages > self.SUSCEPTIBLE_VAR)
        not_recovered = (current_stages < self.RECOVERED_VAR)
        exposed_infected_agents = logical_and(not_susceptible, not_recovered)

        test_eligible = logical_and(exposed_infected_agents, logical_not(is_quarantined))
        test_eligible = logical_and(test_eligible, logical_not(agents_awaiting_test_results))

        # update awaiting test result
        agents_awaiting_test_results = agents_awaiting_test_results + test_eligible*self.AWAITING_RESULT_VAR

        # update test result date: sample test result date and assign it to individual agents
        print("To complete this function...")

    def forward(self, state, action):
        t = state['current_step']
        input_variables = self.input_variables

