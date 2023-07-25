import torch
import torch.nn as nn
import re

from AgentTorch.substep import SubstepTransition
from AgentTorch.helpers import get_by_path, discrete_sample, logical_and, logical_not, logical_or

class COVIDTesting(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = torch.device(self.config['simulation_metadata']['device'])
        self.SUSCEPTIBLE_VAR = self.config['simulation_metadata']['SUSCEPTIBLE_VAR']
        self.RECOVERED_VAR = self.config['simulation_metadata']['RECOVERED_VAR']
        self.EXPOSED_VAR = self.config['simulation_metadata']['EXPOSED_VAR']

        self.num_agents = self.config['simulation_metadata']['num_citizens']

        self.AWAITING_RESULT_VAR = 1
        self.GOT_RESULT_VAR = -1
        self.INFINITY_TIME = 2*self.config['simulation_metadata']['num_steps']

    def get_test_result(self, t, agents_awaiting_results, agent_result_date, current_stages, test_re_eligble_date):
        '''Agents receive test result'''
        true_positive_prob = self.args['true_positive_prob']
        false_positive_prob = self.args['false_positive_prob']
        test_inelgible_days = self.args['test_inelgible_days']

        agents_result_expected_today = (agent_result_date == t).long()

        # 1: reset agents_awaiting_test_result
        agents_awaiting_results = agents_awaiting_results + agents_result_expected_today*self.GOT_RESULT_VAR

        # 2: get true_positive and false_positive results - check candidates + sample based on TPR and FPR
        exposed_infected_agents = logical_and((current_stages > self.SUSCEPTIBLE_VAR).long(), (current_stages < self.RECOVERED_VAR).long())

        true_positive_result_candidates = logical_and(exposed_infected_agents, agents_result_expected_today)
        false_positive_result_candidates = logical_and(logical_not(exposed_infected_agents), agents_result_expected_today)

        true_positive_mask = discrete_sample(sample_prob=true_positive_prob, size=(self.num_agents,), device=self.device)
        true_positive_results = logical_and(true_positive_result_candidates, true_positive_mask)

        false_positive_mask = discrete_sample(sample_prob=false_positive_prob, size=(self.num_agents,), device=self.device)
        false_positive_results = logical_and(false_positive_result_candidates, false_positive_mask)

        positive_results = logical_or(true_positive_results, false_positive_results)

        # 3: agents are in-eligible to test again for the next few days
        test_re_eligble_date[agents_result_expected_today.bool()] = t + test_inelgible_days # not a differentiable op

        return positive_results, agents_awaiting_results, test_re_eligble_date
    
    def get_tested(self, t, current_stages, is_quarantined, agents_awaiting_results, agents_result_date, test_ineligible):
        '''Eligible Agent get themselves tested and receive test result date'''
        eligiblity_compliance_prob = self.args['test_eligibility_compliance_prob']
        result_delay_days = self.args['test_result_delay']

        not_susceptible = (current_stages > self.SUSCEPTIBLE_VAR)
        not_recovered = (current_stages < self.RECOVERED_VAR)
        exposed_infected_agents = logical_and(not_susceptible, not_recovered)

        # 1: get eligible agents -> exposed_infected + not previously ineligible + not quarantined + not awaiting test result
        test_eligible = logical_and(exposed_infected_agents, logical_not(test_ineligible))
        test_eligible = logical_and(test_eligible, logical_not(is_quarantined))
        test_eligible = logical_and(test_eligible, logical_not(agents_awaiting_results))

        # 2. enrol eligible agents based on compliance
        eligibility_comply_agents = discrete_sample(sample_prob=eligiblity_compliance_prob, size=(self.num_agents,), device=self.device)
        test_enrolled_agents = logical_and(eligibility_comply_agents, test_eligible)

        # 3: set awaiting_test_result flag and obtain test result date
        agents_awaiting_results = agents_awaiting_results + test_enrolled_agents*self.AWAITING_RESULT_VAR
        agents_result_date[test_enrolled_agents.bool()] = t + result_delay_days # not a differentiable op

        return agents_awaiting_results, agents_result_date

    def forward(self, state, action=None):
        t = state['current_step']
        input_variables = self.input_variables

        current_stages = get_by_path(state, re.split('/', input_variables['current_stages']))
        is_quarantined = get_by_path(state, re.split('/', input_variables['is_quarantined']))

        agent_result_date = get_by_path(state, re.split('/', input_variables['agents_result_dates']))
        agents_awaiting_results = get_by_path(state, re.split('/', input_variables['agents_awaiting_test_result']))
        is_quarantine_eligible = get_by_path(state, re.split('/', input_variables['is_quarantine_eligible']))
        test_re_eligble_date = get_by_path(state, re.split('/', input_variables['test_re_eligble_date']))

        # step 1: agents receive test result and may test positive
        positive_results, agents_awaiting_results, test_re_eligble_date = self.get_test_result(self, t, agents_awaiting_results, agent_result_date, current_stages, test_re_eligble_date)
        is_quarantine_eligible = logical_or(is_quarantine_eligible, positive_results)

        # step 2: agents take test and join result queue
        test_ineligible = (t < test_re_eligble_date).long()

        agents_awaiting_results, agents_result_date = self.get_tested(t, current_stages, is_quarantined, agents_awaiting_results, agents_result_date, test_ineligible)

        return {self.output_variables[0]: agents_awaiting_results, self.output_variables[1]: agents_result_date, self.output_variables[2]: is_quarantine_eligible, self.output_variables[3]: test_re_eligble_date}