'''
- Individuals file for unemployment assistance (UI or PUA)
- Weekly benefit is setup based on agent availability and persists.
- There is admin delay in processing of PUA: can amplify economic hardships.
- Can receive upto 26 weeks.
- Minimum payment is $124 per week.
'''

import torch
import torch.nn as nn
import re

from AgentTorch.helpers import get_by_path, compare
from AgentTorch.substep import SubstepTransition

from substeps.utils import _income_to_pua, _get_pua_payments

class PandemicUnemploymentAssistance(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = self.config['simulation_metadata']['device']
        self.ENROLLMENT_VAR = 1
    
    def start_pua(self, t, pua_payments, agents_enrollment_status, requesting_pua_agents, agents_income, agents_employment):
        eligible_agents = (agents_employment == 0) # unemployed agents are eligible
        requesting_payment_amounts = _income_to_pua(agents_income)
        request_date = t
        num_steps_shape = torch.zeros((pua_payments.shape[1])) #(num_steps,)

        newly_enrolled_agents = eligible_agents*requesting_pua_agents*(1 - agents_enrollment_status) # all requesting eligible agents receive it if they don't actively do yet
        newly_enrolled_agents_indices = newly_enrolled_agents.nonzero().squeeze() # gather indices

        batched_pua_payments_func = torch.vmap(_get_pua_payments, in_dims=(0, None, None))

        # update pua_payments
        batched_pua_payments = batched_pua_payments_func(requesting_payment_amounts, num_steps_shape, request_date)

        pua_payments[newly_enrolled_agents_indices] = batched_pua_payments
        pua_payments = pua_payments.to_sparse_coo()

        # update enrollment_status
        agents_enrollment_status = agents_enrollment_status*(1 - newly_enrolled_agents) + newly_enrolled_agents*(1 - agents_enrollment_status)

        return pua_payments, agents_enrollment_status

    def receive_pua(self, t, current_assets, pua_enrollment_status, pua_payment):
        '''update current_assets of enrolled agents with their pua_payment for the day'''
        new_assets = current_assets + pua_enrollment_status*pua_payment[:, t].to_dense()

        return new_assets

    def end_pua(self, t, pua_enrollment_status, end_date):
        '''update enrollment_status if t >= end_date. pua_enrollment_status is a binary flag'''
        print("Check for end_date==t condition..")
        new_pua_enrollment_status = compare(end_date, t)*pua_enrollment_status + compare(t, end_date)*(1 - pua_enrollment_status)

        return new_pua_enrollment_status

    def forward(self, state, action):
        t = state['current_step']
        input_variables = self.input_variables

        current_assets = get_by_path(state, re.split("/", input_variables['agent_assets']))
        pua_enrollment_status = get_by_path(state, re.split("/", input_variables['pua_enrollment']))
        pua_payments = get_by_path(state, re.split("/", input_variables['pua_enrollment']))
        end_date = get_by_path(state, re.split("/", input_variables['end_date']))
        admin_delay = get_by_path(state, re.split("/", input_variables['admin_delay']))

        request_pua_action = action['request_pua']

        current_assets = self.receive_pua(t, current_assets, pua_enrollment_status, pua_payments)
        pua_enrollment_status = self.end_pua(t, pua_enrollment_status, end_date)

        pua_enrollment_status, pua_payments, end_date = self.start_pua(t, request_pua_action, pua_enrollment_status, pua_payments, end_date)

        return {self.output_variables[0]: current_assets, self.output_variables[1]: pua_enrollment_status, self.output_variables[2]: pua_payments, self.output_variables[4]: end_date}
