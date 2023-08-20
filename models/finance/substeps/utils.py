import torch
import torch.nn as nn
import math

from AgentTorch.helpers import discrete_sample, compare, max

# StimulusPayment
def initialize_stimulus_eligibility(shape, params):
    '''Agents who earn less than a threshold are eligible'''
    agents_income = params['agents_incomes']
    eligibility_threshold = params['threshold']
    
    stimulus_eligibility = 1 - compare(agents_income, eligibility_threshold)
    assert stimulus_eligibility.shape == tuple(shape)

    return stimulus_eligibility

def initialize_stimulus_payment(shape, params):
    '''Fixed amount for all agents, provided at specific time intervals'''
    amounts = torch.tensor(params['amount'], dtype=torch.float32) # [600., 300,]
    time_steps = params['dates'] # [23, 47]

    stimulus_payments = torch.sparse_coo_tensor(indices=time_steps, values=amounts, size=shape)

    return stimulus_payments


# FPUC: Federal Pandemic Unemployment Compensation
def initialize_fpuc_eligibility(shape, params):
    '''
        - All unemployed individuals are eligible for fpuc
        - This is just initialization, recheck at every timestep    
    '''
    agents_occupation_status = params['agents_occupation_status']
    fpuc_eligibility = (agents_occupation_status == 0) # [0, 1] for (unemployed, employed)

    assert fpuc_eligibility.shape == tuple(shape)

    return fpuc_eligibility

def initialize_fpuc_payment(shape, params):
    '''Everyone receives same payment, from each start-end date interval, once every week'''
    fpuc_amounts = params['amounts']
    fpuc_date_ranges = params['date_ranges']
    frequency = params['payment_frequency'] # in num_steps

    fpuc_payments = torch.zeros(shape)
    for ix in range(len(fpuc_date_ranges)):
        start_date, end_date = fpuc_date_ranges[ix]
        payment_amount = fpuc_amounts[ix]
        fpuc_payments[start_date:end_date+1:frequency] = payment_amount

    fpuc_payments = fpuc_payments.to_sparse_coo()

    return fpuc_payments

# PUA: Pandemic Unemployment Assistance
def initialize_pua_enrollment(shape, params):
    '''
        - some unemployed agents may have requested eligibility beforehand
        - sample and define agent eligibility during the simulation     
    '''
    request_prob = torch.tensor([params['request_pua_prob']], requires_grad=True)
    agents_request_enrollment = discrete_sample(sample_prob=request_prob, size=shape).unsqueeze(dim=1)

    agents_occupation_status = params['agents_occupation_status']
    pua_enrollment = agents_request_enrollment * (agents_occupation_status == 0)

    return pua_enrollment

def _income_to_pua(agent_incomes):
    threshold = 304.0*torch.ones_like(agent_incomes)
    pua_amount = torch.min(agent_incomes / 2, threshold)

    return pua_amount

def _get_pua_payments(payment_amount, shape, request_date, delay=3, frequency=7, num_payments=26):
    '''Computes PUA for a single agent'''
    agent_pua_payments = payment_amount*torch.zeros_like(shape)

    start_date = request_date + delay
    end_date = request_date + frequency*num_payments
    first_amount = payment_amount*math.ceil(delay / frequency)
    recurring_amount = payment_amount

    agent_pua_payments[start_date] = first_amount
    agent_pua_payments[start_date + frequency: end_date: frequency] = recurring_amount

    return agent_pua_payments #(num_agents, num_steps)

def initialize_pua_payments(shape, params):
    '''TODO: Ensure only initialize for eligible agents'''
    agents_income = torch.tensor(params['agent_income'], dtype=torch.float32)

    payment_amount = _income_to_pua(agents_income) #(num_agents,)
    num_steps_tensor = torch.ones((params['num_steps']))
    delay = torch.tensor(params['pua_delay'])

    batched_pua_payments_func = torch.vmap(_get_pua_payments, in_dims=(0, None, None))
    batched_pua_payments = batched_pua_payments_func(payment_amount, num_steps_tensor, delay)

    sparse_batched_pua_payments = batched_pua_payments.to_sparse_coo()

    return sparse_batched_pua_payments