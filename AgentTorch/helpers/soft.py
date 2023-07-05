'''Soft Approximations for AgentTorch modules'''

import torch

def compare_max(a, b, temp):
    return torch.log(torch.exp(temp * a) + torch.exp(temp * b)) / temp

def compare_min(a, b, temp):
    return -compare_max(-a, -b, temp)

def discrete_sample(sample_prob, size, hard=True):
    probs = sample_prob * torch.ones(size)
    probs = torch.vstack((probs, 1.0 - probs)).transpose(0, 1)
    sampled_output = torch.nn.functional.gumbel_softmax(probs.log(),
                                                        tau=0.1,
                                                        hard=hard)[:, 0]
    return sampled_output
