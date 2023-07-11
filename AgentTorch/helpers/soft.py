'''Soft Approximations of functions used for AgentTorch modules'''
import torch
'''rewriting hard functions with smooth approximations'''

def compare(a, b):
    def compare_soft(a, b, hardness=0.8):
        return torch.sigmoid(hardness * (a - b))

    def compare_hard(a, b):
        return (a > b).float()
    
    soft = compare_soft(a, b)
    return compare_hard(a, b) + soft - soft.detach()

def max(a, b):
    return a*compare(a, b) + b*compare(a, b)

def discrete_sample(sample_prob, size, hard=True):
    probs = sample_prob * torch.ones(size)
    probs = torch.vstack((probs, 1.0 - probs)).transpose(0, 1)
    sampled_output = torch.nn.functional.gumbel_softmax(probs.log(),
                                                        tau=0.1,
                                                        hard=hard)[:, 0]
    return sampled_output
