'''Soft Approximations of functions used for AgentTorch modules'''
import torch

def compare(a, b):
    def compare_soft(a, b, hardness=0.8):
        return torch.sigmoid(hardness * (a - b))

    def compare_hard(a, b):
        return (a > b).float()
    
    soft = compare_soft(a, b)
    return compare_hard(a, b) + soft - soft.detach()

def max(a, b):
    return a*compare(a, b) + b*compare(b, a)

def min(a, b):
    return a*compare(b, a) + b*compare(a, b)

def logical_not(a, grad=True):
    def hard_not(a):
        assert a.dtype == torch.long
        return torch.logical_not(a)
    
    def soft_not(a):
        return 1 - a
    
    if not grad:
        return hard_not(a.long())

    soft = soft_not(a)
    return hard_not(a.long()) + (soft - soft.detach())

def logical_or(a, b, grad=True):
    def hard_or(a, b):
        assert a.dtype == torch.long and b.dtype == torch.long
        return torch.logical_or(a, b)
    
    def soft_or(a, b):
        return a + b
    
    if not grad:
        return hard_or(a.long(), b.long())

    soft = soft_or(a, b)
    return hard_or(a.long(), b.long()) + (soft - soft.detach())

def logical_and(a, b, grad=True):
    def hard_and(a, b):
        assert a.dtype == torch.long and b.dtype == torch.long
        return torch.logical_and(a, b)
    
    def soft_and(a, b):
        return a * b
    
    if not grad:
        return hard_and(a.long(), b.long())

    soft = soft_and(a, b)
    return hard_and(a.long(), b.long()) + (soft - soft.detach())

def discrete_sample(sample_prob, size, hard=True):
    probs = sample_prob * torch.ones(size)
    probs = torch.hstack((probs, 1 - probs))
    prob_logits = torch.log(probs+1e-9)
    sampled_output = torch.nn.functional.gumbel_softmax(prob_logits,
                                                        tau=0.1,
                                                        hard=hard, dim=1)[:, 0]
    return sampled_output
