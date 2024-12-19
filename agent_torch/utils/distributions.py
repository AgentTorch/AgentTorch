"""Differentiable Discrete Distributions for Dynamics and Interventions"""

import numpy as np
import torch
import torch.nn as nn


class StraightThroughBernoulli(torch.autograd.Function):
    generate_vmap_rule = True

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, p):
        result = torch.bernoulli(p)
        ctx.save_for_backward(result, p)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, p = ctx.saved_tensors
        ws = torch.ones(result.shape)
        return grad_output * ws


class Bernoulli(torch.autograd.Function):
    generate_vmap_rule = True

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, p):
        result = torch.bernoulli(p)
        ctx.save_for_backward(result, p)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, p = ctx.saved_tensors
        w_minus = (1.0 / p) / 2  # jump down, averaged for eps > 0 and eps < 0
        w_plus = (1.0 / (1.0 - p)) / 2  # jump up, averaged for eps > 0 and eps < 0

        ws = torch.where(
            result == 1, w_minus, w_plus
        )  # stochastic triple: total grad -> + smoothing rule)
        return grad_output * ws


class Binomial(torch.autograd.Function):
    generate_vmap_rule = True

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, n, p):
        result = torch.distributions.binomial.Binomial(n, p).sample()
        ctx.save_for_backward(result, p, n)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, p, n = ctx.saved_tensors
        w_minus = result / p  # derivative contributions of unit jump down
        w_plus = (n - result) / (1.0 - p)  # derivative contributions of unit jump up

        wminus_cont = torch.where(result > 0, w_minus, 0)
        wplus_cont = torch.where(result < n, w_plus, 0)

        ws = (wminus_cont + wplus_cont) / 2

        return None, grad_output * ws
