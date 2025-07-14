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


class Geometric(torch.autograd.Function):
    generate_vmap_rule = True

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, p):
        p = torch.clamp(p, min=1e-5, max=1 - 1e-5)  # avoid numerical issues

        u = torch.rand(p.shape)
        result = torch.ceil(torch.log(1 - u) / torch.log(1 - p))  # Inverse CDF method

        ctx.save_for_backward(result, p)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, p = ctx.saved_tensors

        w_plus = 1.0 / p  # weight for jumping up
        w_minus = (result - 1.0) / (1.0 - p)  # weight for jumping down

        w_plus_cont = torch.where(result > 0, w_plus, torch.zeros_like(w_plus))
        w_minus_cont = torch.where(result > 1, w_minus, torch.zeros_like(w_minus))

        ws = (
            w_plus_cont + w_minus_cont
        ) / 2.0  # average weights for unbiased gradientn
        return grad_output * ws


class Categorical(torch.autograd.Function):
    generate_vmap_rule = True  # for functorch if needed

    @staticmethod
    def forward(ctx, p):
        # p is assumed to be of shape (..., k) representing a categorical distribution.
        result = torch.multinomial(p, num_samples=1)  # shape (..., 1)
        ctx.save_for_backward(result, p)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, p = ctx.saved_tensors
        one_hot = torch.zeros_like(p)
        one_hot.scatter_(-1, result, 1.0)
        eps = 1e-8  # small value to avoid division by zero
        # For the chosen category, analogous to a "jump-down" weight.
        w_chosen = (1.0 / (p + eps)) / 2
        # For non-chosen categories, analogous to a "jump-up" weight.
        w_non_chosen = (1.0 / (1.0 - p + eps)) / 2
        ws = one_hot * w_chosen + (1 - one_hot) * w_non_chosen
        grad_output_expanded = grad_output.expand_as(p)
        return grad_output_expanded * ws
