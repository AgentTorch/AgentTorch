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
    
num_agents = 5
recovered_death_mask = torch.tensor([1, 0, 0, 1, 1])
recovery_rate = torch.tensor(0.7, requires_grad=True)

agent_specific_prob = recovery_rate*recovered_death_mask
st_bernoulli = StraightThroughBernoulli.apply

recovered_agents = st_bernoulli(agent_specific_prob)
print("recovered agents: ", recovered_agents)

total_dead = (1 - recovered_agents).sum()
total_dead.backward()
print(recovery_rate.grad)

# st_bernoulli = StraightThroughBernoulli.apply
# p = torch.tensor(0.4, requires_grad=True)
# r = st_bernoulli(p)
# print(r)
# r.backward()
# print(p.grad)




# R = torch.tensor(torch.tensor([1.0, 2.0, 3.0]), requires_grad=True)
# R = nn.Parameter([1.0, 2.0, 3.0])
# update_value = torch.tensor([0.5, 0.5, 0.5])

# ans = R * update_value
# ans.sum().backward()
# print(R.grad)  # tensor([0.5000, 0.5000, 0.5000])

# R2 = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
# # R = R2  # Make R and R2 refer to the same tensor object
# R = R2

# ans = R * update_value
# ans.sum().backward()
# print(R2.grad)  # tensor([0.5000, 0.5000, 0.5000])