import torch.nn as nn

# each policy has: observation, params

class QuarantinePolicy(nn.Module):
    def __init__(self):
        super(QuarantinePolicy, self).__init__()
    
    def forward(self, observation, params):
        # returns a mask of shape (num_agents,)
        return params['agents']['citizens']['quarantine_status']
    