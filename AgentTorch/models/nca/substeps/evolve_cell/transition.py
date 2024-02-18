from AgentTorch.substep import SubstepTransition

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NCAEvolve(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device(self.config['simulation_metadata']['device'])
        self.channel_n = self.config['simulation_metadata']['n_channels']
        hidden_size = self.config['simulation_metadata']['hidden_size']
        self.fire_rate = self.config['simulation_metadata']['fire_rate']
        self.angle = self.config['simulation_metadata']['angle']

        self.fc0 = nn.Linear(self.channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, self.channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y = torch.cat((x,y1,y2),1)
        return y

    def forward(self, state, action=None):
        x = state['agents']['automata']['cell_state']
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, self.angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>self.fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        new_state = x.transpose(1,3)
        return {self.output_variables[0]: new_state}