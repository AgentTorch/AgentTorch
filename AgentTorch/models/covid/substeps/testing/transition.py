import torch
import torch.nn as nn
import re

from AgentTorch.substep import SubstepTransition

class Testing(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = self.config['simulation_metadata']['device']

    def update_test_eligibility(self):
        pass

    def get_test_results(self):
        pass
    
    def forward(self, state, action):
        pass