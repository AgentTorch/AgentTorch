import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gamma

class GetLamGammaIntegrals(nn.Module):
    def __init__(self, input_variables, output_variables, arguments):
        super().__init__()
        self.arguments = arguments
        self.input_variables = input_variables
        self.output_variables = output_variables
    
    def forward(self, state):
        b = 
        