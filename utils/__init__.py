from utils.initialization import *
from utils.observation import *
from utils.policy import *
# from utils.reward import *
from utils.transition import *

import utils.general
import torch.nn as nn

class BaseUtil(nn.Module):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables
        
        learnable_args, fixed_args = arguments['learnable'], arguments['fixed']
        self.learnable_args, self.fixed_args = learnable_args, fixed_args

        if learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)
    
    def forward(self):
        pass
    