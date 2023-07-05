import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import numpy as np
import re
from abc import ABC, abstractmethod

from AgentTorch.helpers.general import *

class SubstepObservation(nn.Module, ABC):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables
        
        self.learnable_args, self.fixed_args = arguments['learnable'], arguments['fixed']
        if self.learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

        self.args = {**self.fixed_args, **self.learnable_args}

    @abstractmethod
    def forward(self, state):
        pass

class SubstepAction(nn.Module, ABC):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables
        
        self.learnable_args, self.fixed_args = arguments['learnable'], arguments['fixed']
        if self.learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

        self.args = {**self.fixed_args, **self.learnable_args}

    @abstractmethod
    def forward(self, state, observation):
        pass


class SubstepTransition(nn.Module, ABC):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables
        
        self.learnable_args, self.fixed_args = arguments['learnable'], arguments['fixed']
        if self.learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

        self.args = {**self.fixed_args, **self.learnable_args}

    @abstractmethod
    def forward(self, state, action):
        pass

class SubstepTransitionMessagePassing(MessagePassing, ABC):
    def __init__(self, config, input_variables, output_variables, arguments):
        super(SubstepTransitionMessagePassing, self).__init__(aggr='add')
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables
        
        self.learnable_args, self.fixed_args = arguments['learnable'], arguments['fixed']
        if self.learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

    @abstractmethod
    def forward(self, state, action):
        pass