import torch.nn as nn
from torch_geometric.nn import MessagePassing
from abc import ABC, abstractmethod

from agent_torch.helpers.general import *


class SubstepObservation(nn.Module, ABC):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables

        self.learnable_args, self.fixed_args = (
            arguments["learnable"],
            arguments["fixed"],
        )
        if self.learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

        if self.config["simulation_metadata"]["calibration"] == True:
            for key, value in self.learnable_args.items():
                tensor_name = f"calibrate_{key}"
                setattr(self, tensor_name, torch.tensor(value, requires_grad=True))

        self.args = {**self.fixed_args, **self.learnable_args}
        self.custom_observation_network = None

    @abstractmethod
    def forward(self, state):
        pass


class SubstepAction(nn.Module, ABC):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables

        self.learnable_args, self.fixed_args = (
            arguments["learnable"],
            arguments["fixed"],
        )
        if self.learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

        if self.config["simulation_metadata"]["calibration"] == True:
            for key, value in self.learnable_args.items():
                tensor_name = f"calibrate_{key}"
                setattr(self, tensor_name, torch.tensor(value, requires_grad=True))

        self.args = {**self.fixed_args, **self.learnable_args}
        self.custom_action_network = None

    @abstractmethod
    def forward(self, state, observation):
        pass


class SubstepTransition(nn.Module, ABC):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables

        self.learnable_args, self.fixed_args = (
            arguments["learnable"],
            arguments["fixed"],
        )
        if self.learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

        if self.config["simulation_metadata"]["calibration"] == True:
            for key, value in self.learnable_args.items():
                tensor_name = f"calibrate_{key}"
                setattr(self, tensor_name, torch.tensor(value, requires_grad=True))

        self.args = {**self.fixed_args, **self.learnable_args}
        self.custom_transition_network = None

    @abstractmethod
    def forward(self, state, action):
        pass


class SubstepTransitionMessagePassing(MessagePassing, ABC):
    def __init__(self, config, input_variables, output_variables, arguments):
        super(SubstepTransitionMessagePassing, self).__init__(aggr="add")
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables

        self.learnable_args, self.fixed_args = (
            arguments["learnable"],
            arguments["fixed"],
        )
        if self.learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

        if self.config["simulation_metadata"]["calibration"] == True:
            for key, value in self.learnable_args.items():
                tensor_name = f"calibrate_{key}"
                setattr(self, tensor_name, torch.tensor(value, requires_grad=True))

    @abstractmethod
    def forward(self, state, action):
        pass
