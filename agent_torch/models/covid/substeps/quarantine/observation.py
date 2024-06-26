import torch
import torch.nn as nn
import numpy as np
import re
from agent_torch.core.substep import SubstepObservation
from agent_torch.core.helpers import get_by_path


class GetFromState(SubstepObservation):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.input_variables = input_variables
        self.output_variables = output_variables

    def forward(self, state):
        input_variables = self.input_variables

        return {
            ix: get_by_path(state, re.split("/", input_variables[ix]))
            for ix in input_variables.keys()
        }
