from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_var


@Registry.register_substep("new_transmission", "transition")
class NewTransmission(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize any learnable parameters from kwargs
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get("learnable", False):
                setattr(self, key, value["value"])

    def forward(self, state: Dict[str, Any], action) -> Dict[str, Any]:
        """
        Forward pass of the NewTransmission.

        Args:
            state: Current state of the simulation
            action: Action from policy

        Returns:
            Dict containing: disease_stage
        """
        input_variables = self.input_variables

        # Get input variables from state
        disease_stage = get_var(state, input_variables["disease_stage"])
        age = get_var(state, input_variables["age"])

        # TODO: Implement the forward logic

        return {
            self.output_variables[0]: None,  # disease_stage
        }
