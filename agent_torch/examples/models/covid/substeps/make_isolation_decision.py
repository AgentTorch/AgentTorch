from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers import get_var


@Registry.register_substep("make_isolation_decision", "policy")
class MakeIsolationDecision(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize any learnable parameters from kwargs
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get("learnable", False):
                setattr(self, key, value["value"])

    def forward(self, state: Dict[str, Any], observations) -> Dict[str, Any]:
        """
        Forward pass of the MakeIsolationDecision.

        Args:
            state: Current state of the simulation
            observations: Agent observations

        Returns:
            Dict containing: isolation_decision
        """
        input_variables = self.input_variables

        # Get input variables from state
        age = get_var(state, input_variables["age"])

        # TODO: Implement the forward logic

        return {
            self.output_variables[0]: None,  # isolation_decision
        }
