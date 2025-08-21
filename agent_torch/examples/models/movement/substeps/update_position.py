from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_var


@Registry.register_substep("update_position", "transition")
class UpdatePosition(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize learnable parameters
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get("learnable", False):
                setattr(self, key, value["value"])

    def forward(self, state: Dict[str, Any], action) -> Dict[str, Any]:
        """Update agent positions based on movement directions."""
        # Get current positions and movement directions
        positions = get_var(state, self.input_variables["position"])

        self.bounds = self.learnable_args["bounds"]
        directions = action["citizens"]["direction"]
        
        # Ensure bounds tensor is on the same device as positions
        device = positions.device
        bounds = torch.tensor(self.bounds, device=device)

        # Update positions
        new_positions = positions + directions

        # Clip to bounds - ensure min/max tensors are on the same device
        min_bounds = torch.tensor([0, 0], device=device)
        new_positions = torch.clamp(new_positions, min=min_bounds, max=bounds)

        # Return with exact output variable name from config
        outputs = {}
        outputs[self.output_variables[0]] = new_positions
        return outputs
