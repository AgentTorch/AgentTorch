from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers import get_var


@Registry.register_substep("move", "policy")
class RandomMove(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state: Dict[str, Any], observations) -> Dict[str, Any]:
        """Generate random movement directions."""
        # Get current positions
        positions = get_var(state, self.input_variables["position"])
        num_agents = positions.shape[0]
        device = positions.device  # Get device from positions

        # Ensure step_size is on the same device as positions
        step_size = self.learnable_args["step_size"].to(device)

        # Generate random angles - ensure on same device as positions
        angles = torch.rand(num_agents, device=device) * 2 * torch.pi

        # Convert to direction vectors
        direction = torch.stack(
            [torch.cos(angles) * step_size, torch.sin(angles) * step_size],
            dim=1,
        )

        # Return with exact output variable name from config
        outputs = {}
        outputs[self.output_variables[0]] = direction

        return outputs
