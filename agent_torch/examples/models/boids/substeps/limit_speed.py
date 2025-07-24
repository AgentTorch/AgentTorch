from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers import get_var

@Registry.register_substep("limit_speed", "policy")
class LimitSpeed(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize learnable parameters from arguments
        self.max_speed = self.learnable_args.get("max_speed", 4.0)
    
    def forward(self, state: Dict[str, Any], observations) -> Dict[str, Any]:
        """
        Apply speed limits to velocities.
        
        Args:
            state: Current state of the simulation
            observations: Agent observations
            
        Returns:
            Dict containing: limited_velocity
        """
        # Get input variables from state
        velocity = get_var(state, self.input_variables["velocity"])
        max_speed = get_var(state, self.input_variables["max_speed"])
        
        # Calculate current speed (magnitude of velocity)
        current_speed = torch.norm(velocity, dim=1, keepdim=True)
        
        # Apply speed limit: if speed > max_speed, scale down to max_speed
        limited_velocity = torch.where(
            current_speed > max_speed,
            velocity * max_speed / current_speed,
            velocity
        )
        
        return {
            self.output_variables[0]: limited_velocity  # limited_velocity
        } 