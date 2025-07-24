from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_var

@Registry.register_substep("update_velocity", "transition")
class UpdateVelocity(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize any learnable parameters from kwargs
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get('learnable', False):
                setattr(self, key, value['value'])
    
    def forward(self, state: Dict[str, Any], action) -> Dict[str, Any]:
        """
        Update velocity by applying steering force with force limits.
        
        Args:
            state: Current state of the simulation
            action: Action from policy containing steering_force
            
        Returns:
            Dict containing: velocity
        """
        input_variables = self.input_variables
        
        # Get input variables from state
        velocity = get_var(state, input_variables["velocity"])
        max_force = get_var(state, input_variables["max_force"])
        
        # Get steering force from action (output of the policy)
        steering_force = action["boids"]["steering_force"]
        
        # Apply force limit: clamp steering force magnitude to max_force
        force_magnitude = torch.norm(steering_force, dim=1, keepdim=True)
        limited_force = torch.where(
            force_magnitude > max_force,
            steering_force * max_force / force_magnitude,
            steering_force
        )
        
        # Update velocity by adding limited steering force
        new_velocity = velocity + limited_force
        
        return {
            self.output_variables[0]: new_velocity  # velocity
        }
