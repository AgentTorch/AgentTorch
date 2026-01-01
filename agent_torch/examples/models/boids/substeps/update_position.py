from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_var

@Registry.register_substep("update_position", "transition")
class UpdatePosition(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize any learnable parameters from kwargs
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get('learnable', False):
                setattr(self, key, value['value'])
    
    def forward(self, state: Dict[str, Any], action) -> Dict[str, Any]:
        """
        Update position based on velocity and handle boundary wrapping.
        
        Args:
            state: Current state of the simulation
            action: Action from policy containing limited_velocity
            
        Returns:
            Dict containing: position
        """
        input_variables = self.input_variables
        
        # Get input variables from state
        position = get_var(state, input_variables["position"])
        bounds = get_var(state, input_variables["bounds"])
        
        # Get limited velocity from action (output of the policy)
        limited_velocity = action["boids"]["limited_velocity"]
        
        # Update position by adding velocity
        new_position = position + limited_velocity
        
        # Handle boundaries with wrap-around
        # If agent goes beyond right/bottom edge, wrap to left/top edge
        # If agent goes beyond left/top edge, wrap to right/bottom edge
        new_position = torch.where(new_position < 0, 
                                 new_position + bounds, 
                                 new_position)
        new_position = torch.where(new_position >= bounds, 
                                 new_position - bounds, 
                                 new_position)
        
        return {
            self.output_variables[0]: new_position  # position
        }
