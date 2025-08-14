"""
Slot System for P3O Template Optimization
=========================================

Simple slot system that creates P3O-compatible placeholder choices.
Each slot represents a learnable field in a template.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Callable


class Slot:
    """
    A P3O-compatible slot for template optimization.
    
    Creates the exact (num_options, lambda_func) format that P3O expects.
    Each slot represents a learnable field with multiple presentation options.
    
    Args:
        field_name: Name of the field this slot represents
        learnable: Whether this slot should have learnable parameters for P3O
        
    Example:
        # Create a slot for a learnable field
        job_slot = Slot("job_name", learnable=True)
        
        # Get P3O format
        num_options, lambda_func = job_slot.get_p3o_choice()
        
        # Test the lambda with data
        data = {"job_name": "Software Engineer"}
        result = lambda_func(1, data)  # Returns: "Software Engineer"
        result = lambda_func(2, data)  # Returns: "job_name: Software Engineer"
    """
    
    def __init__(self, field_name: str, learnable: bool = True):
        self.field_name = field_name
        self.learnable = learnable
        
        # P3O parameters - 5 options: Skip, Direct, Labeled, Contextual, Descriptive
        self.num_options = 5
        self.theta = None
        
        if learnable:
            self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize learnable parameters for P3O optimization."""
        # Bias against skip (index 0) and toward actual content
        init_values = torch.full((self.num_options,), 0.5)
        init_values[0] = -1.0  # Bias against skip
        self.theta = nn.Parameter(init_values)
    
    def _create_p3o_lambda(self, mapping: Dict[str, Any] = None) -> Callable[[int, Dict[str, Any]], str]:
        """
        Create the P3O lambda function for this field.
        
        Returns:
            Lambda function that takes (category, data) and returns formatted text
        """
        field_name = self.field_name  # Capture in closure
        
        def field_lambda(cat: int, data: Dict[str, Any]) -> str:
            """P3O slot lambda - handles presentation choice based on category."""
            # Non-learnable fields should ALWAYS return their value, ignore P3O category
            if not self.learnable:
                raw_value = data.get(field_name)
                if raw_value is not None:
                    # Handle both raw and pre-mapped values
                    if mapping and field_name in mapping and isinstance(raw_value, (int, float)):
                        mapping_list = mapping[field_name]
                        if 0 <= int(raw_value) < len(mapping_list):
                            return str(mapping_list[int(raw_value)])
                        else:
                            return str(raw_value)
                    elif isinstance(raw_value, str) and raw_value not in ["", "nan", "None"]:
                        return raw_value
                    else:
                        return str(raw_value)
                return ""  # Only return empty if truly no data
            
            if cat == 0:
                # Skip option - P3O can choose to omit this field entirely
                return ""
            

            # Get raw value and apply mapping transformation
            raw_value = data.get(field_name)
            if raw_value is None:
                # Return empty string instead of placeholder when data is missing
                return ""
            
            # Handle both raw values (need mapping) and pre-mapped values (already strings)
            if mapping and field_name in mapping and isinstance(raw_value, (int, float)):
                # Raw value case: apply mapping
                mapping_list = mapping[field_name]
                if 0 <= int(raw_value) < len(mapping_list):
                    mapped_value = mapping_list[int(raw_value)]
                else:
                    mapped_value = str(raw_value)
            elif isinstance(raw_value, str) and raw_value not in ["", "nan", "None"]:
                # Pre-mapped value case: use as-is (P3O assembled data)
                mapped_value = raw_value
            else:
                # Fallback: convert to string
                mapped_value = str(raw_value)
            
            # Apply presentation choice
            if cat == 1:
                # Direct value - just the mapped data
                return str(mapped_value)
            elif cat == 2:
                # Labeled format - "field_name: value"
                return f"{field_name}: {mapped_value}"
            elif cat == 3:
                # Contextual format - "with value"
                return f"with {mapped_value}"
            elif cat == 4:
                # Descriptive format - "The field_name is value"
                return f"The {field_name} is {mapped_value}"
            else:
                # Fallback to direct value
                return str(mapped_value)
        
        return field_lambda
    
    def get_p3o_choice(self, mapping: Dict[str, Any] = None) -> Tuple[int, Callable[[int, Dict[str, Any]], str]]:
        """
        Get P3O placeholder choice format.
        
        Args:
            mapping: Mapping dictionary for value transformations
        
        Returns:
            Tuple of (num_options, lambda_function) that P3O expects
        """
        return (self.num_options, self._create_p3o_lambda(mapping=mapping))
    
    def sample_index(self) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an index for learnable slots during P3O optimization.
        
        Returns:
            Tuple of (sampled_index, log_probability, entropy)
        """
        if not self.learnable or self.theta is None:
            return 1, torch.tensor(0.0), torch.tensor(0.0)  # Default to direct value
        
        probs = F.softmax(self.theta, dim=0)
        dist = torch.distributions.Categorical(probs)
        sampled_idx = dist.sample()
        
        return sampled_idx.item(), dist.log_prob(sampled_idx), dist.entropy()
    
    def get_probabilities(self) -> torch.Tensor:
        """Get probability distribution over options for learnable slots."""
        if not self.learnable or self.theta is None:
            # Return one-hot for direct value (index 1)
            probs = torch.zeros(self.num_options)
            probs[1] = 1.0
            return probs
        
        return F.softmax(self.theta, dim=0)
    
    def get_best_index(self) -> int:
        """Get the best option index based on learned probabilities."""
        if not self.learnable or self.theta is None:
            return 1  # Default to direct value
        
        probs = self.get_probabilities()
        return torch.argmax(probs).item()


def create_slots_from_fields(fields: list) -> Dict[str, Slot]:
    """
    Create Slot objects from parsed template fields.
    
    Args:
        fields: List of (field_name, is_learnable) tuples
        
    Returns:
        Dictionary mapping field names to Slot objects
    """
    slots = {}
    for field_name, is_learnable in fields:
        if is_learnable:  # Only create slots for learnable fields
            slots[field_name] = Slot(field_name, learnable=True)
    return slots