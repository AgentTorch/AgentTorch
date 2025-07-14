import os
from typing import Dict, Any, List
from dataclasses import dataclass
from agent_torch.config.substep_builder import (
    SubstepBuilder,
    PolicyBuilder,
    TransitionBuilder,
    ObservationBuilder,
)


@dataclass
class SubstepImplementation:
    name: str  # snake_case name for registry (e.g. "new_transmission")
    type: str  # "policy" or "transition"
    input_vars: Dict[str, str]
    output_vars: List[str]
    arguments: Dict[str, Any]

    @property
    def class_name(self) -> str:
        """Convert snake_case name to PascalCase class name."""
        return "".join(word.title() for word in self.name.split("_"))


def generate_substep_file(substep: SubstepImplementation, output_dir: str):
    """Generate a Python file implementing the substep."""

    # Create the substeps directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Determine parent class based on type
    parent_class = "SubstepAction" if substep.type == "policy" else "SubstepTransition"

    # Generate the file content
    content = f"""from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import {parent_class}
from agent_torch.core.helpers import get_var

@Registry.register_substep("{substep.name}", "{substep.type}")
class {substep.class_name}({parent_class}):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize any learnable parameters from kwargs
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get('learnable', False):
                setattr(self, key, value['value'])
    
    def forward(self, state: Dict[str, Any], {'action' if substep.type == 'transition' else 'observations'}) -> Dict[str, Any]:
        \"\"\"
        Forward pass of the {substep.class_name}.
        
        Args:
            state: Current state of the simulation
            {'action: Action from policy' if substep.type == 'transition' else 'observations: Agent observations'}
            
        Returns:
            Dict containing: {', '.join(substep.output_vars)}
        \"\"\"
        input_variables = self.input_variables
        
        # Get input variables from state
        {chr(10).join(f'{k.split("/")[-1]} = get_var(state, input_variables["{k}"])' for k in substep.input_vars.keys())}
        
        # TODO: Implement the forward logic
        
        return {{
            {chr(10).join(f"            self.output_variables[{i}]: None,  # {var}" for i, var in enumerate(substep.output_vars))}
        }}
"""

    # Write the file - use original snake_case name
    output_file = os.path.join(output_dir, f"{substep.name}.py")
    with open(output_file, "w") as f:
        f.write(content)

    return output_file


class SubstepBuilderWithImpl(SubstepBuilder):
    """Extended SubstepBuilder that also generates implementation files."""

    def __init__(self, name: str, description: str, output_dir: str):
        super().__init__(name, description)
        self.output_dir = output_dir
        self.implementations = []

    def add_observation(
        self, agent_type: str, observation_builder: ObservationBuilder
    ) -> "SubstepBuilderWithImpl":
        """Add observation and generate implementation files."""
        super().set_observation(agent_type, observation_builder)

        # Extract observations to generate implementations
        for obs_name, obs in observation_builder.config.items():
            impl = SubstepImplementation(
                name=obs_name,  # Use original name instead of converting to lowercase
                type="observation",
                input_vars=obs["input_variables"],
                output_vars=obs["output_variables"],
                arguments=obs["arguments"],
            )
            self.implementations.append(impl)

        return self

    def add_policy(
        self, agent_type: str, policy_builder: PolicyBuilder
    ) -> "SubstepBuilderWithImpl":
        """Add policy and generate implementation files."""
        super().set_policy(agent_type, policy_builder)

        # Extract policies to generate implementations
        for policy_name, policy in policy_builder.config.items():
            impl = SubstepImplementation(
                name=policy_name,  # Use original name instead of converting to lowercase
                type="policy",
                input_vars=policy["input_variables"],
                output_vars=policy["output_variables"],
                arguments=policy["arguments"],
            )
            self.implementations.append(impl)

        return self

    def set_transition(
        self, transition_builder: TransitionBuilder
    ) -> "SubstepBuilderWithImpl":
        """Set transition and generate implementation files."""
        super().set_transition(transition_builder)

        # Extract transitions to generate implementations
        for trans_name, trans in transition_builder.config.items():
            impl = SubstepImplementation(
                name=trans_name,  # Use original name instead of converting to lowercase
                type="transition",
                input_vars=trans["input_variables"],
                output_vars=trans["output_variables"],
                arguments=trans["arguments"],
            )
            self.implementations.append(impl)

        return self

    def generate_implementations(self):
        """Generate all implementation files."""
        generated_files = []
        for impl in self.implementations:
            file_path = generate_substep_file(impl, self.output_dir)
            generated_files.append(file_path)
        return generated_files
