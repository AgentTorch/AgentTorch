from typing import Dict, Any, List, Optional, Union
import yaml
from .state_builder import PropertyBuilder


class ObservationBuilder:
    def __init__(self):
        self.config = {}

    def add_observation(
        self,
        name: str,
        input_vars: Dict[str, str],
        output_vars: List[str],
        arguments: Dict[str, Any],
    ) -> "ObservationBuilder":
        self.config[name] = {
            "input_variables": input_vars,
            "output_variables": output_vars,
            "arguments": arguments,
        }
        return self


class PolicyBuilder:
    def __init__(self):
        self.config = {}

    def add_policy(
        self,
        name: str,
        generator: str,
        input_vars: Dict[str, str],
        output_vars: List[str],
        arguments: Dict[str, Any],
    ) -> "PolicyBuilder":
        self.config[name] = {
            "generator": generator,
            "input_variables": input_vars,
            "output_variables": output_vars,
            "arguments": arguments,
        }
        return self


class TransitionBuilder:
    def __init__(self):
        self.config = {}

    def add_transition(
        self,
        name: str,
        generator: str,
        input_vars: Dict[str, str],
        output_vars: List[str],
        arguments: Dict[str, Union[Dict[str, Any], PropertyBuilder]],
    ) -> "TransitionBuilder":
        """Add a transition with arguments that can be PropertyBuilder objects.

        Args:
            name: Name of the transition
            generator: Generator function name
            input_vars: Input variable mappings
            output_vars: Output variable names
            arguments: Arguments as either PropertyBuilder objects or raw config dicts
        """
        processed_args = {}
        for arg_name, arg_value in arguments.items():
            if isinstance(arg_value, PropertyBuilder):
                processed_args[arg_name] = arg_value.config
            else:
                processed_args[arg_name] = arg_value

        self.config[name] = {
            "generator": generator,
            "input_variables": input_vars,
            "output_variables": output_vars,
            "arguments": processed_args,
        }
        return self


class SubstepBuilder:
    def __init__(self, name: str, description: str):
        self.config = {
            "name": name,
            "description": description,
            "active_agents": [],
            "observation": {},
            "policy": {},
            "reward": None,
            "transition": {},
        }

    def add_active_agent(self, agent_type: str) -> "SubstepBuilder":
        self.config["active_agents"].append(agent_type)
        return self

    def set_observation(
        self, agent_type: str, obs_config: Optional[Dict[str, Any]] = None
    ) -> "SubstepBuilder":
        self.config["observation"][agent_type] = obs_config
        return self

    def set_policy(
        self, agent_type: str, policy_builder: PolicyBuilder
    ) -> "SubstepBuilder":
        self.config["policy"][agent_type] = policy_builder.config
        return self

    def set_transition(self, transition_builder: TransitionBuilder) -> "SubstepBuilder":
        self.config["transition"] = transition_builder.config
        return self

    def set_reward(self, reward_config: Optional[Dict[str, Any]]) -> "SubstepBuilder":
        self.config["reward"] = reward_config
        return self


class ConfigBuilder:
    def __init__(self):
        self.config = {"simulation_metadata": {}, "state": {}, "substeps": {}}

    def set_metadata(self, metadata: Dict[str, Any]) -> "ConfigBuilder":
        self.config["simulation_metadata"] = metadata
        return self

    def set_state(self, state_config: Dict[str, Any]) -> "ConfigBuilder":
        self.config["state"] = state_config
        return self

    def add_substep(
        self, step_id: str, substep_builder: SubstepBuilder
    ) -> "ConfigBuilder":
        self.config["substeps"][step_id] = substep_builder.config
        return self

    def save_yaml(self, filepath: str):
        with open(filepath, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        return self.config
