from typing import Dict, Any, List, Optional, Union
import yaml


class PropertyBuilder:
    def __init__(
        self,
        name: str,
        dtype: str = None,
        shape: List[Union[int, str]] = None,
        learnable: bool = False,
        value: Any = None,
        initialization_function: Dict[str, Any] = None,
    ):
        """Initialize a property with common defaults.

        Args:
            name: Name of the property
            dtype: Data type (e.g. "float", "int", "bool")
            shape: Shape of the property (e.g. [num_agents, 1])
            learnable: Whether the property is learnable
            value: Default value
            initialization_function: Initialization function config
        """
        self.config = {
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "learnable": learnable,
            "initialization_function": initialization_function,
            "value": value,
        }

    @classmethod
    def create_argument(
        cls,
        name: str,
        value: Any,
        dtype: str = "float",
        shape: List[int] = None,
        learnable: bool = False,
    ) -> "PropertyBuilder":
        """Convenience method to create an argument property with common defaults."""
        return cls(
            name=name,
            dtype=dtype,
            shape=shape or [1],
            learnable=learnable,
            value=value,
            initialization_function=None,
        )

    def set_dtype(self, dtype: str) -> "PropertyBuilder":
        self.config["dtype"] = dtype
        return self

    def set_shape(self, shape: List[Union[int, str]]) -> "PropertyBuilder":
        self.config["shape"] = shape
        return self

    def set_learnable(self, learnable: bool) -> "PropertyBuilder":
        self.config["learnable"] = learnable
        return self

    def set_value(self, value: Any) -> "PropertyBuilder":
        self.config["value"] = value
        return self

    def set_initialization(
        self, generator: str, arguments: Dict[str, Any]
    ) -> "PropertyBuilder":
        self.config["initialization_function"] = {
            "generator": generator,
            "arguments": arguments,
        }
        return self


class AgentBuilder:
    def __init__(self, agent_type: str, number: int):
        self.config = {"number": number, "properties": {}}
        self.agent_type = agent_type

    def add_property(self, property_builder: PropertyBuilder) -> "AgentBuilder":
        self.config["properties"][
            property_builder.config["name"]
        ] = property_builder.config
        return self


class EnvironmentBuilder:
    def __init__(self):
        self.config = {}

    def add_variable(self, property_builder: PropertyBuilder) -> "EnvironmentBuilder":
        self.config[property_builder.config["name"]] = property_builder.config
        return self


class NetworkBuilder:
    def __init__(self):
        self.config = {"agent_agent": {}, "objects": None}

    def add_network(
        self, name: str, network_type: str, arguments: Dict[str, Any]
    ) -> "NetworkBuilder":
        self.config["agent_agent"][name] = {
            "type": network_type,
            "arguments": arguments,
        }
        return self


class StateBuilder:
    def __init__(self):
        self.config = {"agents": {}, "environment": {}, "network": {}, "objects": None}

    def add_agent(self, agent_type: str, agent_builder: AgentBuilder) -> "StateBuilder":
        self.config["agents"][agent_type] = agent_builder.config
        return self

    def set_environment(self, env_builder: EnvironmentBuilder) -> "StateBuilder":
        self.config["environment"] = env_builder.config
        return self

    def set_network(self, network_builder: NetworkBuilder) -> "StateBuilder":
        self.config["network"] = network_builder.config
        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.config
