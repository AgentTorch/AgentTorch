# Building Simulations with the Configuration API

This tutorial will guide you through using AgentTorch's Configuration API to build agent-based simulations. The Configuration API provides a powerful and flexible way to define simulation components using a builder pattern.

## Prerequisites

Before starting this tutorial, make sure you have:
- AgentTorch installed (`pip install agent-torch`)
- Basic understanding of agent-based modeling concepts
- Python 3.10 or higher

## Understanding the Configuration Structure

An AgentTorch simulation configuration consists of three main components:

1. **Simulation Metadata**: Global parameters like number of agents, episodes, etc.
2. **State Configuration**: Defines agents, their properties, environment variables, and networks
3. **Substeps**: The simulation logic broken down into observation, policy, and transition functions

Let's build a simple simulation to understand how these components work together.

## Creating a Basic Simulation

We'll create a simple movement simulation where agents move randomly within bounded space. First, import the necessary components:

```python
from agent_torch.config import (
    ConfigBuilder,
    StateBuilder,
    AgentBuilder,
    PropertyBuilder,
    EnvironmentBuilder,
    SubstepBuilder,
    PolicyBuilder,
    TransitionBuilder
)
```

### Step 1: Setting Up Metadata

Start by creating a `ConfigBuilder` and setting simulation metadata:

```python
# Create config builder
config = ConfigBuilder()

# Define metadata
metadata = {
    "num_agents": 100,
    "num_episodes": 10,
    "num_steps_per_episode": 20,
    "device": "cpu"
}
config.set_metadata(metadata)
```

### Step 2: Defining the State

Next, define the agents and their properties:

```python
# Create state builder
state_builder = StateBuilder()

# Create agent builder
agent_builder = AgentBuilder("citizens", metadata["num_agents"])

# Add position property to agents
position = PropertyBuilder("position")\
    .set_dtype("float")\
    .set_shape([metadata["num_agents"], 2])\
    .set_value([0.0, 0.0])

agent_builder.add_property(position)
state_builder.add_agent("citizens", agent_builder)
```

Add environment variables:

```python
# Create environment builder
env_builder = EnvironmentBuilder()

# Add bounds for movement
bounds = PropertyBuilder("bounds")\
    .set_dtype("float")\
    .set_shape([2])\
    .set_value([100.0, 100.0])

env_builder.add_variable(bounds)
state_builder.set_environment(env_builder)

# Set state in config
config.set_state(state_builder.to_dict())
```

### Step 3: Defining Substeps

Create a movement substep with policy and transition:

```python
# Create movement substep
movement = SubstepBuilder("Movement", "Agent movement simulation")
movement.add_active_agent("citizens")

# Add movement policy
policy = PolicyBuilder()
policy.add_policy(
    "move",
    "RandomMove",
    {"position": "agents/citizens/position"},
    ["direction"],
    {"step_size": {"value": 1.0, "learnable": True}}
)
movement.set_policy("citizens", policy)

# Add movement transition
transition = TransitionBuilder()
transition.add_transition(
    "update_position",
    "UpdatePosition",
    {"position": "agents/citizens/position"},
    ["position"],
    {"bounds": {"value": [100.0, 100.0]}}
)
movement.set_transition(transition)

# Add substep to config
config.add_substep("0", movement)
```

### Step 4: Saving and Running

Save the configuration:

```python
config.save_yaml("simulation_config.yaml")
```

Run the simulation:

```python
from agent_torch.core.helpers import read_config
from agent_torch.core import Registry, Runner

# Load config
config = read_config("config.yaml")

# Create registry and runner
registry = Registry()
runner = Runner(config, registry)

# Initialize and run simulation
runner.init()
runner.step(num_steps=10)
```

## Advanced Features

### Property Initialization
You can initialize properties from files or custom functions:

```python
property_builder.set_initialization("read_from_file", {
    "file_path": {"value": "data/values.csv"}
})
```

### Learnable Parameters
Mark parameters that should be optimized during training:

```python
property_builder.set_learnable(True)
```

### Variable References
Use metadata variables in your configuration:

```python
"file_path": {"value": "${simulation_metadata.data_dir}/values.csv"}
```

### Networks
Add agent interaction networks:

```python
network_builder = NetworkBuilder()
network_builder.add_network(
    "social_network",
    "network_from_file",
    {"file_path": "networks/social.csv"}
)
state_builder.set_network(network_builder)
```

## Best Practices

1. **Modular Design**
   - Break simulations into logical substeps
   - Keep substeps focused on single aspects
   - Use clear, descriptive names

2. **Configuration Organization**
   - Group related properties together
   - Use consistent naming conventions
   - Document parameter meanings

3. **Substep Order**
   - Use string numbers ("0", "1", etc.) for substep IDs
   - Order substeps logically
   - Consider dependencies between substeps

## Next Steps

- Check out the [COVID-19 simulation example](../../examples/models/covid/) for a more complex implementation
- Learn about [vectorized operations](../vectorized_operations/) for performance optimization
- Explore [LLM integration](../llm_integration/) for agent behavior

## Reference

For detailed API documentation, see:
- [ConfigBuilder API Reference](../../api/config_builder.md)
- [StateBuilder API Reference](../../api/state_builder.md)
- [SubstepBuilder API Reference](../../api/substep_builder.md) 