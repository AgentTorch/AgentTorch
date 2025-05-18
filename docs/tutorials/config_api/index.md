# Building Simulations with the Configuration API

This tutorial will guide you through using AgentTorch's Configuration API to build agent-based simulations. We'll create a complete simulation including configuration, implementation, and execution.

## Prerequisites

Before starting this tutorial, make sure you have:
- AgentTorch installed (`pip install agent-torch`)
- Basic understanding of agent-based modeling concepts
- Python 3.10 or higher

## Project Structure

A typical AgentTorch simulation project has the following structure:

```
my_simulation/
├── __init__.py           # Registry setup
├── substeps/             # Substep implementations
│   ├── __init__.py
│   ├── movement.py
│   └── collision.py
├── yamls/               # Configuration files
│   └── config.yaml
└── run.py              # Experiment runner
```

## Creating the Configuration

First, let's create a simple movement simulation where agents move randomly within bounded space.

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

def setup_simulation():
    """Setup a complete simulation structure."""
    
    # Create config builder
    config = ConfigBuilder()
    
    # Set simulation metadata
    metadata = {
        "num_agents": 100,
        "num_episodes": 10,
        "num_steps_per_episode": 20,
        "device": "cpu"
    }
    config.set_metadata(metadata)
    
    # Build state
    state_builder = StateBuilder()
    
    # Add agent
    agent_builder = AgentBuilder("citizens", metadata["num_agents"])
    
    # Add position property
    position = PropertyBuilder("position")\
        .set_dtype("float")\
        .set_shape([metadata["num_agents"], 2])\
        .set_value([0.0, 0.0])
    
    agent_builder.add_property(position)
    state_builder.add_agent("citizens", agent_builder)
    
    # Add environment bounds
    env_builder = EnvironmentBuilder()
    bounds = PropertyBuilder("bounds")\
        .set_dtype("float")\
        .set_shape([2])\
        .set_value([100.0, 100.0])
    env_builder.add_variable(bounds)
    state_builder.set_environment(env_builder)
    
    # Set state in config
    config.set_state(state_builder.to_dict())
    
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
    
    return config

# Create and save config
config = setup_simulation()
config.save_yaml("yamls/config.yaml")
```

## Setting Up the Registry

Create an `__init__.py` in your simulation directory to set up the registry:

```python
"""Movement simulation model."""
from agent_torch.core import Registry
from agent_torch.core.helpers import *

from .substeps import *

# Create and populate registry as a module-level variable
registry = Registry()
```

## Implementing Substeps

### Random Movement Policy

Create `substeps/random_move.py`:

```python
from typing import Dict, Any
import torch
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers import get_var

@Registry.register_substep("move", "policy")
class RandomMove(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize learnable parameters
        for key, value in kwargs.items():
            if isinstance(value, dict) and value.get('learnable', False):
                setattr(self, key, value['value'])
    
    def forward(self, state: Dict[str, Any], observations) -> Dict[str, Any]:
        """Generate random movement directions."""
        # Get current positions
        positions = get_var(state, self.input_variables["position"])
        num_agents = positions.shape[0]
        
        # Generate random angles
        angles = torch.rand(num_agents) * 2 * torch.pi
        
        # Convert to direction vectors
        directions = torch.stack([
            torch.cos(angles) * self.step_size,
            torch.sin(angles) * self.step_size
        ], dim=1)
        
        return {"direction": directions}
```

### Position Update Transition

Create `substeps/update_position.py`:

```python
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
            if isinstance(value, dict) and value.get('learnable', False):
                setattr(self, key, value['value'])
    
    def forward(self, state: Dict[str, Any], action) -> Dict[str, Any]:
        """Update agent positions based on movement directions."""
        # Get current positions and movement directions
        positions = get_var(state, self.input_variables["position"])
        directions = action["direction"]
        bounds = torch.tensor(self.bounds)
        
        # Update positions
        new_positions = positions + directions
        
        # Clip to bounds
        new_positions = torch.clamp(new_positions, min=0, max=bounds)
        
        return {"position": new_positions}
```

### Substeps Init

Create `substeps/__init__.py`:

```python
"""Substep implementations for movement simulation."""
from .random_move import *
from .update_position import *
```

## Running the Simulation

Create `run.py`:

```python
from agent_torch.core.environment import envs
from agent_torch.populations import sample  # Or your own population loader

def run_simulation():
    """Run the movement simulation."""
    print("\n=== Running Movement Simulation ===")
    
    # Create the runner
    print("\nCreating simulation runner...")
    runner = envs.create(
        model="path.to.your.model",  # Update with your model's import path
        population=sample
    )
    
    # Get simulation parameters
    sim_steps = runner.config["simulation_metadata"]["num_steps_per_episode"]
    num_episodes = runner.config["simulation_metadata"]["num_episodes"]
    
    print(f"\nSimulation parameters:")
    print(f"- Steps per episode: {sim_steps}")
    print(f"- Number of episodes: {num_episodes}")
    print(f"- Number of agents: {runner.config['simulation_metadata']['num_agents']}")
    
    # Run all episodes
    print("\nRunning simulation episodes...")
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        # Reset state at the start of each episode
        if episode > 0:
            runner.reset()
        
        # Run one episode
        runner.step(sim_steps)
        
        # Print statistics
        final_state = runner.state
        positions = final_state["agents"]["citizens"]["position"]
        print(f"- Average position: {positions.mean(dim=0)}")
    
    print("\nSimulation completed!")
    return runner

if __name__ == "__main__":
    runner = run_simulation()
```

## Running the Experiment

To run your simulation:

```bash
python -m my_simulation.run
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

1. **Project Organization**
   - Keep substeps in separate files
   - Use clear, descriptive names for substeps
   - Maintain a clean registry structure

2. **Configuration Management**
   - Group related properties together
   - Use consistent naming conventions
   - Document parameter meanings
   - Store configurations in YAML files

3. **Substep Implementation**
   - Handle tensor operations efficiently
   - Use vectorized operations where possible
   - Properly register substeps with the registry
   - Document input/output variables

4. **Experiment Running**
   - Create dedicated run scripts
   - Add proper logging and statistics
   - Handle episode resets correctly
   - Save experiment results

## Next Steps

- Check out the [COVID-19 simulation example](../../examples/models/covid/) for a more complex implementation
- Explore [LLM integration](../creating-archetypes/) for agent behavior 