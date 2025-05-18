# AgentTorch Configuration API Tutorial
This tutorial will guide you through using AgentTorch's Configuration API to set up simulations. We'll use a simple movement simulation as our primary example and then explore advanced features.

## Basic Usage: Movement Simulation

Let's create a simulation where agents move randomly within a bounded space. The full code is available [here](../../../agent_torch/examples/models/movement/). This example demonstrates the core concepts of AgentTorch's configuration system.

### Project Structure
```
agent_torch/examples/models/movement/
├── __init__.py
├── substeps/
│   ├── __init__.py
│   ├── random_move.py
│   └── update_position.py
└── yamls/
    └── config.yaml
```

### Setting Up the Configuration

First, import the necessary builders:

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

### 1. Simulation Metadata

Define the basic simulation parameters (note that all these are REQUIRED PARAMETERS):

```python
metadata = {
    "num_agents": 1000,
    "num_episodes": 10,
    "num_steps_per_episode": 20,
    "num_substeps_per_step": 1,
    "device": "cpu",
    "calibration": False # if learnable parameters will be optimized externally [more details in gradient-based calibration tutorial]
}
config.set_metadata(metadata)
```

### 2. State Configuration

Configure the simulation state including agents and environment:

```python
# Build state
state_builder = StateBuilder()

# Add agent with position property
agent_builder = AgentBuilder("citizens", metadata["num_agents"])
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
```

### 3. Substep Configuration

Define the simulation substeps with their policies and transitions:

```python
# Create movement substep
movement = SubstepBuilder("Movement", "Agent movement simulation")
movement.add_active_agent("citizens")
movement.config["observation"] = {"citizens": None}

# Add movement policy
policy = PolicyBuilder()
step_size = PropertyBuilder.create_argument(
    name="Step size parameter",
    value=1.0,
    learnable=True
).config

policy.add_policy(
    "move",
    "RandomMove",
    {"position": "agents/citizens/position"},
    ["direction"],
    {"step_size": step_size}
)
movement.set_policy("citizens", policy)

# Add movement transition
transition = TransitionBuilder()
bounds_param = PropertyBuilder.create_argument(
    name="Environment bounds",
    value=[100.0, 100.0],
    shape=[2],
    learnable=True
).config

transition.add_transition(
    "update_position",
    "UpdatePosition",
    {"position": "agents/citizens/position"},
    ["position"],
    {"bounds": bounds_param}
)
movement.set_transition(transition)

# Add substep to config and save
config.add_substep("0", movement)
config.save_yaml("models/movement/yamls/config.yaml")
```

### 4. Manual Substep Implementation

The substeps can be implemented manually as Python classes. Here's an example of the movement policy:

```python
@Registry.register_substep("move", "policy")
class RandomMove(SubstepAction):
    def forward(self, state: Dict[str, Any], observations) -> Dict[str, Any]:
        positions = get_var(state, self.input_variables["position"])
        num_agents = positions.shape[0]
        
        # Get step size from learnable arguments
        self.step_size = self.learnable_args["step_size"]
        
        # Generate random angles and directions
        angles = torch.rand(num_agents) * 2 * torch.pi
        direction = torch.stack([
            torch.cos(angles) * self.step_size,
            torch.sin(angles) * self.step_size
        ], dim=1)
        
        # Return using output variable name from config
        outputs = {}
        outputs[self.output_variables[0]] = direction
        return outputs
```

### 5. Setting Up Module Structure

The substeps need to be properly imported and registered. This is done through `__init__.py` files:

```python
# models/movement/substeps/__init__.py
"""Substep implementations for movement simulation."""
from .random_move import *
from .update_position import *
```

```python
# models/movement/__init__.py
"""Movement simulation model."""
from agent_torch.core import Registry
from agent_torch.core.helpers import *

from .substeps import *

# Create and populate registry as a module-level variable
registry = Registry()
```

These files ensure that:
1. All substep implementations are imported when the movement module is imported
2. The registry is created at the module level
3. Substeps are automatically registered via their decorators when imported

### 6. Running the Simulation

The movement simulation can be run using:

```python
# run_movement_sim.py
from agent_torch.populations import sample2
from agent_torch.examples.models import movement
from agent_torch.core.environment import envs

def run_movement_simulation():
    """Run the movement simulation."""
    # Create simulation runner
    runner = envs.create(
        model=movement,
        population=sample2
    )
    
    # Get simulation parameters
    sim_steps = runner.config["simulation_metadata"]["num_steps_per_episode"]
    num_episodes = runner.config["simulation_metadata"]["num_episodes"]
    
    # Run episodes
    for episode in range(num_episodes):
        if episode > 0:
            runner.reset()
        runner.step(sim_steps)
        
        # Print statistics
        positions = runner.state["agents"]["citizens"]["position"]
        print(f"Episode {episode + 1} - Average position: {positions.mean(dim=0)}")

if __name__ == "__main__":
    runner = run_movement_simulation()
```

To run the simulation:
```bash
python -m agent_torch.examples.run_movement_sim
```

## Advanced Usage: Automatic Substep Generation

Instead of writing substeps manually, AgentTorch provides `SubstepBuilderWithImpl` to automatically generate implementation templates. Let's see how we could have used it for our movement example:

```python
from agent_torch.config import SubstepBuilderWithImpl

# Create substep with implementation generation
movement_substep = SubstepBuilderWithImpl(
    name="Movement", 
    description="Agent movement simulation",
    output_dir="models/movement/substeps"
)
movement_substep.add_active_agent("citizens")
movement_substep.config["observation"] = {"citizens": None}

# Add same policy and transition configurations
policy = PolicyBuilder()
policy.add_policy(
    "move",
    "RandomMove",
    {"position": "agents/citizens/position"},
    ["direction"],
    {"step_size": step_size}
)
movement_substep.add_policy("citizens", policy)

transition = TransitionBuilder()
transition.add_transition(
    "update_position",
    "UpdatePosition",
    {"position": "agents/citizens/position"},
    ["position"],
    {"bounds": bounds_param}
)
movement_substep.set_transition(transition)

# Generate implementation files
generated_files = movement_substep.generate_implementations()
```

This will generate template files for both `random_move.py` and `update_position.py` with:
- Proper imports and class structure
- Registry decorators
- Input/output variable handling
- TODO sections for implementation logic

You would then only need to fill in the forward logic in the TODO sections.

## More Examples

For a more complex example, check out the COVID-19 simulation in `agent_torch/examples/models/covid/`. This example demonstrates:
- Multiple substeps with complex interactions
- Custom observation and reward functions
- Network-based agent interactions
- Advanced use of learnable parameters

You can run it with:
```bash
python -m agent_torch.examples.run_covid_sim
```

## Best Practices

1. **Metadata**: Always include required fields like `num_agents`, `num_episodes`, `num_steps_per_episode`, and `num_substeps_per_step`.

2. **PropertyBuilder Arguments**: Use `PropertyBuilder.create_argument()` for learnable parameters, and access them in substeps via `self.learnable_args` or `self.arguments` (if not learnable).

3. **Output Variables**: Always use `self.output_variables[index]` as dictionary keys when returning from substep forward methods to ensure consistency with the configuration.

4. **Observation Structure**: When no observation is needed, use `{"citizens": None}` instead of an empty dict.

5. **Implementation Generation**: Use `SubstepBuilderWithImpl` for complex simulations to automatically generate consistent substep templates. This is especially helpful when you have many substeps or want to ensure consistent structure across your codebase. 
