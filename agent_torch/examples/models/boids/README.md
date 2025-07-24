# Boids Flocking Model

A classic flocking simulation implemented in AgentTorch, demonstrating emergent collective behavior from simple local rules.

## Overview

The Boids model simulates flocking behavior using three fundamental rules:
1. **Separation**: Avoid crowding neighbors
2. **Alignment**: Steer toward average heading of neighbors  
3. **Cohesion**: Steer toward average position of neighbors

This implementation showcases AgentTorch's Config API and substep architecture for building scalable, differentiable agent-based simulations.

## Quick Start

### 1. Run the Simulation

```bash
# Simple simulation (no visualization)
python -m agent_torch.examples.run_boids_sim_simple

# Full simulation with visualization
python -m agent_torch.examples.run_boids_sim
```

### 2. Expected Output

You should see emergent flocking behavior:
- Boids start with random positions and velocities
- They gradually form cohesive flocks
- Average speed decreases as agents align
- Collective movement patterns emerge

## Model Architecture

### Substeps

The model consists of two main substeps:

1. **FlockingBehavior** (`calculate_flocking_forces`)
   - Calculates separation, alignment, and cohesion forces
   - Handles initial randomization of agent positions/velocities
   - Returns steering forces for each agent

2. **MovementUpdate** (`update_velocity`, `limit_speed`, `update_position`)
   - Applies steering forces to update velocities
   - Limits maximum speed
   - Updates positions with boundary wrapping

### Configuration

The model is defined in `yamls/config.yaml` with:
- **Agents**: 100 boids with position and velocity properties
- **Environment**: Bounds, perception radius, separation distance, max speed/force
- **Learnable Parameters**: Speed range, position margin, flocking weights

## Customization

### Modifying Flocking Behavior

Edit the flocking weights in `calculate_flocking_forces.py`:

```python
self.separation_weight = self.learnable_args.get("separation_weight", 1.5)
self.alignment_weight = self.learnable_args.get("alignment_weight", 1.0)  
self.cohesion_weight = self.learnable_args.get("cohesion_weight", 1.0)
```

### Changing Initialization

Modify the `_initialize_boids` method in `calculate_flocking_forces.py`:

```python
def _initialize_boids(self, state: Dict[str, Any]) -> None:
    # Custom initialization logic here
    # Access learnable parameters via self.speed_range, self.position_margin
```

### Adjusting Simulation Parameters

Update `yamls/config.yaml`:

```yaml
environment:
  properties:
    perception_radius:
      value: 50.0  # How far boids can see neighbors
    separation_distance: 
      value: 25.0  # Minimum distance to maintain
    max_speed:
      value: 2.0   # Maximum agent speed
    max_force:
      value: 0.1   # Maximum steering force
```

## Learnable Parameters

The model supports optimization of:
- `speed_range`: Initial speed distribution
- `position_margin`: Boundary margin for initialization
- `separation_weight`: Strength of separation force
- `alignment_weight`: Strength of alignment force  
- `cohesion_weight`: Strength of cohesion force

These can be optimized via gradient descent for data-driven parameter fitting.

## File Structure

```
boids/
├── README.md                    # This file
├── __init__.py                  # Model registration
├── create_boids_config.py       # Config generation script
├── yamls/
│   └── config.yaml             # Model configuration
└── substeps/
    ├── __init__.py             # Substeps registration
    ├── calculate_flocking_forces.py  # Main flocking logic
    ├── update_velocity.py      # Velocity updates
    ├── limit_speed.py          # Speed limiting
    └── update_position.py      # Position updates
```

## Advanced Usage

### Creating Custom Substeps

1. Define the substep in `create_boids_config.py`:
```python
SubstepBuilderWithImpl(
    name="my_custom_substep",
    observation=SubstepObservation(
        input_variables=["position", "velocity"]
    ),
    policy=SubstepAction(
        input_variables=["position", "velocity"],
        output_variables=["custom_output"]
    ),
    transition=SubstepTransition(
        input_variables=["custom_output"],
        output_variables=["position"]
    )
)
```

2. Implement the logic in `substeps/my_custom_substep.py`

### Integration with Other Models

The boids model can be extended to interact with other agent types by:
- Adding new agent types to the configuration
- Creating interaction substeps between agent types
- Modifying the flocking logic to consider different agent types

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the AgentTorch virtual environment
2. **Memory errors**: Reduce the number of agents in the configuration
3. **No flocking behavior**: Check that perception_radius > separation_distance

### Debug Mode

Add debug prints to substeps to trace execution:

```python
print(f"Debug: Processing {num_agents} agents")
```

## Contributing

When extending the boids model:
1. Follow the existing substep pattern
2. Add learnable parameters where appropriate
3. Update this README with new features
4. Test with different parameter configurations

## References

- [Boids Algorithm](https://en.wikipedia.org/wiki/Boids) - Original flocking model
- [AgentTorch Config API](https://agenttorch.readthedocs.io/) - Framework documentation
- [Emergent Behavior](https://en.wikipedia.org/wiki/Emergence) - Collective intelligence concepts 