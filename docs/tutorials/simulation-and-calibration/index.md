---
title: Creating and Calibrating a Simulation
description: Learn how to create a simulation instance and integrate it with calibration logic
---

# Creating and Calibrating a Simulation

This tutorial will guide you through the process of creating a simulation instance and integrating it with calibration logic using AgentTorch. We'll cover:

1. Setting up a basic simulation
2. Configuring simulation parameters
3. Running the simulation
4. Integrating with calibration
5. Advanced parameter tuning

## Prerequisites

Before starting this tutorial, make sure you have:

- AgentTorch installed (`pip install agent-torch`)
- Basic understanding of Python and PyTorch
- Your model and population data ready

## Basic Simulation Setup

First, let's create a basic simulation instance. Here's a minimal example:

```python
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation
from agent_torch.models import covid  # Example model
from agent_torch.populations import sample2  # Example population

def setup_simulation(model, population):
    # Create a population loader
    loader = LoadPopulation(population)
    
    # Initialize the simulation executor
    simulation = Executor(model=model, pop_loader=loader)
    
    # Get the runner instance
    runner = simulation.runner
    
    # Initialize the simulation
    runner.init()
    
    return runner

# Create the simulation instance
runner = setup_simulation(covid, sample2)
```

## Configuring Parameters

You can configure simulation parameters in two ways:

1. During initialization:
```python
simulation_config = {
    'simulation_metadata': {
        'num_steps_per_episode': 100,
        'num_episodes': 1
    }
}
runner = setup_simulation(covid, sample2, config=simulation_config)
```

2. After initialization using the parameter API:
```python
def set_parameter(runner, param_path, new_value):
    """
    param_path: String path to the parameter (e.g., 'initializer.transition_function.0.new_transmission.learnable_args.R0')
    new_value: New tensor value for the parameter
    """
    params_dict = {param_path: new_value}
    runner._set_parameters(params_dict)
```

## Running the Simulation

To run the simulation:

```python
def run_simulation(runner):
    # Get simulation parameters
    num_steps = runner.config['simulation_metadata']['num_steps_per_episode']
    
    # Run simulation steps
    runner.step(num_steps)
    
    # Get final trajectory
    final_trajectory = runner.state_trajectory[-1][-1]
    
    return final_trajectory

# Run simulation and get results
results = run_simulation(runner)
```

## Integrating with Calibration

For calibration, we need to:
1. Define parameters to calibrate
2. Create a loss function
3. Set up optimization

Here's how to do it:

```python
import torch
import torch.optim as optim

def setup_calibration(runner):
    # Get learnable parameters
    learn_params = [(name, params) for (name, params) in runner.named_parameters()]
    
    # Create optimizer
    optimizer = optim.Adam(runner.parameters(), lr=0.01)
    
    return optimizer

def calibration_step(runner, optimizer, target_data):
    # Zero gradients
    optimizer.zero_grad()
    
    # Run simulation
    trajectory = run_simulation(runner)
    
    # Calculate loss (example using infected counts)
    preds = trajectory['environment']['daily_infected']
    loss = torch.nn.functional.mse_loss(preds, target_data)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    return loss.item()

# Setup calibration
optimizer = setup_calibration(runner)

# Run calibration loop
for epoch in range(100):
    loss = calibration_step(runner, optimizer, target_data)
    print(f"Epoch {epoch}, Loss: {loss}")
```

## Advanced Parameter Tuning

For more fine-grained control over parameters:

```python
def get_parameter(runner, param_path):
    """Get current parameter value"""
    tensor_func = map_and_replace_tensor(param_path)
    return tensor_func(runner)

def update_parameter(runner, param_path, new_value):
    """Update specific parameter with gradient tracking"""
    assert isinstance(new_value, torch.Tensor)
    assert new_value.requires_grad
    set_parameter(runner, param_path, new_value)
```

## Best Practices

1. Always validate parameter changes:
   ```python
   def validate_parameter(runner, param_path, new_value):
       current_value = get_parameter(runner, param_path)
       assert new_value.shape == current_value.shape, "Shape mismatch"
       assert new_value.requires_grad == current_value.requires_grad, "Gradient requirement mismatch"
   ```

2. Save and load calibrated parameters:
   ```python
   def save_parameters(runner, filepath):
       torch.save(runner.state_dict(), filepath)
   
   def load_parameters(runner, filepath):
       runner.load_state_dict(torch.load(filepath))
   ```

## Common Issues and Solutions

1. **Gradient Issues**: If you encounter gradient-related errors, ensure all parameters that need gradients have `requires_grad=True`.

2. **Memory Management**: For large simulations, consider using:
   ```python
   def clear_memory(runner):
       runner.state_trajectory = []  # Clear trajectory history
       torch.cuda.empty_cache()  # If using GPU
   ```

3. **Parameter Bounds**: Implement parameter constraints:
   ```python
   def constrain_parameters(runner, param_path, min_val, max_val):
       value = get_parameter(runner, param_path)
       constrained_value = torch.clamp(value, min_val, max_val)
       set_parameter(runner, param_path, constrained_value)
   ```

## Next Steps

- Explore more advanced calibration techniques in the [Calibration Advanced Guide](../calibrating-a-model/advanced.md)
- Learn about analyzing simulation results in the [Simulation Analysis Tutorial](../using-simulation-analyzer/index.md)
- Understand how to integrate custom models in the [Custom Models Guide](../creating-a-model/index.md) 