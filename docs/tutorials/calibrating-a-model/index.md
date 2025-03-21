# Calibrating an AgentTorch Model

This tutorial demonstrates how to calibrate parameters in an AgentTorch model using different optimization approaches. We'll explore three methods for parameter optimization and discuss when to use each approach.

## Prerequisites

- Basic understanding of PyTorch and gradient-based optimization
- Familiarity with AgentTorch's basic concepts
- Python 3.8+
- Required packages listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Overview

Model calibration is a crucial step in agent-based modeling. AgentTorch provides several approaches to optimize model parameters:

1. Internal parameter optimization
2. External parameter optimization
3. Generator-based parameter optimization

## Basic Setup

First, let's set up our environment and import the necessary modules:

```python
import warnings
warnings.simplefilter("ignore")

import torch
import torch.nn as nn
from agent_torch.models import covid
from agent_torch.populations import sample
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation

# Initialize simulation
sim = Executor(covid, pop_loader=LoadPopulation(sample))
runner = sim.runner
runner.init()
```

## Helper Classes and Functions

We'll define some helper components that we'll use throughout the tutorial:

```python
class LearnableParams(nn.Module):
    """A neural network module that generates bounded parameters"""
    def __init__(self, num_params, device='cpu'):
        super().__init__()
        self.device = device
        self.num_params = num_params
        self.learnable_params = nn.Parameter(torch.rand(num_params, device=self.device))
        self.min_values = torch.tensor(2.0, device=self.device)
        self.max_values = torch.tensor(3.5, device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        out = self.learnable_params
        # Bound output between min_values and max_values
        out = self.min_values + (self.max_values - self.min_values) * self.sigmoid(out)
        return out

def execute(runner, n_steps=5):
    """Execute simulation and compute loss"""
    runner.step(n_steps)
    labels = runner.state_trajectory[-1][-1]['environment']['daily_infected']
    return labels.sum()
```

## Method 1: Internal Parameter Optimization

The first approach optimizes parameters that are internal to the simulation. This is useful when you want to directly optimize parameters that are part of your model's structure.

```python
def optimize_internal_params():
    # Execute simulation and compute gradients
    loss = execute(runner)
    loss.backward()
    
    # Get gradients of learnable parameters
    learn_params_grad = [(name, param, param.grad) 
                        for (name, param) in runner.named_parameters()]
    return learn_params_grad

# Example usage
gradients = optimize_internal_params()
print("Internal parameter gradients:", gradients)
```

### When to use this method?
- When parameters are naturally part of your simulation structure
- When you want direct control over parameter optimization
- For simpler models with fewer parameters

## Method 2: External Parameter Optimization

The second approach involves optimizing external parameters that are fed into the simulation. This provides more flexibility in parameter management.

```python
def optimize_external_params():
    # Create external parameters
    external_params = nn.Parameter(
        torch.tensor([2.7, 3.8, 4.6], requires_grad=True)[:, None]
    )
    
    # Set parameters in the runner
    learnable_params = runner.named_parameters()
    params_dict = {next(iter(learnable_params))[0]: external_params}
    runner._set_parameters(params_dict)
    
    # Execute and compute gradients
    loss = execute(runner)
    loss.backward()
    return external_params.grad

# Example usage
gradients = optimize_external_params()
print("External parameter gradients:", gradients)
```

### When to use this method?
- When you want to manage parameters outside the simulation
- For parameter sweeps or sensitivity analysis
- When parameters need to be shared across different components

## Method 3: Generator-Based Parameter Optimization

The third approach uses a generator function to predict optimal parameters. This is particularly useful for complex parameter relationships.

```python
def optimize_with_generator():
    # Create generator model
    learn_model = LearnableParams(3)
    params = learn_model()[:, None]
    
    # Execute and compute gradients
    loss = execute(runner)
    loss.backward()
    
    # Get gradients of generator parameters
    learn_params_grad = [(param, param.grad) 
                        for (name, param) in learn_model.named_parameters()]
    return learn_params_grad

# Example usage
gradients = optimize_with_generator()
print("Generator parameter gradients:", gradients)
```

### When to use this method?
- When parameters have complex relationships
- For learning parameter patterns
- When you want to generate parameters based on conditions

## Putting It All Together

Here's how to use all three methods in a complete optimization loop:

```python
def calibrate_model(method='internal', num_epochs=10):
    for epoch in range(num_epochs):
        if method == 'internal':
            gradients = optimize_internal_params()
        elif method == 'external':
            gradients = optimize_external_params()
        else:  # generator
            gradients = optimize_with_generator()
            
        print(f"Epoch {epoch}, gradients: {gradients}")
        # Add your optimizer step here

# Example usage
calibrate_model(method='internal', num_epochs=3)
```

## Best Practices

1. **Choose the Right Method**: Consider your specific use case when selecting an optimization approach.
2. **Monitor Convergence**: Always track your loss function to ensure proper optimization.
3. **Validate Results**: Cross-validate your calibrated parameters with held-out data.
4. **Handle Constraints**: Use appropriate bounds and constraints for your parameters.

## Common Pitfalls

- Ensure parameters have appropriate ranges
- Watch out for local optima
- Be careful with learning rates in optimization
- Consider the computational cost of each approach

## Conclusion

We've explored three different approaches to model calibration in AgentTorch. Each method has its strengths and is suited for different scenarios. Choose the approach that best matches your specific needs and model complexity.

## Additional Resources

- [AgentTorch Documentation](https://agent-torch.ai/)
- [PyTorch Optimization](https://pytorch.org/docs/stable/optim.html)
- [Related Tutorials](../index.md)
