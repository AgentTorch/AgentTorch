# Distributed Multi-GPU Simulation

AgentTorch supports distributed simulation across multiple GPUs with **minimal code changes**.

## Quick Start: Only 2 Lines Changed!

Convert any existing AgentTorch simulation to distributed execution:

```python
# BEFORE (Single GPU)
from agent_torch.populations import sample2
from agent_torch.examples.models import movement
from agent_torch.core.environment import envs

runner = envs.create(model=movement, population=sample2)
runner.init()
runner.step(20)
```

```python
# AFTER (Distributed Multi-GPU) - Only 2 lines changed!
from agent_torch.populations import sample2
from agent_torch.examples.models import movement
from agent_torch.core.environment import envs

runner = envs.create(
    model=movement, 
    population=sample2, 
    distributed=True,           # <-- CHANGE 1: Enable distributed
    world_size=4               # <-- CHANGE 2: Specify GPU count
)
runner.init()  # Same as before
runner.step(20)  # Same as before
```

That's it! Your simulation now runs across 4 GPUs automatically.

## Key Features

- **No config files needed** - just change 2 parameters in `envs.create()`
- **Automatic agent partitioning** across GPUs
- **Zero model code changes** required
- **Backward compatible** with all existing models
- **Auto-fallback** to single GPU if needed
- **Same API** - `.init()`, `.step()`, `.reset()` work identically

Scale to **hundreds of** millions of agents seamlessly.

## Usage Examples

### Basic Distributed Simulation
```python
# Use all available GPUs
runner = envs.create(
    model=your_model,
    population=your_population,
    distributed=True  # Auto-detects GPU count
)
```

### Specific GPU Count
```python
# Use exactly 4 GPUs
runner = envs.create(
    model=your_model,
    population=your_population,
    distributed=True,
    world_size=4
)
```

### Custom Sync Settings  
```python
# Advanced: Custom synchronization
distributed_config = {
    "strategy": "data_parallel",
    "sync_frequency": 5  # Sync every 5 steps
}

runner = envs.create(
    model=your_model,
    population=your_population,
    distributed=True,
    world_size=4,
    distributed_config=distributed_config
)
```

## 🎯 Complete Working Example

See `agent_torch/examples/run_movement_sim_distributed.py`:

```python
def run_movement_simulation_distributed(world_size=2):
    """Run movement simulation on multiple GPUs."""
    
    # Only these lines changed from single GPU version:
    runner = envs.create(
        model=movement, 
        population=sample2, 
        distributed=True,           # Enable distributed
        world_size=world_size       # Specify GPU count
    )
    
    # Everything else stays exactly the same:
    runner.init()
    
    sim_steps = runner.config["simulation_metadata"]["num_steps_per_episode"]
    num_episodes = runner.config["simulation_metadata"]["num_episodes"]
    
    for episode in range(num_episodes):
        if episode > 0:
            runner.reset()
        runner.step(sim_steps)
        
        # Print results
        positions = runner.state["agents"]["citizens"]["position"]
        print(f"Average position: {positions.mean(dim=0)}")
```

## 🛠️ How It Works Behind The Scenes

1. **`envs.create()`** detects `distributed=True` and creates a `DistributedRunnerWrapper`
2. **Automatic agent partitioning** - framework splits agents across GPUs
3. **Same API** - wrapper provides identical interface to regular runner
4. **PyTorch multiprocessing** - spawns processes across GPUs automatically
5. **Auto-synchronization** - handles data gathering and state management

## 📝 Requirements

- Multiple CUDA-capable GPUs
- PyTorch with CUDA support  
- Existing AgentTorch model and population

**No additional configuration files or setup needed!**

## 🔧 Advanced Configuration

### Partitioning Strategies
```python
# Data parallelism (default)
distributed_config = {"strategy": "data_parallel"}

# Future: Spatial parallelism
distributed_config = {"strategy": "spatial_parallel"}
```

### Synchronization Control
```python
distributed_config = {
    "sync_frequency": 10,    # Sync every 10 steps
    "compression": "gzip"    # Compress communications
}
```

## 🐛 Troubleshooting

### Check GPU Availability
```python
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
```

### Memory Issues
- Reduce agent count per GPU
- Increase sync frequency to reduce memory usage
- Use gradient checkpointing for large models

### Performance Issues  
- Ensure agent count is divisible by GPU count
- Monitor load balance across GPUs
- Use appropriate sync frequency

## 💡 Migration Guide

Converting existing AgentTorch code is trivial:

1. **Find your `envs.create()` call**
2. **Add `distributed=True`**  
3. **Optionally add `world_size=N`**
4. **Done!**

No other changes needed - your existing model, population, and simulation logic work unchanged. 
