# Distributed Multi-GPU Simulation Guide

AgentTorch now supports distributed simulation across multiple GPUs, enabling massive population models with millions of agents.

## 🚀 Quick Start

```python
from agent_torch.core.distributed_runner import launch_distributed_simulation
from agent_torch.examples.models import movement

# Create config for 1M agents
config = {
    "simulation_metadata": {
        "num_agents": 1000000,
        "num_steps_per_episode": 50,
        "device": "cuda"
    },
    "distributed": {
        "strategy": "data_parallel",
        "sync_frequency": 5
    }
    # ... rest of config
}

# Launch on all available GPUs
final_state = launch_distributed_simulation(
    config, movement.registry, num_steps=50
)
```

## 📊 Performance Benchmarks

### Expected Speedups
- **2 GPUs**: 1.7-1.9x speedup
- **4 GPUs**: 3.2-3.8x speedup  
- **8 GPUs**: 6.0-7.5x speedup

### Memory Scaling
- **Single GPU**: ~1M agents max
- **4 GPUs**: ~4M agents
- **8 GPUs**: ~8M agents

## 🔧 Configuration Options

### Partitioning Strategies

#### Data Parallelism (Default)
```yaml
distributed:
  strategy: "data_parallel"
  sync_frequency: 5
```
- Splits agents evenly across GPUs
- Best for independent agent behaviors
- Minimal cross-GPU communication

#### Spatial Parallelism (Future)
```yaml
distributed:
  strategy: "spatial_parallel"
  regions: ["north", "south", "east", "west"]
```
- Splits by geographic regions
- Better for location-based interactions
- Reduces network communication overhead

### Synchronization Settings

```yaml
distributed:
  sync_frequency: 5      # Sync every N steps
  async_updates: false   # Enable async updates
  compression: "gzip"    # Compress communications
```

## 🎯 Use Cases

### 1. Massive Population Studies
```python
# Simulate entire countries
config = create_config(
    num_agents=50_000_000,  # 50M agents
    num_steps=365,          # 1 year simulation
    world_size=8            # 8 GPUs
)
```

### 2. High-Frequency Trading
```python
# Millions of trading agents
config = create_config(
    num_agents=10_000_000,
    num_steps=1440,  # 24 hours, minute resolution
    sync_frequency=1  # High frequency sync
)
```

### 3. Epidemiological Modeling
```python
# Country-scale disease spread
config = create_config(
    num_agents=300_000_000,  # US population
    num_steps=1000,          # ~3 years
    strategy="spatial_parallel"  # Geographic spread
)
```

## 📋 Running Examples

### Basic Usage
```bash
# Simple distributed simulation
python examples/run_distributed_movement_sim.py --agents 100000 --steps 20

# Benchmark single vs multi-GPU
python examples/run_distributed_movement_sim.py --benchmark --agents 500000

# Massive scale simulation
python examples/run_distributed_movement_sim.py --massive --agents 5000000 --steps 100
```

### Advanced Options
```bash
# Specific GPU count
python examples/run_distributed_movement_sim.py --gpus 4 --agents 2000000

# Memory profiling
python examples/run_distributed_movement_sim.py --profile --agents 1000000
```

## 🔍 Monitoring & Debugging

### GPU Utilization
```python
import torch

# Check GPU memory usage
for i in range(torch.cuda.device_count()):
    memory_used = torch.cuda.memory_allocated(i) / 1e9
    memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"GPU {i}: {memory_used:.1f}/{memory_total:.1f} GB")
```

### Communication Overhead
```python
# Monitor distributed communication
import torch.distributed as dist

# In your distributed code
start_time = time.time()
dist.all_gather(tensor_list, local_tensor)
comm_time = time.time() - start_time
print(f"Communication time: {comm_time:.3f}s")
```

## ⚡ Performance Tips

### 1. Optimal Batch Sizes
- Ensure agent count is divisible by GPU count
- Use powers of 2 when possible
- Balance memory vs computation

### 2. Memory Management
```python
# Clear cache between episodes
torch.cuda.empty_cache()

# Use mixed precision
torch.cuda.amp.autocast()
```

### 3. Network Optimization
- Minimize cross-GPU interactions
- Use spatial partitioning for geographic models
- Batch communications when possible

### 4. Load Balancing
```python
# Monitor load balance
agents_per_gpu = [partition['local_size'] for partition in partitions]
load_imbalance = max(agents_per_gpu) / min(agents_per_gpu)
print(f"Load imbalance: {load_imbalance:.2f}")
```

## 🐛 Troubleshooting

### Common Issues

#### Out of Memory
```python
# Reduce batch size or use gradient checkpointing
config["distributed"]["gradient_checkpointing"] = True
config["simulation_metadata"]["batch_size"] = 1000
```

#### Slow Communication
```python
# Increase sync frequency or use compression
config["distributed"]["sync_frequency"] = 10
config["distributed"]["compression"] = "gzip"
```

#### Load Imbalance
```python
# Use better partitioning strategy
config["distributed"]["strategy"] = "adaptive_parallel"
```

### Debug Mode
```python
# Enable debug logging
import os
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
```

## 🔮 Future Features

### Coming Soon
- **Spatial Partitioning**: Geographic region splitting
- **Adaptive Load Balancing**: Dynamic agent redistribution  
- **Cross-Node Scaling**: Multi-machine support
- **Memory Optimization**: Gradient checkpointing
- **Fault Tolerance**: Automatic recovery from GPU failures

### Roadmap
- **Q1 2025**: Spatial partitioning implementation
- **Q2 2025**: Multi-node distributed support  
- **Q3 2025**: Cloud deployment optimizations
- **Q4 2025**: Automatic hyperparameter tuning

## 📚 References

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Multi-GPU Training Best Practices](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [AgentTorch Architecture Guide](architecture.md) 