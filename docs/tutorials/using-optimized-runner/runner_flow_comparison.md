## Runner Flow: Basic vs Optimized

This document compares a basic Runner usage pattern with an optimized pattern and highlights added optimizations and functions in the current Runner implementation.

### Summary of Differences

- **Device awareness:**
  - Basic: CPU-oriented, no device attribute.
  - Optimized: Detects CUDA (`use_gpu`), sets `self.device`, and routes to GPU/CPU paths.

- **Initialization:**
  - Both: use `Initializer.initialize()` and record an initial CPU snapshot via `to_cpu(self.state)`.
  - Optimized: wires a pooled buffer allocator into transition modules on CUDA (`_wire_transition_buffer_allocator`).

- **Step execution:**
  - Basic: single `step(...)` loop with per-substep observe → act → progress, always snapshot to CPU.
  - Optimized: dispatches to `_step_cpu_base` (same as basic) or `_step_gpu_optimized` with vectorization, memory pooling, and reduced transfers.

- **Performance instrumentation:**
  - Basic: none.
  - Optimized: `perf_stats` (gpu_to_cpu_transfers, tensor_allocations, memory_reused, vectorized_operations) and `get_performance_stats()` helper.

- **Parameter updates:**
  - Basic: `_set_parameters` replaces tensors directly.
  - Optimized: `_set_parameters` ensures replacement tensors are moved to the correct device.

### Basic pattern (minimal CPU)

```python
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation

from agent_torch.models import covid
from agent_torch.populations import astoria

pop_loader = LoadPopulation(astoria)
simulation = Executor(model=covid, pop_loader=pop_loader)
runner = simulation.runner

runner.init()
runner.step(runner.config["simulation_metadata"]["num_steps_per_episode"])
```

### Optimized pattern (auto-selects GPU Runner, timings, stats)

```python
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation, DataLoader
from agent_torch.models import covid
from agent_torch.populations import astoria
import time

# Setup
t0 = time.perf_counter()
pop_loader = LoadPopulation(astoria)
dl = DataLoader(covid, pop_loader)  # lets Executor pick optimized Runner when available
simulation = Executor(model=covid, data_loader=dl)
runner = simulation.runner

# Time init
t_init_start = time.perf_counter()
runner.init()
t_init_end = time.perf_counter()

print(f"\n Init timings: runner.init()={t_init_end - t_init_start:.3f}s, total={t_init_end - t0:.3f}s")

# Simulate and optionally print perf stats
num_steps = runner.config["simulation_metadata"]["num_steps_per_episode"]
runner.step(num_steps)
if hasattr(runner, 'get_performance_stats'):
    stats = runner.get_performance_stats()
    print("\nPerformance Stats:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
```

### Device‑safe parameter updates

```python
# Ensure new tensors match runner.device when setting parameters directly
device = runner.device
new_tensor = torch.tensor([3.5, 4.2, 5.6], requires_grad=True, device=device)

param_path = "initializer.transition_function.0.new_transmission.learnable_args.R2"
runner._set_parameters({param_path: new_tensor})
```

### Added Optimizations (GPU path)

1. `use_gpu` flag, CUDA params, and dedicated snapshot stream
   - Fields: `_snapshot_stream`, `_batch_size`, `_pool_limit_per_shape`, `_inplace_progress`
   - Purpose: manage memory and transfers efficiently on GPU

2. Memory pooling and reuse
   - `_get_pooled_tensor`, `_return_to_pool`
   - Reduces allocations; tracks `memory_reused` and `tensor_allocations`

3. Vectorized substep processing
   - `_process_substep_vectorized`, `_gpu_optimized_agent_processing`
   - Batches observation/action updates over active indices; increments `vectorized_operations`

4. Active-set detection
   - `_compute_active_indices` identifies which agents to process (e.g., by disease stage)

5. Snapshot compression
   - `_compress_state_for_snapshot` downcasts/normalizes tensors, uses non-blocking transfers on a CUDA stream

6. Optimized progress
   - `_progress_state_optimized` placeholder for in-place/vectorized transitions (currently calls controller.progress)

### Function-by-Function Mapping (current Runner)

- `init()` → initialize and record initial CPU snapshot; wire buffer allocator on CUDA
- `step(num_steps=None)` → dispatch to CPU base or GPU optimized path
- `_step_cpu_base(...)` → mirrors the basic loop
- `_step_gpu_optimized(...)` → vectorization, pooling, snapshot compression, perf counters
- `_process_substep_vectorized(...)`, `_gpu_optimized_agent_processing(...)` → batched observe/act
- `_observe_with_batches(...)`, `_act_with_batches(...)`, `_process_tensor_active_batched(...)` → active-index updates and reuse
- `_wire_transition_buffer_allocator()` → pooled buffer hooks for transitions
- `_get_pooled_tensor(...)`, `_return_to_pool(...)` → memory pool
- `get_performance_stats()` → perf summary (GPU) or mode (CPU)

### When to use the optimized pattern

- Large populations (tens of thousands of agents)
- CUDA available (GPU compute)
- Need basic performance stats with minimal code changes

## Using the Optimized Runner

This guide shows how to use the GPU‑optimized Runner demonstrated in `example_v2.py`, and how it differs from the basic usage in `example.py`.

### Overview

- **example.py (basic):**
  - Loads a population via `LoadPopulation`
  - Constructs `Executor(model, pop_loader=...)`
  - Calls `runner.init()` and then `runner.step(...)`

- **example_v2.py (optimized):**
  - Measures init and step timings
  - Uses `DataLoader(model, pop_loader)` to let AgentTorch pick an optimized Runner (CUDA when available)
  - Prints performance statistics when supported

### Optimized setup (from example_v2.py)

```python
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation, DataLoader

from agent_torch.models import covid
from agent_torch.populations import astoria

import time

def setup(model, population):
    # Timers
    t0 = time.perf_counter()

    # Executor + DataLoader → auto-detect optimized backend (e.g., CUDA)
    t_loader_start = time.perf_counter()
    pop_loader = LoadPopulation(population)
    dl = DataLoader(model, pop_loader)
    simulation = Executor(model=model, data_loader=dl)
    t_loader_end = time.perf_counter()
    
    runner = simulation.runner

    # Time runner.init()
    t_init_start = time.perf_counter()
    runner.init()
    t_init_end = time.perf_counter()

    # Print init timing summary
    loader_exec_s = t_loader_end - t_loader_start
    runner_init_s = t_init_end - t_init_start
    total_init_s = t_init_end - t0
    print(f"\n Init timings: loader+executor={loader_exec_s:.3f}s, runner.init()={runner_init_s:.3f}s, total={total_init_s:.3f}s")

    return runner
```

### Running a simulation step and collecting performance stats

```python
def simulate(runner):
    num_steps_per_episode = runner.config["simulation_metadata"]["num_steps_per_episode"]
    
    runner.step(num_steps_per_episode)
    traj = runner.state_trajectory[-1][-1]
    preds = traj["environment"]["daily_infected"]
    loss = preds.sum()
    
    # Print performance stats if available
    if hasattr(runner, 'get_performance_stats'):
        stats = runner.get_performance_stats()
        print(f"\nPerformance Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    return loss
```

### Device‑safe parameter updates

When setting learnable parameters directly on substeps, ensure new tensors are on the same device as the Runner.

```python
runner = setup(covid, astoria)

# Get a device‑correct tensor
device = runner.device
new_tensor = torch.tensor([3.5, 4.2, 5.6], requires_grad=True, device=device)

# Example: map a dotted path to a learnable parameter and set it
input_string = "initializer.transition_function.0.new_transmission.learnable_args.R2"
params_dict = {input_string: new_tensor}
runner._set_parameters(params_dict)
```

### End‑to‑end script timing

`example_v2.py` also measures total script time and per‑section timings to help profile initialization and simulation step performance.

```python
if __name__ == "__main__":
    script_t0 = time.perf_counter()
    runner = setup(covid, astoria)

    sim_t0 = time.perf_counter()
    loss = simulate(runner)
    sim_t1 = time.perf_counter()
    print(f"\nLoss: {loss}")

    script_t1 = time.perf_counter()
    print(f"\n Timings: simulation_step={sim_t1 - sim_t0:.3f}s, script_total={script_t1 - script_t0:.3f}s")
```

### Key differences vs basic example

- Uses `DataLoader` so the `Executor` can choose the best Runner automatically (e.g., GPU‑accelerated).
- Tracks timings around loader/executor creation and `runner.init()`.
- Optionally prints performance stats via `runner.get_performance_stats()`.
- Ensures tensors are created on `runner.device` when directly setting parameters.

Use this pattern when you want the fastest available execution and basic profiling hooks with minimal changes to your workflow.


