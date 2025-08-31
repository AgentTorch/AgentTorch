## Example v2: Runtime optimizations explained

This note documents the performance-minded choices used in `example_v2.py`, why they help, and how they reduce end‑to‑end time.

### 1) Auto device selection (CPU vs CUDA)

- What: The `Executor` auto-detects CUDA and instantiates the appropriate runner.
- Why it helps: Utilizes GPU kernels for heavy tensor work without manual configuration.
- How it helps: Large state updates and transitions run on the GPU where available, improving throughput.

```100:106:example_v2.py
    runner = setup(covid, astoria)
    learn_params = [(name, params) for (name, params) in runner.named_parameters()]
    # Ensure new tensor is on the same device as the runner
    device = runner.device
```

### 2) Minimize host↔device transfers (create tensors on runner.device)

- What: New tensors (e.g., learnable parameters) are created directly on `runner.device`.
- Why it helps: Avoids implicit CPU→GPU copies that add latency and can fragment GPU memory.
- How it helps: Keeps data resident on the device where compute happens.

```105:113:example_v2.py
    device = runner.device
    new_tensor = torch.tensor([3.5, 4.2, 5.6], requires_grad=True, device=device)
    input_string = learn_params[0][0]
    input_string = "initializer.transition_function.0.new_transmission.learnable_args.R2"
    params_dict = {input_string: new_tensor}
    runner._set_parameters(params_dict)
```

### 3) Batch stepping (reduce Python overhead)

- What: Call `runner.step(num_steps_per_episode)` once instead of stepping inside a Python loop.
- Why it helps: Minimizes Python overhead and lets the runner fuse/sequence operations internally.
- How it helps: The inner loop runs in optimized code paths (vectorized kernels / graph where applicable).

```79:88:example_v2.py
def simulate(runner):
    num_steps_per_episode = runner.config["simulation_metadata"]["num_steps_per_episode"]
    runner.step(num_steps_per_episode)
    traj = runner.state_trajectory[-1][-1]
    preds = traj["environment"]["daily_infected"]
    loss = preds.sum()
    ...
    return loss
```

### 4) Single-pass loss aggregation on-device

- What: Compute `loss = preds.sum()` directly from the final trajectory slice.
- Why it helps: Avoids copying intermediate results back to the CPU and re-iterating in Python.
- How it helps: Keeps reductions in optimized tensor ops.

```84:88:example_v2.py
    traj = runner.state_trajectory[-1][-1]
    preds = traj["environment"]["daily_infected"]
    loss = preds.sum()
```

### 5) Targeted parameter updates (no per-parameter Python loops)

- What: Build a small `params_dict` and call `runner._set_parameters` once.
- Why it helps: Fewer Python attribute lookups and dynamic dispatches.
- How it helps: Consolidates mutation into a single call; avoids repeated crossing of Python↔backend boundaries.

```109:113:example_v2.py
    input_string = "initializer.transition_function.0.new_transmission.learnable_args.R2"
    params_dict = {input_string: new_tensor}
    runner._set_parameters(params_dict)
```

### 6) Coarse-grained timing with `perf_counter`

- What: Time loader/executor construction, `runner.init()`, and simulation separately.
- Why it helps: Identifies the real bottlenecks (I/O vs. initialization vs. step kernel time) quickly.
- How it helps: Guides where to invest optimization effort (e.g., caching data vs. kernel fusion).

```52:76:example_v2.py
def setup(model, population):
    t0 = time.perf_counter()
    t_loader_start = time.perf_counter()
    pop_loader = LoadPopulation(population)
    dl = DataLoader(model, pop_loader)
    simulation = Executor(model=model, data_loader=dl)
    t_loader_end = time.perf_counter()
    runner = simulation.runner
    t_init_start = time.perf_counter()
    runner.init()
    t_init_end = time.perf_counter()
    ...
    return runner
```

### 7) Optional kernel/perf telemetry

- What: Query `runner.get_performance_stats()` if available.
- Why it helps: Surfaces GPU/CPU timing breakdowns from the runtime without adding profiling overhead everywhere.
- How it helps: Quick feedback loop for tuning step sizes, batch shapes, or device placement.

```89:96:example_v2.py
    if hasattr(runner, 'get_performance_stats'):
        stats = runner.get_performance_stats()
        print("\nPerformance Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
```

### 8) DataLoader/LoadPopulation pre-processing (I/O efficiency)

- What: Use `LoadPopulation` + `DataLoader` to encapsulate population I/O and preprocessing.
- Why it helps: Centralizes data loading, caching, and potential format conversions away from the hot loop.
- How it helps: Reduces per-run I/O and keeps the simulation loop compute-bound.

```56:61:example_v2.py
    pop_loader = LoadPopulation(population)
    dl = DataLoader(model, pop_loader)
    simulation = Executor(model=model, data_loader=dl)
```

---

In combination, these choices keep compute on the right device, reduce Python overhead, avoid unnecessary transfers, and surface timing where it matters—yielding faster, more predictable runs.

## Engine‑level optimizations (Runner/Executor/Initializer)

Beyond the simple loop patterns above, the runtime includes several built‑in optimizations that improve scaling and latency:

### Runner (CUDA path)
- GPU streams and async transfers: overlaps minimal snapshot I/O with compute, reducing stalls.
- Memory pooling and tensor reuse: cuts allocator churn and fragmentation; fewer device alloc/free cycles.
- Vectorized processing hooks: processes many agents in batched ops rather than Python loops.
- Active‑set batching: updates only relevant agents (e.g., exposed/infected), keeping per‑step cost proportional to active agents.
- Minimal/env‑only snapshots: bounds trajectory payload; optional downcasting (e.g., int64→int32, bool→uint8).
- Optional mixed precision: leverages fp16/bf16 where profitable to increase throughput.

Why it helps: less time in the Python interpreter and allocator, fewer/lighter device transfers, and more FLOPs devoted to the subset of agents that actually changed.

### VectorizedRunner
- Detects and uses vectorized implementations (e.g., via vmap) when available, falling back otherwise.
- Reduces host overhead and enables fused batched kernels for observation/policy/transition.

Why it helps: shifts work from many small ops to fewer larger ops, improving arithmetic intensity and kernel efficiency.

### Initializer and data residency
- Single device move at init: tensor leaves are moved once; subsequent steps keep data resident on GPU.
- Dedicated snapshot stream: subsequent snapshots are async and minimal rather than full copies.

Why it helps: avoids repeated CPU↔GPU shuttling and keeps the hot path compute‑bound.

### Utils and graph representation (GPU path)
- Sparse edge representations instead of dense adjacency matrices in contact networks.

Why it helps: lowers memory footprint and avoids O(N^2) dense ops, improving performance on large populations.

### Perf scaffolding
- Built‑in counters/timers (allocations, reuses, vectorized ops, transfer counts) and `get_performance_stats()`.

Why it helps: fast feedback to tune batch sizes, active‑set thresholds, and snapshot frequency without full profilers.

### Evidence of impact
- See `analyze_performance/perf-astoria-cuda-vs-cpu.md`: CUDA + optimized utils achieve ~71.6× total speedup vs CPU base for Astoria (37,518 agents), with step time ~138× faster. Scaling to NYC shows good efficiency with much larger populations.


