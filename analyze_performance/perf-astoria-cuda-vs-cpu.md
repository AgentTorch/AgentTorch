## Astoria: CUDA  vs CPU

### Run setup
- Population: Astoria (37,518 agents)
- Steps: 21 (per-substep snapshots)

### Summary metrics

| Run | Init (loader+exec / runner.init) | Avg/step | Total | Notes |
|---|---|---:|---:|---|
| CUDA  | 0.193s / 0.132s | 0.011s | 0.232s | Env-only compressed snapshots; async GPU→CPU |
| CPU  | 0.211s / 14.547s | 1.609s | 33.792s | Full-state CPU snapshots each substep |

### Step timings (selected)

| Step | CUDA (s) | CPU (s) |
|---:|---:|---:|
| 1 | 0.125 | 1.581 |
| 6 | 0.006 | 1.661 |
| 11 | 0.006 | 1.332 |
| 16 | 0.006 | 1.820 |
| 21 | 0.006 | 1.778 |


### Why CUDA times can show small step‑to‑step variance
- Snapshot steps perform minimal env‑only transfers (fp32), while non‑snapshot steps keep GPU‑resident references only.
- CUDA autotuning/allocator warming in early steps can cause minor variation.


### What changed: CUDA vs Base
- Device management: single boolean gate (`use_gpu`) with device moves at init; optional mixed precision (autocast) on CUDA.
- Trajectory handling:
  - Per‑substep snapshots retained (parity with base) but payload minimized (env‑only by default).
  - Compression/compaction: keep env floats in fp32; downcast int64→int32; bool→uint8.
  - Async, non‑blocking GPU→CPU copies on a dedicated CUDA stream; bounded history via ring buffer.
- Performance scaffolding: step timings, counters (GPU→CPU transfers, allocations, reuse, vectorized ops).
- Memory reuse: small tensor pool keyed by (shape, dtype, device); hooks in observe/act paths.
- Active‑set + batching scaffolding (CUDA path): compute heuristic active indices (e.g., infected/exposed) and process large per‑agent tensors in fixed‑size batches; scatter results back.
- Snapshot cadence control: `trajectory_save_frequency` (defaults to every substep for parity); reporting prints each step.

### How the Runner now differs from the base Runner
- Same flow: for each step and substep → observe → act → progress → record trajectory.
- Differences are implementation‑level to keep per‑step cost flat:
  - Base: records full state to CPU every substep; unbounded trajectory growth; dense CPU‑heavy paths → near‑constant time per step, but scales poorly with agent count.
  - CUDA: records minimal state (env‑only) and does compressed, async transfers; keeps GPU‑resident refs between snapshots; bounds history and allocs; prepares for active‑set and batched compute.

### How init differs on CUDA vs CPU
- CPU
  - Initializer builds state on host; runner stores an initial CPU snapshot.
  - All subsequent work happens on CPU; snapshots are full copies to CPU (base behavior) or env‑only (optimized path when enabled).
- CUDA
  - Initializer builds state and the runner moves tensor leaves to GPU (shallow move across environment/agents/objects).
  - A dedicated snapshot stream is created; subsequent snapshots transfer minimal CPU payload asynchronously.
  - Optional mixed precision for observe/act when profitable.

### Why these changes matter
- Avoids full‑state copies every substep and keeps transfers bounded.
- Prepares the compute path to be proportional to the active set (not all agents) and batched for cache locality.
- Keeps trajectory growth bounded and I/O overlapped with compute.

### Planned next steps
- Implement true batched per‑agent updates in hot substeps (transmission/progression) and edge filtering per active set.
- Delta snapshots for large agent labels with periodic full “base” snapshots.