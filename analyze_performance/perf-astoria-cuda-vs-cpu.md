## Astoria: CUDA vs CPU Performance Comparison

### Run setup
- Population: Astoria (37,518 agents)
- Steps: 21 (per-substep snapshots)
- Utils: CUDA run uses optimized utils.py, CPU run uses base utils_base.py

### Summary metrics

| Run | Init (loader+exec / runner.init) | Step Time | Total | Loss | Speedup |
|---|---|---:|---:|---:|---:|
| CUDA + optimized | 0.178s / 0.108s | 0.217s | 0.502s | 68.71 | **71.6x** |
| CPU + base | 0.071s / 5.898s | 29.956s | 35.927s | 155.20 | 1.0x |

### Key Performance Insights

**Initialization:**
- CUDA init is **20.9x faster** (5.970s vs 0.285s total)
- CPU runner.init() takes significantly longer (5.898s vs 0.108s) due to base utils processing

**Simulation Steps:**
- CUDA step execution is **138.1x faster** (0.217s vs 29.956s)
- This dramatic difference shows the power of GPU acceleration for large agent populations

**Overall Performance:**
- Total runtime improvement: **71.6x faster** on CUDA
- Memory efficiency: CUDA uses pooled tensors with 82 reuses vs 2 new allocations


### Why CUDA times can show small step‑to‑step variance
- Snapshot steps perform minimal env‑only transfers (fp32), while non‑snapshot steps keep GPU‑resident references only.
- CUDA autotuning/allocator warming in early steps can cause minor variation.


### NYC Population Scaling Results

For comparison, NYC population (2,712,360 agents - 72x larger than Astoria):
- **CUDA + optimized**: Init: 3.553s, Step: 19.427s, Total: 22.980s
- **Scaling efficiency**: 72x more agents takes only 46x more time, showing good scaling

### What changed: CUDA vs Base
- Device management: single boolean gate (`use_gpu`) with device moves at init; optional mixed precision (autocast) on CUDA.
- Trajectory handling:
  - Per‑substep snapshots retained (parity with base) but payload minimized (env‑only by default).
  - Compression/compaction: keep env floats in fp32; downcast int64→int32; bool→uint8.
  - Async, non‑blocking GPU→CPU copies on a dedicated CUDA stream.
- Performance scaffolding: step timings, counters (GPU→CPU transfers, allocations, reuse, vectorized ops).
- Memory reuse: tensor pool keyed by (shape, dtype, device) with leased-tensor mechanism; hooks in observe/act paths and transition modules.
- Active‑set + batching scaffolding (CUDA path): compute heuristic active indices (e.g., infected/exposed) and process large per‑agent tensors in fixed‑size batches; scatter results back.
- Utils optimization: CUDA path uses sparse edge representations vs dense adjacency matrices in base utils.

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

### Generated Visualizations

The benchmark run generated the following plots in `plots/`:
- `astoria_init_times.png`: Comparison of initialization times (CUDA vs CPU)
- `astoria_step_times.png`: Comparison of simulation step times (CUDA vs CPU) 
- `astoria_total_times.png`: Comparison of total runtime (CUDA vs CPU)
- `nyc_cuda_times.png`: NYC population timing breakdown (init/step/total)

### Planned next steps
- Implement true batched per‑agent updates in hot substeps (transmission/progression) and edge filtering per active set.
- Delta snapshots for large agent labels with periodic full "base" snapshots.
- Further optimization of memory pooling and tensor reuse patterns.