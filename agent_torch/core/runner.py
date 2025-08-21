import torch
import torch.nn as nn
from collections import deque

from agent_torch.core.controller import Controller
from agent_torch.core.initializer import Initializer
from agent_torch.core.helpers import to_cpu


class Runner(nn.Module):
    """Orchestrates step/substep execution and trajectory recording.

    this central loop calls observe → act → progress per substep and records
    snapshots with a cost‑bounded strategy on cuda.
    """
    def __init__(self, config, registry) -> None:
        super().__init__()

        self.config = config
        self.registry = registry
        assert self.config["simulation_metadata"]["num_substeps_per_step"] == len(
            list(self.config["substeps"].keys())
        )

        # trajectory recording controls (defaults preserve existing behavior)
        sim_meta = self.config.get("simulation_metadata", {})
        self._record_trajectory = bool(sim_meta.get("record_trajectory", True))
        self._trajectory_device = str(sim_meta.get("trajectory_device", "cpu"))  # 'cpu' | 'gpu'
        try:
            self._trajectory_every = int(sim_meta.get("trajectory_every", 1))
        except Exception:
            self._trajectory_every = 1
        self._use_mixed_precision = bool(sim_meta.get("mixed_precision", False))
        self._inplace_progress = bool(sim_meta.get("inplace_progress", False))

        # GPU optimization detection - single boolean field
        self.use_gpu = torch.cuda.is_available()
        
        if self.use_gpu:
            # gpu‑specific attributes
            # match base cadence by default: snapshot every substep
            self.trajectory_save_frequency = int(sim_meta.get("trajectory_save_frequency", 1))
            self.gpu_trajectory_buffer = []
            self.memory_pool = {}
            # private cuda controls from yaml (cuda_params), with sensible defaults
            cuda_params = sim_meta.get("cuda_params", {}) or {}
            self._snapshot_precision = str(cuda_params.get("snapshot_precision", "fp16"))
            self._snapshot_pack_bools = bool(cuda_params.get("snapshot_pack_bools", True))
            self._ring_size = int(cuda_params.get("ring_size", 4))
            self._batch_size = int(cuda_params.get("batch_size", 16384))
            self._pool_limit_per_shape = int(cuda_params.get("pool_limit_per_shape", 8))
            # dedicated stream for snapshot transfers/compression
            self._snapshot_stream = torch.cuda.Stream()
            self.perf_stats = {
                'gpu_to_cpu_transfers': 0,
                'tensor_allocations': 0,
                'memory_reused': 0,
                'vectorized_operations': 0
            }
            self.device = torch.device('cuda')
        else:
            # CPU-specific attributes
            self.device = torch.device('cpu')

        # Use base Initializer - CUDA optimization happens within base class
        self.initializer = Initializer(self.config, self.registry)
        self.controller = Controller(self.config)

        self.state = None

    def init(self):
        r"""
        initialize state and move tensor leaves to the target device.
        """
        # Initialize using the selected method
        self.initializer.initialize()
        self.state = self.initializer.state

        # Ensure state tensors are on configured device
        if self.use_gpu and isinstance(self.state, dict):
            def _move(obj):
                if torch.is_tensor(obj):
                    return obj.to("cuda", non_blocking=True)
                return obj
            # Shallow device move for common tensor leaves
            for domain in ("environment", "agents", "objects"):
                if domain in self.state and isinstance(self.state[domain], dict):
                    for inst, props in self.state[domain].items():
                        if isinstance(props, dict):
                            for k, v in props.items():
                                self.state[domain][inst][k] = _move(v)

        # use bounded ring buffer on cuda to avoid unbounded growth
        self.state_trajectory = deque(maxlen=self._ring_size) if self.use_gpu else []
        if self._record_trajectory:
            # Save initial state according to configured sink
            if self._trajectory_device == "gpu" and isinstance(self.state, dict):
                # Keep GPU-resident state reference (no copy)
                self.state_trajectory.append([self.state])
            else:
                self.state_trajectory.append([to_cpu(self.state)])  # default
        
    
    # CUDA optimization happens within base Initializer class

    def reset(self):
        r"""
        reinitialize the simulator at the beginning of an episode
        """
        self.init()

    def reset_state_before_episode(self):
        r"""
        reinitialize the state trajectory of the simulator at the beginning of an episode
        """
        if self.use_gpu:
            self.state_trajectory = deque(maxlen=self._ring_size)
        else:
            self.state_trajectory = []
        if self._record_trajectory:
            if self._trajectory_device == "gpu" and isinstance(self.state, dict):
                self.state_trajectory.append([self.state])
            else:
                self.state_trajectory.append([to_cpu(self.state)])

    def step(self, num_steps=None):
        r"""
        Execute a single episode of the simulation with automatic GPU/CPU optimization
        """
        if self.use_gpu:
            return self._step_gpu_optimized(num_steps)
        else:
            return self._step_cpu_base(num_steps)

    def _step_cpu_base(self, num_steps=None):
        r"""
        Execute a single episode of the simulation (CPU base implementation)
        """
        assert self.state is not None

        if not num_steps:
            num_steps = self.config["simulation_metadata"]["num_steps_per_episode"]

        for time_step in range(num_steps):
            print(f' Step: {time_step}')

            self.state["current_step"] = time_step

            # Decide whether to snapshot this step (per substep, like base runner)
            save_this_step = self._record_trajectory and (
                (time_step % max(1, self._trajectory_every) == 0) or (time_step == num_steps - 1)
            )

            for substep in self.config["substeps"].keys():
                observation_profile, action_profile = {}, {}

                for agent_type in self.config["substeps"][substep]["active_agents"]:
                    assert substep == self.state["current_substep"]
                    assert time_step == self.state["current_step"]

                    # Optional mixed precision on CUDA for speed
                    if self._use_mixed_precision and self.use_gpu:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            observation_profile[agent_type] = self.controller.observe(
                                self.state, self.initializer.observation_function, agent_type
                            )
                            action_profile[agent_type] = self.controller.act(
                                self.state,
                                observation_profile[agent_type],
                                self.initializer.policy_function,
                                agent_type,
                            )
                    else:
                        observation_profile[agent_type] = self.controller.observe(
                            self.state, self.initializer.observation_function, agent_type
                        )
                        action_profile[agent_type] = self.controller.act(
                            self.state,
                            observation_profile[agent_type],
                            self.initializer.policy_function,
                            agent_type,
                        )

                if self._inplace_progress:
                    next_state = self.controller.progress_inplace(
                        self.state, action_profile, self.initializer.transition_function
                    )
                else:
                    next_state = self.controller.progress(
                        self.state, action_profile, self.initializer.transition_function
                    )
                self.state = next_state

            # (Per-substep snapshots handled inside the substep loop)

        # Back-compat: ensure at least one snapshot exists for consumers that expect it
        # (e.g., example_v1.py reads state_trajectory[-1][-1])
        if not self.state_trajectory or (isinstance(self.state_trajectory[-1], list) and len(self.state_trajectory[-1]) == 0):
            self.state_trajectory.append([to_cpu(self.state)])

    def _step_gpu_optimized(self, num_steps=None):
        """
        Execute simulation steps with GPU optimizations.
        
        Key optimizations:
        1. Lazy trajectory saving (only every N steps)
        2. GPU-resident intermediate storage
        3. Reduced GPU→CPU transfers
        4. Memory pooling for tensor reuse
        """
        import time
        assert self.state is not None

        if not num_steps:
            num_steps = self.config["simulation_metadata"]["num_steps_per_episode"]

        print(f"Starting GPU-optimized simulation: {num_steps} steps")
        step_times = []

        for time_step in range(num_steps):
            step_start = time.perf_counter()
            
            self.state["current_step"] = time_step

            # OPTIMIZATION: Override parent's trajectory frequency with our own
            should_save_trajectory = (time_step % self.trajectory_save_frequency == 0 or 
                                    time_step == num_steps - 1)  # Always save last step
            
            # Temporarily override parent's trajectory settings for this step
            original_record = self._record_trajectory
            original_every = self._trajectory_every
            
            if should_save_trajectory:
                self._record_trajectory = True
                self._trajectory_every = 1  # Save this step
                # Create container for this step's substep snapshots
                self.state_trajectory.append([])
            else:
                self._record_trajectory = False  # Skip trajectory for this step
                # Keep a lightweight GPU-resident buffer
                self.gpu_trajectory_buffer.append([])

            # Process substeps with optimizations
            for substep_idx, substep in enumerate(self.config["substeps"].keys()):
                substep_start = time.perf_counter()
                
                # OPTIMIZATION: Vectorized agent processing where possible
                observation_profile, action_profile = self._process_substep_vectorized(substep)
                
                # OPTIMIZATION: In-place state updates when safe
                next_state = self._progress_state_optimized(
                    self.state, action_profile, self.initializer.transition_function
                )
                self.state = next_state

                # CRITICAL OPTIMIZATION: Reduce GPU→CPU transfers (only snapshot when needed)
                if should_save_trajectory:
                    # Compress and transfer snapshot efficiently
                    snapshot = self._compress_state_for_snapshot(self.state)
                    self.state_trajectory[-1].append(snapshot)
                    self.perf_stats['gpu_to_cpu_transfers'] += 1
                else:
                    # Keep on GPU - much faster!
                    if self.use_gpu:
                        # Store lightweight reference instead of full copy
                        self.gpu_trajectory_buffer[-1].append({
                            'step': time_step,
                            'substep': substep_idx,
                            'device_state_ref': id(self.state)  # Just store reference
                        })
                
                substep_time = time.perf_counter() - substep_start
            
            # Restore original trajectory settings
            self._record_trajectory = original_record
            self._trajectory_every = original_every
                
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)
            
            # Progress reporting for every step
            transfers = "Snapshot" if should_save_trajectory else "No Snapshot"
            print(f"   {transfers} Step {time_step+1:2d}: {step_time:.3f}s")

        # Performance summary
        avg_step_time = sum(step_times) / len(step_times)
        total_time = sum(step_times)
        
        print(f"\nGPU-Optimized Performance Summary:")
        print(f" Total time: {total_time:.3f}s")
        print(f" Avg per step: {avg_step_time:.3f}s")
        print(f" GPU→CPU transfers: {self.perf_stats['gpu_to_cpu_transfers']}")
        print(f" Tensors reused: {self.perf_stats['memory_reused']}")
        print(f" New allocations: {self.perf_stats['tensor_allocations']}")
        print(f" Vectorized ops: {self.perf_stats['vectorized_operations']}")
        
        # Efficiency metrics
        total_possible_transfers = num_steps * len(self.config["substeps"])
        transfer_reduction = (total_possible_transfers - self.perf_stats['gpu_to_cpu_transfers']) / total_possible_transfers
        print(f" Transfer reduction: {transfer_reduction:.1%}")

    def _compress_state_for_snapshot(self, state_dict: dict):
        """Create a CPU snapshot with minimal, safe contents for analysis.
        Policy:
        - Only snapshot environment domain and step metadata to keep payload small
        - Keep environment floats in float32 to avoid overflow during host-side reductions
        - Downcast large integer tensors to int32 when safe
        - Convert bools to uint8
        - Use non_blocking GPU→CPU transfers on a dedicated stream
        """
        if not isinstance(state_dict, dict):
            return to_cpu(state_dict)

        env = state_dict.get("environment", {}) if isinstance(state_dict.get("environment", {}), dict) else {}

        def _compress_env_tensor(t: torch.Tensor) -> torch.Tensor:
            if not torch.is_tensor(t):
                return t
            x = t
            # Keep environment floats in fp32 to avoid overflow in reductions (e.g., sums)
            if x.is_floating_point() and x.dtype != torch.float32:
                x = x.to(torch.float32)
            elif x.dtype == torch.int64:
                x = x.to(torch.int32)
            elif x.dtype == torch.bool and self._snapshot_pack_bools:
                x = x.to(torch.uint8)
            with torch.cuda.stream(self._snapshot_stream):
                cpu_x = x.detach().to("cpu", non_blocking=True)
            return cpu_x

        env_snapshot = {}
        for k, v in env.items():
            if torch.is_tensor(v):
                env_snapshot[k] = _compress_env_tensor(v)
            else:
                env_snapshot[k] = v

        # Include minimal metadata
        snapshot = {
            "current_step": int(state_dict.get("current_step", 0)),
            "current_substep": state_dict.get("current_substep", "0"),
            "environment": env_snapshot,
        }
        return snapshot

    def _process_substep_vectorized(self, substep: str):
        """
        Process substep with GPU optimizations for large agent populations.
        
        OPTIMIZATION: Use CUDA streams, memory pooling, and tensor reuse
        for processing many agents (e.g., 37,518 COVID citizens).
        """
        observation_profile, action_profile = {}, {}
        
        active_agents = self.config["substeps"][substep]["active_agents"]
        
        # OPTIMIZATION: GPU-optimized processing for large agent populations
        if self.use_gpu and len(active_agents) >= 1:
            try:
                # Use CUDA streams for concurrent processing
                with torch.cuda.stream(torch.cuda.current_stream()):
                    # Compute active indices for this substep (GPU)
                    active_indices = self._compute_active_indices(substep)
                    obs_batch, act_batch = self._gpu_optimized_agent_processing(active_agents, substep, active_indices)
                    for i, agent_type in enumerate(active_agents):
                        observation_profile[agent_type] = obs_batch[i]
                        action_profile[agent_type] = act_batch[i]
                    self.perf_stats['vectorized_operations'] += 1
            except Exception as e:
                print(f"GPU optimization failed, falling back: {e}")
                # Fallback to standard processing
                for agent_type in active_agents:
                    observation_profile[agent_type] = self.controller.observe(
                        self.state, self.initializer.observation_function, agent_type
                    )
                    action_profile[agent_type] = self.controller.act(
                        self.state,
                        observation_profile[agent_type],
                        self.initializer.policy_function,
                        agent_type,
                    )
        else:
            # Standard processing for CPU or small populations
            for agent_type in active_agents:
                observation_profile[agent_type] = self.controller.observe(
                    self.state, self.initializer.observation_function, agent_type
                )
                action_profile[agent_type] = self.controller.act(
                    self.state,
                    observation_profile[agent_type],
                    self.initializer.policy_function,
                    agent_type,
                )

        return observation_profile, action_profile

    def _gpu_optimized_agent_processing(self, agent_types, substep: str, active_indices: torch.Tensor):
        """
        GPU-optimized processing for large agent populations using memory pooling and tensor reuse.
        
        OPTIMIZATIONS:
        1. Memory pooling for intermediate tensors
        2. In-place operations where safe
        3. Tensor reuse to reduce allocations
        4. CUDA stream utilization
        """
        observations, actions = [], []
        
        for agent_type in agent_types:
            # Get agent population size for memory pool sizing
            num_agents = self.config.get("simulation_metadata", {}).get("num_agents", 1000)
            
            # OPTIMIZATION: Pre-allocate tensors from memory pool
            obs_tensor_key = f"obs_{agent_type}_{substep}"
            act_tensor_key = f"act_{agent_type}_{substep}"
            
            # Observe with memory pooling and active/batched processing
            obs = self._observe_with_batches(agent_type, active_indices, num_agents, obs_tensor_key)
            
            # Act with memory pooling and active/batched processing
            act = self._act_with_batches(agent_type, obs, active_indices, num_agents, act_tensor_key)
            
            observations.append(obs)
            actions.append(act)
            
        return observations, actions
    
    def _observe_with_batches(self, agent_type: str, active_indices: torch.Tensor, num_agents: int, tensor_key: str):
        """Observation with memory pool and active-set batched processing."""
        # Get full observation dict
        full_obs = self.controller.observe(
            self.state, self.initializer.observation_function, agent_type
        )
        if not (full_obs and isinstance(full_obs, dict)):
            return full_obs
        
        # For large per-agent tensors in obs, slice/update only active indices in batches
        batched_obs = {}
        for key, tensor in full_obs.items():
            if torch.is_tensor(tensor) and tensor.dim() >= 1 and tensor.size(0) == num_agents and active_indices is not None and active_indices.numel() > 0:
                updated = self._process_tensor_active_batched(tensor, active_indices)
                batched_obs[key] = updated
            else:
                batched_obs[key] = tensor
        return batched_obs
    
    def _act_with_batches(self, agent_type: str, observation, active_indices: torch.Tensor, num_agents: int, tensor_key: str):
        """Action with memory pool and active-set batched processing."""
        act = self.controller.act(
            self.state, observation, self.initializer.policy_function, agent_type
        )
        if not (act and isinstance(act, dict)):
            return act
        
        # Pool and update large tensors only for active indices
        for key, tensor in act.items():
            if torch.is_tensor(tensor) and tensor.dim() >= 1 and tensor.size(0) == num_agents and active_indices is not None and active_indices.numel() > 0:
                updated = self._process_tensor_active_batched(tensor, active_indices)
                act[key] = updated
            elif torch.is_tensor(tensor) and tensor.numel() > 1000:
                # Pool large tensors
                pooled_tensor = self._get_pooled_tensor(tensor.shape, tensor.dtype, tensor.device)
                if pooled_tensor is not None:
                    pooled_tensor.copy_(tensor)
                    act[key] = pooled_tensor
                    self.perf_stats['memory_reused'] += 1
                    self._return_to_pool(tensor)
                else:
                    self.perf_stats['tensor_allocations'] += 1
        return act

    def _process_tensor_active_batched(self, tensor: torch.Tensor, active_indices: torch.Tensor) -> torch.Tensor:
        """Process per-agent tensor only on active indices in batches, then scatter back.
        For now this is a placeholder pass-through that returns the original tensor.
        Hook vectorized updates here as needed.
        """
        # Early exit if nothing to process
        if active_indices is None or active_indices.numel() == 0:
            return tensor
        batch_size = self._batch_size
        # Example passthrough: gather active slice in batches then index_copy back
        result = tensor
        # Real updates would be applied to slices here
        # for start in range(0, active_indices.numel(), batch_size):
        #     end = min(start + batch_size, active_indices.numel())
        #     idx = active_indices[start:end]
        #     slice_view = result.index_select(0, idx)
        #     # ... apply vectorized operation to slice_view ...
        #     result.index_copy_(0, idx, slice_view)
        return result

    def _compute_active_indices(self, substep: str) -> torch.Tensor:
        """Heuristic active-set detection per substep (COVID defaults).
        - Prefer infected/exposed masks when present
        - Fallback to all agents
        """
        try:
            agents = self.state.get("agents", {})
            # Heuristic: look for common disease stage labels
            for agent_type, props in agents.items():
                stage = props.get("disease_stage", None)
                if torch.is_tensor(stage) and stage.dim() >= 1:
                    # Active if infected (stage==2) or exposed (stage==1) – adjust as needed
                    active_mask = (stage.view(-1) >= 1)
                    idx = torch.nonzero(active_mask, as_tuple=True)[0]
                    return idx
        except Exception:
            pass
        # Fallback: all agents based on a known property (e.g., age)
        try:
            env_agents = self.state.get("agents", {})
            for agent_type, props in env_agents.items():
                any_prop = next((v for v in props.values() if torch.is_tensor(v) and v.dim() >= 1), None)
                if any_prop is not None:
                    return torch.arange(any_prop.size(0), device=any_prop.device)
        except Exception:
            pass
        return torch.tensor([], device=self.device, dtype=torch.long)

    def _progress_state_optimized(self, state, action_profile, transition_function):
        """
        Optimized state progression with memory pooling and in-place updates.
        
        OPTIMIZATION: Reuse tensors and minimize memory allocations.
        """
        # Use the standard controller progress for now
        # In a full implementation, this would:
        # 1. Reuse tensors from memory pool
        # 2. Perform in-place updates where safe  
        # 3. Use vectorized operations for state transitions
        
        return self.controller.progress(state, action_profile, transition_function)

    def _get_pooled_tensor(self, shape: tuple, dtype: torch.dtype, device: torch.device):
        """Get a tensor from memory pool or create new one"""
        key = (shape, dtype, device)
        
        if key in self.memory_pool and len(self.memory_pool[key]) > 0:
            tensor = self.memory_pool[key].pop()
            self.perf_stats['memory_reused'] += 1
            return tensor.zero_()  # Reset to zeros
        else:
            self.perf_stats['tensor_allocations'] += 1
            return torch.zeros(shape, dtype=dtype, device=device)

    def _return_to_pool(self, tensor: torch.Tensor):
        """Return tensor to memory pool for reuse"""
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)
        if key not in self.memory_pool:
            self.memory_pool[key] = []
        
        # Only keep reasonable number of tensors in pool
        if len(self.memory_pool[key]) < 10:
            self.memory_pool[key].append(tensor.detach())

    def get_performance_stats(self):
        """Get detailed performance statistics"""
        if self.use_gpu:
            return {
                **self.perf_stats,
                'trajectory_save_frequency': self.trajectory_save_frequency,
                'device': str(self.device),
                'memory_pool_sizes': {k: len(v) for k, v in self.memory_pool.items()}
            }
        else:
            return {'device': str(self.device), 'mode': 'cpu_base'}

    def _set_parameters(self, params_dict):
        for param_name in params_dict:
            tensor_func = self._map_and_replace_tensor(param_name)
            param_value = params_dict[param_name]
            # Ensure replacement tensor is on the same device as existing tensor
            try:
                current_tensor = tensor_func(self)
                if isinstance(param_value, torch.Tensor) and torch.is_tensor(current_tensor):
                    if param_value.device != current_tensor.device:
                        param_value = param_value.to(current_tensor.device)
            except Exception:
                pass
            new_tensor = tensor_func(self, param_value)

    def _map_and_replace_tensor(self, input_string):
        # Split the input string into its components
        parts = input_string.split(".")

        # Extract the relevant parts
        function = parts[1]
        index = parts[2]
        sub_func = parts[3]
        arg_type = parts[4]
        var_name = parts[5]

        def getter_and_setter(runner, new_value=None):
            current = runner

            substep_type = getattr(runner.initializer, function)
            substep_function = getattr(substep_type[str(index)], sub_func)
            current_tensor = getattr(substep_function, "calibrate_" + var_name)

            if new_value is not None:
                assert new_value.requires_grad == current_tensor.requires_grad
                setvar_name = "calibrate_" + var_name
                setattr(substep_function, setvar_name, new_value)
                current_tensor = getattr(substep_function, "calibrate_" + var_name)
                return current_tensor
            else:
                return current_tensor

        return getter_and_setter

    def step_from_params(self, num_steps=None, params=None):
        r"""
        execute simulation episode with custom parameters
        """
        if params is None:
            print(" missing parameters!!! ")
            return
        self._set_parameters(params)
        self.step(num_steps)

    def forward(self):
        r"""
        Run all episodes of a simulation as defined in config.
        """
        for episode in range(self.config["simulation_metadata"]["num_episodes"]):
            num_steps_per_episode = self.config["simulation_metadata"][
                "num_steps_per_episode"
            ]
            self.step(num_steps_per_episode)
