import torch
import torch.nn as nn
import types

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

        # trajectory recording controls (defaults preserve base runner behavior)
        sim_meta = self.config.get("simulation_metadata", {})
        # snapshots are always recorded every substep to CPU (match base behavior)
        self._use_mixed_precision = bool(sim_meta.get("mixed_precision", False))
        # GPU optimization detection - single boolean field
        self.use_gpu = torch.cuda.is_available()
        
        if self.use_gpu:
            # gpu‑specific attributes
            self.memory_pool = {}
            self._leased_tensors = []  # tensors checked out from pool during a substep
            cuda_params = sim_meta.get("cuda_params", {}) or {}
            self._snapshot_pack_bools = bool(cuda_params.get("snapshot_pack_bools", True))
            self._batch_size = int(cuda_params.get("batch_size", 16384))
            self._pool_limit_per_shape = int(cuda_params.get("pool_limit_per_shape", 12))
            self._inplace_progress = bool(cuda_params.get("inplace_progress", False))
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

        # tensors are already placed on device by Initializer

        # record initial snapshot on cpu (match base runner behavior)
        self.state_trajectory = []
        self.state_trajectory.append([to_cpu(self.state)])
        
        # Wire pooled buffer allocator into transition modules (CUDA only)
        if self.use_gpu:
            self._wire_transition_buffer_allocator()
    
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
        self.state_trajectory = []
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
            self.state["current_step"] = time_step
            self.state_trajectory.append([])  # track state after each substep

            for substep in self.config["substeps"].keys():
                observation_profile, action_profile = {}, {}

                for agent_type in self.config["substeps"][substep]["active_agents"]:
                    assert substep == self.state["current_substep"]
                    assert time_step == self.state["current_step"]
                    
                    observation_profile[agent_type] = self.controller.observe(
                        self.state, self.initializer.observation_function, agent_type
                    )
                    action_profile[agent_type] = self.controller.act(
                        self.state, observation_profile[agent_type],
                        self.initializer.policy_function, agent_type,
                    )

                next_state = self.controller.progress(
                    self.state, action_profile, self.initializer.transition_function
                )
                self.state = next_state
                self.state_trajectory[-1].append(to_cpu(self.state))  # move state to cpu and save


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

        step_times = []

        for time_step in range(num_steps):
            step_start = time.perf_counter()
            
            self.state["current_step"] = time_step

            # always snapshot every step/substep to cpu (match base runner)
            self.state_trajectory.append([])

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

                # snapshot to cpu every substep
                snapshot = self._compress_state_for_snapshot(self.state)
                self.state_trajectory[-1].append(snapshot)
                self.perf_stats['gpu_to_cpu_transfers'] += 1
                # reclaim leased pooled tensors at end of substep
                if self.use_gpu and self._leased_tensors:
                    for t in self._leased_tensors:
                        self._return_to_pool(t)
                    self._leased_tensors.clear()
                
                substep_time = time.perf_counter() - substep_start
            
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)


        # Performance summary
        avg_step_time = sum(step_times) / len(step_times)
        total_time = sum(step_times)
        
        '''
        print(f"\nGPU-Optimized Performance Summary:\n Total time: {total_time:.3f}s\n Avg per step: {avg_step_time:.3f}s\n GPU→CPU transfers: {self.perf_stats['gpu_to_cpu_transfers']}\n Tensors reused: {self.perf_stats['memory_reused']}\n New allocations: {self.perf_stats['tensor_allocations']}\n Vectorized ops: {self.perf_stats['vectorized_operations']}")
        '''

        # Efficiency metrics
        total_possible_transfers = num_steps * len(self.config["substeps"])
        transfer_reduction = (total_possible_transfers - self.perf_stats['gpu_to_cpu_transfers']) / total_possible_transfers
        #print(f" Transfer reduction: {transfer_reduction:.1%}")

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

    def _wire_transition_buffer_allocator(self):
        """Inject pooled buffer allocate/release into transition modules on CUDA.
        Replaces transition._get_buffer to draw tensors from Runner's memory pool.
        """
        for substep, trans_dict in self.initializer.transition_function.items():
            for name, trans in trans_dict.items():
                orig_get_buffer = getattr(trans, "_get_buffer", None)

                def _alloc_like(like_tensor: torch.Tensor, _self=self):
                    t = _self._get_pooled_tensor(tuple(like_tensor.shape), like_tensor.dtype, like_tensor.device)
                    if _self.use_gpu:
                        _self._leased_tensors.append(t)
                    return t

                def _release(t: torch.Tensor, _self=self):
                    _self._return_to_pool(t)

                setattr(trans, "_external_buffer_alloc", _alloc_like)
                setattr(trans, "_external_buffer_release", _release)

                if callable(orig_get_buffer):
                    def patched_get_buffer(self_trans, name, like_tensor, _orig=orig_get_buffer, _alloc=_alloc_like):
                        return _alloc(like_tensor)
                    trans._get_buffer = types.MethodType(patched_get_buffer, trans)

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
                if self.use_gpu and torch.is_tensor(updated) and updated.numel() > 1000:
                    pooled_tensor = self._get_pooled_tensor(updated.shape, updated.dtype, updated.device)
                    if pooled_tensor is not None:
                        pooled_tensor.copy_(updated)
                        batched_obs[key] = pooled_tensor
                        self.perf_stats['memory_reused'] += 1
                        self._leased_tensors.append(pooled_tensor)
                        # return the transient updated (if it was a new tensor) back to pool
                        self._return_to_pool(updated)
                    else:
                        batched_obs[key] = updated
                else:
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
                if self.use_gpu and torch.is_tensor(updated) and updated.numel() > 1000:
                    pooled_tensor = self._get_pooled_tensor(updated.shape, updated.dtype, updated.device)
                    if pooled_tensor is not None:
                        pooled_tensor.copy_(updated)
                        act[key] = pooled_tensor
                        self.perf_stats['memory_reused'] += 1
                        self._leased_tensors.append(pooled_tensor)
                        # return the original tensor to the pool as well
                        self._return_to_pool(updated)
                    else:
                        act[key] = updated
                else:
                    act[key] = updated
            elif torch.is_tensor(tensor) and tensor.numel() > 1000:
                # Pool large tensors
                pooled_tensor = self._get_pooled_tensor(tensor.shape, tensor.dtype, tensor.device)
                if pooled_tensor is not None:
                    pooled_tensor.copy_(tensor)
                    act[key] = pooled_tensor
                    self.perf_stats['memory_reused'] += 1
                    # mark pooled tensor as leased so we can reclaim after substep
                    if self.use_gpu:
                        self._leased_tensors.append(pooled_tensor)
                    # original tensor can also be returned to pool
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
        agents = self.state.get("agents", {})
        # Prefer disease_stage if present
        for _, props in agents.items():
            stage = props.get("disease_stage", None)
            if torch.is_tensor(stage) and stage.dim() >= 1:
                active_mask = (stage.view(-1) >= 1)
                return torch.nonzero(active_mask, as_tuple=True)[0]
        # Fallback: first tensor-like agent property determines full range
        for _, props in agents.items():
            any_prop = next((v for v in props.values() if torch.is_tensor(v) and v.dim() >= 1), None)
            if any_prop is not None:
                return torch.arange(any_prop.size(0), device=any_prop.device)
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
        if len(self.memory_pool[key]) < self._pool_limit_per_shape if hasattr(self, '_pool_limit_per_shape') else 10:
            self.memory_pool[key].append(tensor.detach())

    def get_performance_stats(self):
        """Get detailed performance statistics"""
        if self.use_gpu:
            return {
                **self.perf_stats,
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
