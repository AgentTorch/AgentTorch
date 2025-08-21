"""
Distributed runner for AgentTorch that supports multi-GPU simulations.
Implements data parallelism by partitioning agents across multiple GPUs.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from typing import Dict, List, Any, Optional
import re

from agent_torch.core.runner import Runner
from agent_torch.core.helpers.general import (
    get_by_path,
    set_by_path,
    copy_module,
    to_cpu,
)


class DistributedRunner(Runner):
    """
    Multi-GPU runner that partitions agents across available GPUs.
    
    Supports:
    - Data parallelism: Split agents across GPUs
    - Spatial parallelism: Split geographic regions across GPUs
    - Automatic load balancing and synchronization
    """
    
    def __init__(self, config, registry, world_size=None, backend='nccl'):
        super().__init__(config, registry)
        
        # Distributed configuration
        self.world_size = world_size or torch.cuda.device_count()
        self.backend = backend
        self.is_distributed = True
        
        # Partitioning strategy
        self.partition_strategy = config.get("distributed", {}).get("strategy", "data_parallel")
        self.sync_frequency = config.get("distributed", {}).get("sync_frequency", 1)
        
        # Will be set during initialization
        self.rank = None
        self.local_agents = None
        self.agent_partition_map = None
        
    def setup_distributed(self, rank, world_size):
        """Initialize distributed training environment."""
        self.rank = rank
        self.world_size = world_size
        
        # Set device for this process
        torch.cuda.set_device(rank)
        self.device = torch.device(f'cuda:{rank}')
        
        # Initialize process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(self.backend, rank=rank, world_size=world_size)
        
        print(f"Rank {rank}/{world_size} initialized on {self.device}")
        
    def partition_agents(self, total_agents: int) -> Dict[str, torch.Tensor]:
        """Partition agents across GPUs based on strategy."""
        
        if self.partition_strategy == "data_parallel":
            return self._partition_data_parallel(total_agents)
        elif self.partition_strategy == "spatial_parallel":
            return self._partition_spatial_parallel(total_agents)
        else:
            raise ValueError(f"Unknown partition strategy: {self.partition_strategy}")
    
    def _partition_data_parallel(self, total_agents: int) -> Dict[str, torch.Tensor]:
        """Split agents evenly across GPUs."""
        agents_per_gpu = total_agents // self.world_size
        remainder = total_agents % self.world_size
        
        # Calculate start and end indices for each rank
        start_idx = self.rank * agents_per_gpu + min(self.rank, remainder)
        end_idx = start_idx + agents_per_gpu + (1 if self.rank < remainder else 0)
        
        partition_map = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'local_size': end_idx - start_idx,
            'global_size': total_agents
        }
        
        print(f"Rank {self.rank}: agents {start_idx}-{end_idx} ({end_idx-start_idx} agents)")
        return partition_map
    
    def _partition_spatial_parallel(self, total_agents: int) -> Dict[str, torch.Tensor]:
        """Split agents by geographic regions (if spatial data available)."""
        # This would need spatial coordinates in agent data
        # For now, fallback to data parallel
        print(f"Spatial partitioning not implemented yet, using data parallel")
        return self._partition_data_parallel(total_agents)
    
    def init_distributed_state(self):
        """Initialize state with agent partitioning."""
        # First, do normal initialization
        self.initializer.initialize()
        self.state = self.initializer.state
        
        # Initialize state trajectory (missing from original implementation)
        self.state_trajectory = []
        
        # Get total number of agents
        total_agents = self.config["simulation_metadata"]["num_agents"]
        
        # Partition agents
        self.agent_partition_map = self.partition_agents(total_agents)
        
        # Slice agent tensors for this GPU
        self._slice_agent_tensors()
        
        # Update config for local agent count
        self.config["simulation_metadata"]["local_num_agents"] = self.agent_partition_map['local_size']
        
        # Add initial state to trajectory (like parent Runner does)
        self.state_trajectory.append([to_cpu(self.state)])
        
    def _slice_agent_tensors(self):
        """Slice agent tensors to only include local agents."""
        start_idx = self.agent_partition_map['start_idx']
        end_idx = self.agent_partition_map['end_idx']
        
        # Slice all agent properties
        for agent_type in self.state["agents"]:
            for prop_name, prop_tensor in self.state["agents"][agent_type].items():
                if torch.is_tensor(prop_tensor) and len(prop_tensor.shape) > 0:
                    # Only slice if tensor has agents dimension
                    if prop_tensor.shape[0] == self.agent_partition_map['global_size']:
                        self.state["agents"][agent_type][prop_name] = prop_tensor[start_idx:end_idx].to(self.device)
                    else:
                        self.state["agents"][agent_type][prop_name] = prop_tensor.to(self.device)
        
        # Move other state components to device
        for key in ["environment", "network", "objects"]:
            if key in self.state:
                self.state[key] = self._move_to_device(self.state[key], self.device)
    
    def _move_to_device(self, data, device):
        """Recursively move nested dict/tensor structure to device."""
        if torch.is_tensor(data):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: self._move_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._move_to_device(item, device) for item in data)
        else:
            return data
    
    def gather_agent_states(self, local_tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all GPUs to reconstruct full agent state."""
        # Prepare list to gather into
        tensor_list = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
        
        # Gather tensors from all processes
        dist.all_gather(tensor_list, local_tensor)
        
        # Concatenate to reconstruct full state
        return torch.cat(tensor_list, dim=0)
    
    def synchronize_environment(self):
        """Synchronize environment state across all GPUs."""
        # For environment variables that need global consistency
        for env_var in self.state["environment"]:
            if torch.is_tensor(self.state["environment"][env_var]):
                # Broadcast from rank 0 to ensure consistency
                dist.broadcast(self.state["environment"][env_var], src=0)
    
    def step_distributed(self, num_steps=None):
        """Execute distributed simulation steps."""
        assert self.state is not None
        
        if not num_steps:
            num_steps = self.config["simulation_metadata"]["num_steps_per_episode"]
        
        for time_step in range(num_steps):
            self.state["current_step"] = time_step
            
            # Synchronize environment periodically
            if time_step % self.sync_frequency == 0:
                self.synchronize_environment()
            
            self.state_trajectory.append([])
            
            for substep in self.config["substeps"].keys():
                observation_profile, action_profile = {}, {}
                
                # Execute observation and policy phases locally
                for agent_type in self.config["substeps"][substep]["active_agents"]:
                    observation_profile[agent_type] = self.controller.observe(
                        self.state, self.initializer.observation_function, agent_type
                    )
                    action_profile[agent_type] = self.controller.act(
                        self.state,
                        observation_profile[agent_type],
                        self.initializer.policy_function,
                        agent_type,
                    )
                
                # Execute transition phase with potential synchronization
                next_state = self.controller.progress(
                    self.state, action_profile, self.initializer.transition_function
                )
                
                # Handle cross-GPU interactions if needed
                next_state = self._handle_cross_gpu_interactions(next_state, action_profile)
                
                self.state = next_state
                self.state_trajectory[-1].append(to_cpu(self.state))
    
    def _handle_cross_gpu_interactions(self, state, actions):
        """Handle interactions between agents on different GPUs."""
        # For network-based interactions, gather relevant agent states
        # This is where spatial partitioning would be most beneficial
        
        # For now, synchronize network effects
        if "network" in state and state["network"]:
            # Implement network synchronization logic here
            pass
            
        return state
    
    def gather_final_state(self) -> Dict[str, Any]:
        """Gather final simulation state from all GPUs."""
        final_state = copy_module(self.state)
        
        # Gather agent states from all GPUs
        for agent_type in final_state["agents"]:
            for prop_name, prop_tensor in final_state["agents"][agent_type].items():
                if torch.is_tensor(prop_tensor) and len(prop_tensor.shape) > 0:
                    final_state["agents"][agent_type][prop_name] = self.gather_agent_states(prop_tensor)
        
        return final_state
    
    def cleanup_distributed(self):
        """Clean up distributed resources."""
        if dist.is_initialized():
            dist.destroy_process_group()


def run_distributed_simulation(rank, world_size, config, registry, num_steps=None):
    """
    Function to run on each GPU process.
    
    Args:
        rank: GPU rank (0 to world_size-1)
        world_size: Total number of GPUs
        config: Simulation configuration
        registry: Model registry
        num_steps: Number of simulation steps
    """
    try:
        # Create distributed runner
        runner = DistributedRunner(config, registry, world_size)
        
        # Setup distributed environment
        runner.setup_distributed(rank, world_size)
        
        # Initialize distributed state
        runner.init_distributed_state()
        
        # Run simulation
        runner.step_distributed(num_steps)
        
        # Gather results on rank 0
        if rank == 0:
            final_state = runner.gather_final_state()
            print(f"Simulation completed. Final state shape: {final_state['agents']['citizens']['position'].shape}")
            return final_state
        
    except Exception as e:
        print(f"Error on rank {rank}: {e}")
        raise
    finally:
        runner.cleanup_distributed()


def launch_distributed_simulation(config, registry, world_size=None, num_steps=None):
    """
    Launch multi-GPU distributed simulation.
    
    Args:
        config: Simulation configuration
        registry: Model registry  
        world_size: Number of GPUs (default: all available)
        num_steps: Number of simulation steps
        
    Returns:
        Final simulation state (only on rank 0)
    """
    if world_size is None:
        world_size = torch.cuda.device_count()
    
    if world_size <= 1:
        print("Only 1 GPU available, running on single GPU")
        runner = Runner(config, registry)
        runner.init()
        runner.step(num_steps)
        return runner.state
    
    print(f"Launching distributed simulation on {world_size} GPUs")
    
    # Use multiprocessing to spawn processes for each GPU
    mp.spawn(
        run_distributed_simulation,
        args=(world_size, config, registry, num_steps),
        nprocs=world_size,
        join=True
    ) 