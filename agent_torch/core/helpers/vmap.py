"""
Vmap Utilities for AgentTorch
=============================

This module provides auto-vectorization tools for writing substeps.

Key components:
- `@vmap` decorator: Auto-vectorize single-agent logic over all agents
- `sample_grid()`: Sample a 2D grid at a position (for use inside vmapped functions)

Example:
    from agent_torch.core.substep import SubstepTransition, vmap
    from agent_torch.core.helpers.vmap import sample_grid
    
    @vmap(agent_args=["position", "alive"], shared_args=["sugar_grid"], outputs=["position"])
    class AgentMovement(SubstepTransition):
        def forward(self, state, action=None):
            position = state["position"]      # [2] - single agent
            sugar_grid = state["sugar_grid"]  # [H, W] - shared
            
            sugar_here = sample_grid(sugar_grid, position)
            ...
            return {"position": new_position}
"""
import torch
from functools import wraps


# =============================================================================
# VMAP HELPER FUNCTIONS
# =============================================================================

def sample_grid(grid, position):
    """
    Sample a 2D grid at a single position using bilinear interpolation.
    Use this inside @vmap decorated forward() methods.
    
    Args:
        grid: [H, W] tensor - the environment grid
        position: [2] tensor - (x, y) position to sample
    
    Returns:
        Scalar tensor - interpolated value at position
    
    Example (inside vmapped forward):
        sugar_value = sample_grid(sugar_grid, position)
    """
    H, W = grid.shape
    x, y = position[0], position[1]
    
    # Clamp to valid range
    x = torch.clamp(x, 0, H - 1.001)  # Small epsilon to avoid edge issues
    y = torch.clamp(y, 0, W - 1.001)
    
    # Get integer and fractional parts
    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = torch.clamp(x0 + 1, max=H - 1)
    y1 = torch.clamp(y0 + 1, max=W - 1)
    
    # Fractional weights
    wx = x - x0.float()
    wy = y - y0.float()
    
    # Bilinear interpolation
    v00 = grid[x0, y0]
    v01 = grid[x0, y1]
    v10 = grid[x1, y0]
    v11 = grid[x1, y1]
    
    return (v00 * (1 - wx) * (1 - wy) +
            v01 * (1 - wx) * wy +
            v10 * wx * (1 - wy) +
            v11 * wx * wy)


# =============================================================================
# VMAP CLASS DECORATOR
# =============================================================================

def vmap(agent_args, shared_args=None, outputs=None, compile=False):
    """
    Class decorator to auto-vectorize a SubstepTransition over agents.
    
    Write single-agent logic in forward(), and it will be automatically
    vectorized over all agents using torch.vmap.
    
    Args:
        agent_args: List of agent state fields to vectorize over (dim 0)
                   e.g., ["position", "alive", "sugar_endowment"]
        shared_args: List of environment/shared fields (not vectorized)
                    e.g., ["sugar_grid", "grid_size"]
        outputs: List of output field names
                e.g., ["position"]
        compile: If True, also apply torch.compile for better performance
    
    Example:
        @vmap(agent_args=["position", "alive"], shared_args=["sugar_grid"], outputs=["position"])
        class AgentMovement(SubstepTransition):
            def forward(self, state, action=None):
                # state contains SINGLE AGENT data!
                position = state["position"]      # [2] - one agent
                alive = state["alive"]            # scalar
                sugar_grid = state["sugar_grid"]  # [H, W] - shared
                
                new_position = ...
                return {"position": new_position}
    
    Notes:
        - Write logic for ONE agent (no batch dimensions)
        - Use torch.where() instead of if/else for conditionals
        - Use sample_grid() to read from environment grids
        - Return a dict with output field names matching 'outputs' arg
    """
    shared_args = shared_args or []
    outputs = outputs or []
    
    def decorator(cls):
        # Store vmap config on the class
        cls._vmap_config = {
            "agent_args": agent_args,
            "shared_args": shared_args,
            "outputs": outputs,
            "compile": compile
        }
        
        # Save original forward
        original_forward = cls.forward
        
        def vmapped_forward(self, state, action=None):
            """Wrapper that handles vmap extraction and execution."""
            config = self._vmap_config
            
            # Get agent type (assume first/only agent type for now)
            agent_types = list(self.config["agents"].keys())
            if not agent_types:
                raise ValueError("No agent types found in config")
            agent_type = agent_types[0]
            
            # =============================================================
            # STEP 1: Extract tensors from state
            # =============================================================
            # Agent tensors (to be vmapped over, dim 0)
            agent_tensors = {}
            for field in config["agent_args"]:
                if field in state["agents"][agent_type]:
                    agent_tensors[field] = state["agents"][agent_type][field]
                else:
                    raise KeyError(f"Agent field '{field}' not found in state")
            
            # Shared tensors (environment, not vmapped)
            shared_tensors = {}
            for field in config["shared_args"]:
                if field in state["environment"]:
                    shared_tensors[field] = state["environment"][field]
                elif field in state["agents"][agent_type]:
                    # Allow shared args from agent state too (e.g., constants)
                    shared_tensors[field] = state["agents"][agent_type][field]
                else:
                    raise KeyError(f"Shared field '{field}' not found in state")
            
            # Get number of agents
            first_agent_tensor = next(iter(agent_tensors.values()))
            num_agents = first_agent_tensor.shape[0]
            
            # =============================================================
            # STEP 2: Define single-agent function
            # =============================================================
            def single_agent_fn(*agent_values):
                # Build single-agent state dict
                single_state = {}
                
                # Add agent fields (single agent, no batch dim)
                for i, field in enumerate(config["agent_args"]):
                    single_state[field] = agent_values[i]
                
                # Add shared fields (unchanged)
                for field, tensor in shared_tensors.items():
                    single_state[field] = tensor
                
                # Add learnable args
                if hasattr(self, 'learnable_args') and self.learnable_args:
                    for key, param in self.learnable_args.items():
                        single_state[key] = param
                
                # Add fixed args
                if hasattr(self, 'fixed_args') and self.fixed_args:
                    for key, value in self.fixed_args.items():
                        single_state[key] = value
                
                # Call original forward with single-agent state
                result = original_forward(self, single_state, action)
                
                # Return as tuple of tensors (for vmap)
                if isinstance(result, dict):
                    return tuple(result.values())
                return (result,)
            
            # =============================================================
            # STEP 3: Apply vmap
            # =============================================================
            # Build in_dims: 0 for each agent tensor
            in_dims = tuple([0] * len(agent_tensors))
            
            # Create vmapped function
            vmapped_fn = torch.vmap(single_agent_fn, in_dims=in_dims)
            
            # Optionally compile
            if config["compile"]:
                vmapped_fn = torch.compile(vmapped_fn)
            
            # =============================================================
            # STEP 4: Execute vmapped function
            # =============================================================
            agent_values = tuple(agent_tensors.values())
            results = vmapped_fn(*agent_values)
            
            # =============================================================
            # STEP 5: Build output dict
            # =============================================================
            # Get output field names
            if config["outputs"]:
                output_fields = config["outputs"]
            else:
                # Try to infer from a single call (fallback)
                output_fields = [f"output_{i}" for i in range(len(results))]
            
            output_dict = {}
            for i, field in enumerate(output_fields):
                if i < len(results):
                    output_dict[field] = results[i]
            
            return output_dict
        
        # Replace forward with vmapped version
        cls.forward = vmapped_forward
        
        return cls
    
    return decorator

