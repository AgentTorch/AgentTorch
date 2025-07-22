"""
Example: Distributed Multi-GPU Movement Simulation

This script demonstrates how to run AgentTorch simulations across multiple GPUs
for massive population scales (millions of agents).

Usage:
    python run_distributed_movement_sim.py --agents 1000000 --gpus 4 --steps 50
"""

import argparse
import time
import torch
from agent_torch.populations import sample2
from agent_torch.examples.models import movement
from agent_torch.core.distributed_runner import launch_distributed_simulation
from agent_torch.core.helpers import read_config
import tempfile
import yaml
import os


def create_large_scale_config(num_agents, num_steps, num_episodes=1):
    """
    Create a configuration for large-scale distributed simulation.
    
    Args:
        num_agents: Total number of agents across all GPUs
        num_steps: Number of simulation steps
        num_episodes: Number of episodes
    """
    config = {
        "simulation_metadata": {
            "calibration": False,
            "device": "cuda",  # Will be overridden by distributed runner
            "num_agents": num_agents,
            "num_episodes": num_episodes,
            "num_steps_per_episode": num_steps,
            "num_substeps_per_step": 1,
        },
        "distributed": {
            "strategy": "data_parallel",  # or "spatial_parallel"
            "sync_frequency": 5,  # Synchronize every 5 steps
        },
        "state": {
            "agents": {
                "citizens": {
                    "number": num_agents,
                    "properties": {
                        "position": {
                            "dtype": "float",
                            "initialization_function": None,
                            "learnable": False,
                            "name": "position",
                            "shape": [num_agents, 2],
                            "value": [0.0, 0.0]
                        }
                    }
                }
            },
            "environment": {
                "bounds": {
                    "dtype": "float",
                    "initialization_function": None,
                    "learnable": False,
                    "name": "bounds",
                    "shape": [2],
                    "value": [100.0, 100.0]
                }
            },
            "network": {},
            "objects": None
        },
        "substeps": {
            "0": {
                "active_agents": ["citizens"],
                "description": "Distributed agent movement simulation",
                "name": "Movement",
                "observation": {
                    "citizens": None
                },
                "policy": {
                    "citizens": {
                        "move": {
                            "arguments": {
                                "step_size": {
                                    "dtype": "float",
                                    "initialization_function": None,
                                    "learnable": True,
                                    "name": "Step size parameter",
                                    "shape": [1],
                                    "value": 1.0
                                }
                            },
                            "generator": "RandomMove",
                            "input_variables": {
                                "position": "agents/citizens/position"
                            },
                            "output_variables": ["direction"]
                        }
                    }
                },
                "reward": None,
                "transition": {
                    "update_position": {
                        "arguments": {
                            "bounds": {
                                "dtype": "float",
                                "initialization_function": None,
                                "learnable": True,
                                "name": "Environment bounds",
                                "shape": [2],
                                "value": [100.0, 100.0]
                            }
                        },
                        "generator": "UpdatePosition",
                        "input_variables": {
                            "position": "agents/citizens/position"
                        },
                        "output_variables": ["position"]
                    }
                }
            }
        }
    }
    return config


def benchmark_distributed_vs_single_gpu(num_agents, num_steps):
    """Compare distributed vs single GPU performance."""
    
    print(f"\n=== Benchmarking: {num_agents:,} agents, {num_steps} steps ===")
    
    config = create_large_scale_config(num_agents, num_steps)
    
    # Single GPU benchmark
    print("\n1. Single GPU Simulation:")
    try:
        start_time = time.time()
        single_result = launch_distributed_simulation(
            config, movement.registry, world_size=1, num_steps=num_steps
        )
        single_gpu_time = time.time() - start_time
        print(f"   ✅ Completed in {single_gpu_time:.2f} seconds")
        print(f"   📊 Throughput: {num_agents * num_steps / single_gpu_time:,.0f} agent-steps/sec")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        single_gpu_time = float('inf')
    
    # Multi-GPU benchmark (if available)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"\n2. Multi-GPU Simulation ({num_gpus} GPUs):")
        try:
            start_time = time.time()
            multi_result = launch_distributed_simulation(
                config, movement.registry, world_size=num_gpus, num_steps=num_steps
            )
            multi_gpu_time = time.time() - start_time
            print(f"   ✅ Completed in {multi_gpu_time:.2f} seconds")
            print(f"   📊 Throughput: {num_agents * num_steps / multi_gpu_time:,.0f} agent-steps/sec")
            
            if single_gpu_time != float('inf'):
                speedup = single_gpu_time / multi_gpu_time
                print(f"   🚀 Speedup: {speedup:.2f}x over single GPU")
                efficiency = speedup / num_gpus * 100
                print(f"   ⚡ Efficiency: {efficiency:.1f}%")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    else:
        print(f"\n2. Multi-GPU: Only {num_gpus} GPU available, skipping multi-GPU test")


def run_massive_simulation(num_agents, num_steps, world_size=None):
    """Run a massive distributed simulation."""
    
    print(f"\n=== Massive Scale Simulation ===")
    print(f"🏢 Agents: {num_agents:,}")
    print(f"⏱️  Steps: {num_steps}")
    print(f"🖥️  GPUs: {world_size or torch.cuda.device_count()}")
    
    config = create_large_scale_config(num_agents, num_steps)
    
    start_time = time.time()
    
    try:
        final_state = launch_distributed_simulation(
            config, movement.registry, world_size=world_size, num_steps=num_steps
        )
        
        simulation_time = time.time() - start_time
        
        print(f"\n✅ Simulation completed successfully!")
        print(f"⏱️  Total time: {simulation_time:.2f} seconds")
        print(f"📊 Throughput: {num_agents * num_steps / simulation_time:,.0f} agent-steps/sec")
        
        if final_state and 'agents' in final_state:
            positions = final_state['agents']['citizens']['position']
            print(f"📍 Final position tensor shape: {positions.shape}")
            print(f"📈 Average final position: {positions.mean(dim=0)}")
            print(f"📏 Position standard deviation: {positions.std(dim=0)}")
            
        return final_state
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Distributed Movement Simulation")
    parser.add_argument("--agents", type=int, default=10000, 
                      help="Number of agents (default: 10,000)")
    parser.add_argument("--steps", type=int, default=20,
                      help="Number of simulation steps (default: 20)")
    parser.add_argument("--gpus", type=int, default=None,
                      help="Number of GPUs to use (default: all available)")
    parser.add_argument("--benchmark", action="store_true",
                      help="Run benchmark comparing single vs multi-GPU")
    parser.add_argument("--massive", action="store_true",
                      help="Run massive scale simulation")
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires GPU support.")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"🖥️  Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    if args.benchmark:
        # Run performance benchmarks
        benchmark_distributed_vs_single_gpu(args.agents, args.steps)
    
    elif args.massive:
        # Run massive simulation
        run_massive_simulation(args.agents, args.steps, args.gpus)
    
    else:
        # Standard distributed simulation
        print(f"\n=== Standard Distributed Simulation ===")
        config = create_large_scale_config(args.agents, args.steps)
        
        start_time = time.time()
        final_state = launch_distributed_simulation(
            config, movement.registry, world_size=args.gpus, num_steps=args.steps
        )
        
        if final_state:
            simulation_time = time.time() - start_time
            print(f"✅ Simulation completed in {simulation_time:.2f} seconds")
            
            positions = final_state['agents']['citizens']['position']
            print(f"📍 Final positions shape: {positions.shape}")
            print(f"📈 Average position: {positions.mean(dim=0)}")


if __name__ == "__main__":
    main() 