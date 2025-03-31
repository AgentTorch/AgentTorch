"""
pr_benchmark.py - Quick benchmarking tool for PR comparisons
Automatically compares current branch with main branch performance
"""
import argparse
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import sys
from pathlib import Path

# Get the AgentTorch root directory
AGENTTORCH_ROOT = str(Path(__file__).parents[3])
sys.path.append(AGENTTORCH_ROOT)

# Import both implementations
from agent_torch.models.predator_prey.predator_prey_vmap import SimParams, run_simulation

def run_pr_benchmark(config_path, num_steps=50, num_runs=2, save_results=True, use_compile=False, profile=False):
    """
    Run a quick benchmark for PR comparison
    
    Args:
        config_path: Path to the config file for standard implementation
        num_steps: Number of steps to run
        num_runs: Number of runs
        save_results: Whether to save results to a file
        use_compile: Whether to use torch.compile (if available)
        profile: Whether to enable profiling
    
    Returns:
        Dict with benchmark results
    """
    print(f"Running PR benchmark with {num_steps} steps, {num_runs} runs")
    
    # Check if torch.compile is available
    compile_available = hasattr(torch, 'compile')
    if use_compile and not compile_available:
        print("Warning: torch.compile not available, running without compilation")
        use_compile = False
    
    # Get current git branch
    try:
        current_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
            cwd=AGENTTORCH_ROOT
        ).decode().strip()
        print(f"Current branch: {current_branch}")
    except Exception as e:
        current_branch = "unknown"
        print(f"Could not determine git branch: {e}")
    
    # Run standard implementation
    from agent_torch.core import Registry, Runner
    from agent_torch.core.helpers import read_config, read_from_file, grid_network
    from agent_torch.models.predator_prey.helpers.map import map_network
    
    # Load configuration
    config = read_config(config_path)
    
    # Override step settings
    config["simulation_metadata"]["num_steps_per_episode"] = num_steps
    
    # Create registry and register functions
    registry = Registry()
    registry.register(read_from_file, "read_from_file", "initialization")
    registry.register(grid_network, "grid", key="network")
    registry.register(map_network, "map", key="network")
    
    # Ensure classes are registered
    try:
        from agent_torch.core.registry import Registry
        from agent_torch.models.predator_prey.substeps.move import FindNeighbors, DecideMovement, UpdatePositions
        from agent_torch.models.predator_prey.substeps.hunt import FindTargets, HuntPrey
        from agent_torch.models.predator_prey.substeps.eat import FindEatableGrass, EatGrass
        from agent_torch.models.predator_prey.substeps.grow import GrowGrass
        
        class_registry = {
            "find_neighbors": (FindNeighbors, "observation"),
            "decide_movement": (DecideMovement, "policy"),
            "update_positions": (UpdatePositions, "transition"),
            "find_targets": (FindTargets, "policy"),
            "hunt_prey": (HuntPrey, "transition"),
            "find_eatable_grass": (FindEatableGrass, "policy"),
            "eat_grass": (EatGrass, "transition"),
            "grow_grass": (GrowGrass, "transition")
        }
        
        for name, (cls, key) in class_registry.items():
            if name not in Registry.helpers[key]:
                Registry.register_helper(name, key)(cls)
    except Exception as e:
        print(f"Warning: Class registration error: {e}")
    
    # Results storage
    results = {
        "standard": [],
        "vmap": []
    }
    
    if use_compile:
        results["standard_compiled"] = []
        results["vmap_compiled"] = []
    
    # Run standard implementation
    print("\nRunning standard implementation...")
    
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}")
        
        # Create and initialize runner
        runner = Runner(config, registry)
        runner.init()
        
        # Run simulation
        start_time = time.time()
        runner.step(num_steps)
        end_time = time.time()
        
        time_taken = end_time - start_time
        results["standard"].append(time_taken)
        print(f"  Time: {time_taken:.4f} seconds")
    
    # Run standard with compile if available
    if use_compile:
        print("\nRunning standard implementation with torch.compile...")
        
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}")
            
            # Create and initialize runner
            runner = Runner(config, registry)
            runner.init()
            
            # Compile the step method
            runner.step_compiled = torch.compile(runner.step, fullgraph=False, dynamic=True)
            
            # Run warmup step
            _ = runner.step_compiled(1)
            runner.reset()
            
            # Run simulation
            start_time = time.time()
            runner.step_compiled(num_steps)
            end_time = time.time()
            
            time_taken = end_time - start_time
            results["standard_compiled"].append(time_taken)
            print(f"  Time: {time_taken:.4f} seconds")
    
    # Run vmap implementation
    print("\nRunning vmap implementation...")
    
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}")
        
        # Create params
        params = SimParams()
        params.num_steps = num_steps
        
        # Enable profiling if requested
        if profile and i == 0:
            import cProfile
            import pstats
            
            profiler = cProfile.Profile()
            profiler.enable()
            
            # Run simulation
            start_time = time.time()
            _, _ = run_simulation(params)
            end_time = time.time()
            
            profiler.disable()
            
            # Save profiling results
            profile_path = 'vmap_profile_results.txt'
            with open(profile_path, 'w') as f:
                stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
                stats.print_stats(30)  # Print top 30 functions
            
            print(f"Profile results saved to {profile_path}")
        else:
            # Run simulation
            start_time = time.time()
            _, _ = run_simulation(params)
            end_time = time.time()
        
        time_taken = end_time - start_time
        results["vmap"].append(time_taken)
        print(f"  Time: {time_taken:.4f} seconds")
    
    # Run vmap with compile if available
    if use_compile:
        print("\nRunning vmap implementation with torch.compile...")
        
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}")
            
            # Create params
            params = SimParams()
            params.num_steps = num_steps
            
            # Import required functions
            from agent_torch.models.predator_prey.predator_prey_vmap import simulation_step, initialize_state
            
            # Initialize state
            state = initialize_state(params)
            
            # Compile the simulation step function
            simulation_step_compiled = torch.compile(simulation_step, fullgraph=False, dynamic=True)
            
            # Run warmup step
            _ = simulation_step_compiled(state, params)
            
            # Reset state
            state = initialize_state(params)
            
            # Track stats
            stats = {
                "prey_count": [],
                "predator_count": [],
                "grass_count": []
            }
            
            # Run simulation
            start_time = time.time()
            for _ in range(num_steps):
                state = simulation_step_compiled(state, params)
                
                # Record stats
                stats["prey_count"].append((state["prey_energy"] > 0).sum().item())
                stats["predator_count"].append((state["pred_energy"] > 0).sum().item())
                stats["grass_count"].append((state["grass_growth"] == 1).sum().item())
            
            end_time = time.time()
            
            time_taken = end_time - start_time
            results["vmap_compiled"].append(time_taken)
            print(f"  Time: {time_taken:.4f} seconds")
    
    # Calculate statistics
    stats = {}
    
    for name, times in results.items():
        if times:
            mean_time = np.mean(times)
            std_time = np.std(times)
            speedup = np.mean(results["standard"]) / mean_time if mean_time > 0 else 0
            
            stats[name] = {
                "mean": mean_time,
                "std": std_time,
                "speedup": speedup
            }
    
    # Print results
    print("\nBenchmark Results:")
    for name, stat in stats.items():
        print(f"{name}: {stat['mean']:.4f} ± {stat['std']:.4f} seconds, Speedup: {stat['speedup']:.2f}x")
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Bar chart of execution times
    plt.subplot(1, 2, 1)
    names = list(stats.keys())
    means = [stats[name]["mean"] for name in names]
    stds = [stats[name]["std"] for name in names]
    
    bars = plt.bar(names, means, yerr=stds, capsize=10)
    
    # Add values on top of bars
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2., mean + 0.1,
                f'{mean:.2f}s', ha='center', va='bottom')
    
    plt.ylabel("Execution Time (seconds)")
    plt.title(f"PR Benchmark: {current_branch} ({num_steps} steps)")
    plt.xticks(rotation=45)
    
    # Speedup comparison
    plt.subplot(1, 2, 2)
    speedups = [stats[name]["speedup"] for name in names]
    
    bars = plt.bar(names, speedups)
    plt.axhline(y=1.0, color='r', linestyle='--')
    
    # Add values on top of bars
    for bar, speedup in zip(bars, speedups):
        plt.text(bar.get_x() + bar.get_width()/2., speedup + 0.1,
                f'{speedup:.2f}x', ha='center', va='bottom')
    
    plt.ylabel("Speedup vs. Standard")
    plt.title("Performance Speedup")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save results
    if save_results:
        results_dir = 'pr_benchmark_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(os.path.join(results_dir, f'pr_benchmark_{current_branch}.png'))
        
        # Save raw data
        with open(os.path.join(results_dir, f'pr_benchmark_{current_branch}.txt'), 'w') as f:
            f.write(f"PR Benchmark Results for {current_branch}\n")
            f.write(f"=================================\n\n")
            f.write(f"Configuration:\n")
            f.write(f"- Steps per run: {num_steps}\n")
            f.write(f"- Number of runs: {num_runs}\n")
            f.write(f"- Using torch.compile: {use_compile}\n\n")
            
            f.write(f"Results:\n")
            for name, stat in stats.items():
                f.write(f"{name}:\n")
                f.write(f"  Mean time: {stat['mean']:.4f} seconds\n")
                f.write(f"  Std dev: {stat['std']:.4f} seconds\n")
                f.write(f"  Speedup vs. Standard: {stat['speedup']:.2f}x\n\n")
                
                # Write individual runs
                f.write(f"  Individual runs:\n")
                for i, time_val in enumerate(results[name]):
                    f.write(f"    Run {i+1}: {time_val:.4f} seconds\n")
                f.write("\n")
        
        print(f"\nResults saved to {results_dir}")
    
    # Return statistics
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PR benchmark for predator-prey model')
    parser.add_argument('-c', '--config', default=os.path.join(AGENTTORCH_ROOT, 'agent_torch/models/predator_prey/config.yaml'),
                        help='Path to configuration file')
    parser.add_argument('--steps', type=int, default=50, 
                        help='Number of simulation steps')
    parser.add_argument('--runs', type=int, default=2, 
                        help='Number of runs per implementation')
    parser.add_argument('--compile', action='store_true', 
                        help='Enable torch.compile benchmarking')
    parser.add_argument('--profile', action='store_true', 
                        help='Enable profiling for vmap implementation')
    parser.add_argument('--no-save', action='store_true', 
                        help='Do not save benchmark results')
    
    args = parser.parse_args()
    
    run_pr_benchmark(
        config_path=args.config,
        num_steps=args.steps,
        num_runs=args.runs,
        save_results=not args.no_save,
        use_compile=args.compile,
        profile=args.profile
    )