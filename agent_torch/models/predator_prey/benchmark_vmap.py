# agent_torch/models/predator_prey/benchmark_vmap.py
import argparse
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import cProfile
import pstats
import io
import tracemalloc
import traceback
import yaml

# Get the current directory and add it to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Try to import torch.compile if available
COMPILE_AVAILABLE = False
try:
    import torch
    if hasattr(torch, 'compile'):
        COMPILE_AVAILABLE = True
except ImportError:
    pass

# Import vmap runner
from vmap_run import run_simulation as run_vmap_simulation

def load_config(config_path):
    """Load configuration from a YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return None

def run_standard_implementation(config_path, num_steps=100, profile=False, use_compile=False):
    """
    Run the standard implementation using parameters from the config file
    """
    try:
        # Start profiling if requested
        profiler = None
        if profile:
            tracemalloc.start()
            profiler = cProfile.Profile()
            profiler.enable()
        
        # Load the config file
        config = load_config(config_path)
        if config is None:
            print("Using default parameters since config file couldn't be loaded")
            grid_size = (18, 25)
            num_prey = 40
            num_predators = 10
            regrowth_time = 20
            nutrition_value = 10.0
            prey_energy = 50.0
            pred_energy = 80.0
            prey_work = 1.0
            pred_work = 2.0
        else:
            # Extract parameters from config
            metadata = config.get('simulation_metadata', {})
            grid_size = (metadata.get('max_x', 18), metadata.get('max_y', 25))
            num_prey = metadata.get('num_prey', 40)
            num_predators = metadata.get('num_predators', 10)
            
            # Get more parameters if available
            try:
                # Get grass regrowth time
                regrowth_time = float(config['state']['objects']['grass']['properties']['regrowth_time']['value'])
            except (KeyError, TypeError):
                regrowth_time = 20
                
            try:
                # Get nutritional value
                nutrition_value = float(config['state']['objects']['grass']['properties']['nutritional_value']['value'])
            except (KeyError, TypeError):
                nutrition_value = 10.0
                
            try:
                # Get prey energy
                prey_energy = float(config['state']['agents']['prey']['properties']['energy']['arguments']['upper_limit']['value'])
            except (KeyError, TypeError):
                prey_energy = 50.0
                
            try:
                # Get predator energy
                pred_energy = float(config['state']['agents']['predator']['properties']['energy']['arguments']['upper_limit']['value'])
            except (KeyError, TypeError):
                pred_energy = 80.0
                
            try:
                # Get prey work
                prey_work = float(config['state']['agents']['prey']['properties']['stride_work']['value'])
            except (KeyError, TypeError):
                prey_work = 1.0
                
            try:
                # Get predator work
                pred_work = float(config['state']['agents']['predator']['properties']['stride_work']['value'])
            except (KeyError, TypeError):
                pred_work = 2.0
        
        print(f"Using parameters: grid_size={grid_size}, prey={num_prey}, predators={num_predators}")
        print(f"Additional parameters: regrowth_time={regrowth_time}, nutrition={nutrition_value}")
        print(f"Energy parameters: prey_energy={prey_energy}, pred_energy={pred_energy}")
        print(f"Work parameters: prey_work={prey_work}, pred_work={pred_work}")
        
        # Create a simple standard implementation
        print(f"Creating standard implementation simulation...")
        
        # Initialize grid and agents
        grid_area = grid_size[0] * grid_size[1]
        
        # Create a simulation state
        import torch
        import random
        
        # Initialize state similar to the vmap implementation
        adjacency = torch.zeros((grid_area, grid_area))
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                node = (grid_size[1] * x) + y
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]:
                        neighbor = (grid_size[1] * nx) + ny
                        adjacency[node, neighbor] = 1
        
        prey_pos = torch.stack([
            torch.tensor([random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1)])
            for _ in range(num_prey)
        ])
        
        pred_pos = torch.stack([
            torch.tensor([random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1)])
            for _ in range(num_predators)
        ])
        
        prey_energy = torch.full((num_prey, 1), prey_energy)
        pred_energy = torch.full((num_predators, 1), pred_energy)
        
        grass_growth = torch.zeros(grid_area, 1)
        initial_grown = torch.randperm(grid_area)[:grid_area//3]
        grass_growth[initial_grown] = 1
        
        growth_countdown = torch.randint(0, int(regrowth_time), (grid_area, 1)).float()
        
        state = {
            "prey_pos": prey_pos,
            "prey_energy": prey_energy,
            "pred_pos": pred_pos,
            "pred_energy": pred_energy,
            "adjacency": adjacency,
            "grass_growth": grass_growth,
            "growth_countdown": growth_countdown,
            "bounds": torch.tensor(grid_size),
            "prey_work": torch.tensor(prey_work),
            "pred_work": torch.tensor(pred_work),
            "nutrition": torch.tensor(nutrition_value),
            "regrowth_time": torch.tensor(regrowth_time),
            "step": 0
        }
        
        # Define a standard simulation step function
        def standard_simulation_step(state):
            # Extract state variables
            prey_pos = state["prey_pos"]
            prey_energy = state["prey_energy"]
            pred_pos = state["pred_pos"]
            pred_energy = state["pred_energy"]
            adjacency = state["adjacency"]
            grass_growth = state["grass_growth"]
            growth_countdown = state["growth_countdown"]
            bounds = state["bounds"]
            prey_work = state["prey_work"]
            pred_work = state["pred_work"]
            nutrition = state["nutrition"]
            regrowth_time = state["regrowth_time"]
            
            # The simulation step will mimic the basic behavior without vmap
            # This is a simplified version that doesn't use the full AgentTorch machinery
            # but gives us a comparative baseline for timing purposes
            
            # 1. Movement
            # Get neighbors for each position
            prey_neighbors = []
            for pos in prey_pos:
                x, y = pos
                max_x, max_y = bounds
                node = (max_y * x) + y
                
                if node >= adjacency.shape[0]:
                    prey_neighbors.append([])
                    continue
                
                connected = adjacency[node].nonzero().squeeze(1)
                neighbors = []
                for idx in connected:
                    ny = int(idx % max_y)
                    nx = int((idx - ny) / max_y)
                    neighbors.append(torch.tensor([nx, ny]))
                
                prey_neighbors.append(neighbors)
            
            pred_neighbors = []
            for pos in pred_pos:
                x, y = pos
                max_x, max_y = bounds
                node = (max_y * x) + y
                
                if node >= adjacency.shape[0]:
                    pred_neighbors.append([])
                    continue
                
                connected = adjacency[node].nonzero().squeeze(1)
                neighbors = []
                for idx in connected:
                    ny = int(idx % max_y)
                    nx = int((idx - ny) / max_y)
                    neighbors.append(torch.tensor([nx, ny]))
                
                pred_neighbors.append(neighbors)
            
            # Move agents
            for i, (pos, energy, nbrs) in enumerate(zip(prey_pos, prey_energy, prey_neighbors)):
                if energy > 0 and nbrs:
                    prey_pos[i] = random.choice(nbrs)
                    prey_energy[i] = energy - prey_work
            
            for i, (pos, energy, nbrs) in enumerate(zip(pred_pos, pred_energy, pred_neighbors)):
                if energy > 0 and nbrs:
                    pred_pos[i] = random.choice(nbrs)
                    pred_energy[i] = energy - pred_work
            
            # 2. Grow grass
            for i in range(len(growth_countdown)):
                growth_countdown[i] -= 1
                if growth_countdown[i] <= 0:
                    grass_growth[i] = 1
            
            # 3. Prey eat grass
            max_x, max_y = bounds
            for i, (pos, energy) in enumerate(zip(prey_pos, prey_energy)):
                if energy > 0:
                    x, y = pos
                    node = (max_y * x) + y
                    if node < len(grass_growth) and grass_growth[node] == 1:
                        prey_energy[i] += nutrition
                        grass_growth[node] = 0
                        growth_countdown[node] = regrowth_time
            
            # 4. Predators hunt prey
            for i, (pred_p, pred_e) in enumerate(zip(pred_pos, pred_energy)):
                if pred_e > 0:
                    for j, (prey_p, prey_e) in enumerate(zip(prey_pos, prey_energy)):
                        if prey_e > 0 and torch.all(pred_p == prey_p):
                            pred_energy[i] += nutrition
                            prey_energy[j] = 0
            
            return state
        
        # Compile the standard step if requested
        if use_compile and COMPILE_AVAILABLE:
            print("Compiling standard simulation step with torch.compile...")
            standard_simulation_step = torch.compile(standard_simulation_step)
        
        # Run the standard implementation
        print(f"Running {num_steps} steps with standard implementation...")
        start_time = time.time()
        
        # Run simulation steps
        for step in range(num_steps):
            state["step"] = step
            state = standard_simulation_step(state)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Standard implementation completed in {execution_time:.4f} seconds")
        
        # Collect profiling results if requested
        profile_results = None
        memory_stats = None
        
        if profile:
            if profiler:
                profiler.disable()
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(30)
                profile_results = s.getvalue()
            
            current, peak = tracemalloc.get_traced_memory()
            memory_stats = {
                "current_memory_mb": current / 1024 / 1024,
                "peak_memory_mb": peak / 1024 / 1024
            }
            tracemalloc.stop()
        
        return execution_time, {"profile": profile_results, "memory": memory_stats}
    
    except Exception as e:
        print(f"Error running standard implementation: {e}")
        print(traceback.format_exc())
        return None, None

def run_benchmark(config_path, num_steps=100, num_runs=3, output_dir="benchmark_results", 
                 profile=False, use_compile=False):
    """Run benchmarking comparing standard and vmap implementations"""
    print(f"Benchmarking with {num_steps} steps and {num_runs} runs each")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config to get parameters
    config = load_config(config_path)
    if config is None:
        print("Using default parameters for vmap implementation")
        prey = 40
        predators = 10
        grid = (18, 25)
    else:
        metadata = config.get('simulation_metadata', {})
        grid = (metadata.get('max_x', 18), metadata.get('max_y', 25))
        prey = metadata.get('num_prey', 40)
        predators = metadata.get('num_predators', 10)
    
    # Store results
    standard_times = []
    vmap_times = []
    standard_profile = None
    vmap_profile = None
    
    # Run standard implementation
    print("\nRunning standard implementation:")
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}")
        # Only profile the first run
        execution_time, profile_data = run_standard_implementation(
            config_path=config_path,
            num_steps=num_steps, 
            profile=(profile and i == 0),
            use_compile=use_compile
        )
        if execution_time is not None:
            standard_times.append(execution_time)
            print(f"  Time: {execution_time:.4f} seconds")
            
            # Save profile data from the first run
            if i == 0 and profile_data and profile_data["profile"]:
                standard_profile = profile_data
    
    # Run vmap implementation
    print("\nRunning vmap implementation:")
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}")
        
        # Only profile the first run
        profile_this_run = profile and i == 0
        results = run_vmap_simulation(
            steps=num_steps,
            prey=prey,
            predators=predators,
            grid=grid,
            profile_memory=profile_this_run
        )
        
        if results is not None and 'execution_time' in results:
            execution_time = results['execution_time']
            vmap_times.append(execution_time)
            print(f"  Time: {execution_time:.4f} seconds")
            
            # Save profile data from the first run
            if i == 0 and "timing_data" in results and "memory_data" in results:
                vmap_profile = {
                    "timing_data": results["timing_data"],
                    "memory_data": results["memory_data"]
                }
    
    # Save profile results if we have them
    if standard_profile and standard_profile["profile"]:
        with open(os.path.join(output_dir, 'standard_profile_results.txt'), 'w') as f:
            f.write(standard_profile["profile"])
        print(f"Standard profile results saved to {os.path.join(output_dir, 'standard_profile_results.txt')}")
    
    # Calculate statistics
    can_calculate_speedup = standard_times and vmap_times
    
    if vmap_times:
        vmap_mean = np.mean(vmap_times)
        vmap_std = np.std(vmap_times)
        
        # Print vmap results
        print("\nVmap Implementation Results:")
        print(f"Vmap: {vmap_mean:.4f} ± {vmap_std:.4f} seconds")
    
    if standard_times:
        standard_mean = np.mean(standard_times)
        standard_std = np.std(standard_times)
        
        # Print standard results
        print("\nStandard Implementation Results:")
        print(f"Standard: {standard_mean:.4f} ± {standard_std:.4f} seconds")
    
    # If we have both standard and vmap results, calculate speedup
    if can_calculate_speedup:
        speedup = standard_mean / vmap_mean
        print("\nSpeedup Results:")
        print(f"Speedup: {speedup:.2f}x (standard / vmap)")
        
        # Save combined data
        with open(os.path.join(output_dir, 'benchmark_data.txt'), 'w') as f:
            f.write("Implementation,Run,Time(s)\n")
            for i, t in enumerate(standard_times):
                f.write(f"Standard,{i+1},{t:.6f}\n")
            for i, t in enumerate(vmap_times):
                f.write(f"Vmap,{i+1},{t:.6f}\n")
            
            f.write("\nSummary:\n")
            f.write(f"Standard: {standard_mean:.4f} ± {standard_std:.4f} seconds\n")
            f.write(f"Vmap: {vmap_mean:.4f} ± {vmap_std:.4f} seconds\n")
            f.write(f"Speedup: {speedup:.2f}x\n")
        
        # Create comparison plot
        plt.figure(figsize=(15, 10))
        
        # Bar plot for execution times
        plt.subplot(2, 2, 1)
        bars = plt.bar(
            ['Standard', 'Vmap'], 
            [standard_mean, vmap_mean],
            yerr=[standard_std, vmap_std],
            capsize=10,
            color=['blue', 'orange']
        )
        
        # Add values on top of bars
        for bar, time_val in zip(bars, [standard_mean, vmap_mean]):
            plt.text(bar.get_x() + bar.get_width()/2, time_val + max(standard_std, vmap_std) + 0.1,
                   f'{time_val:.2f}s', ha='center')
        
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'Execution Time Comparison ({num_steps} steps)')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Speedup visualization
        plt.subplot(2, 2, 2)
        speedup_bar = plt.bar(['Speedup'], [speedup], color='green')
        plt.text(speedup_bar[0].get_x() + speedup_bar[0].get_width()/2, 
               speedup + 0.1, f'{speedup:.2f}x', ha='center')
        
        plt.ylabel('Speedup Factor')
        plt.title('Vmap Implementation Speedup')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Individual run times
        plt.subplot(2, 2, 3)
        x_standard = np.arange(len(standard_times))
        x_vmap = np.arange(len(vmap_times))
        plt.plot(x_standard, standard_times, 'o-', label='Standard')
        plt.plot(x_vmap, vmap_times, 'o-', label='Vmap')
        plt.xlabel('Run #')
        plt.ylabel('Time (seconds)')
        plt.title('Individual Run Times')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Additional analysis - speedup ratio per run
        if len(standard_times) == len(vmap_times):
            plt.subplot(2, 2, 4)
            individual_speedups = [s/v for s, v in zip(standard_times, vmap_times)]
            plt.bar(range(len(individual_speedups)), individual_speedups, color='purple')
            plt.axhline(y=speedup, color='r', linestyle='--', label='Average')
            plt.xlabel('Run #')
            plt.ylabel('Speedup Factor')
            plt.title('Speedup by Run')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'))
        print(f"Comparison plot saved to {os.path.join(output_dir, 'benchmark_comparison.png')}")
    
    # Return all collected data
    return {
        "standard": {
            "times": standard_times, 
            "mean": np.mean(standard_times) if standard_times else None,
            "std": np.std(standard_times) if standard_times else None,
            "profile": standard_profile
        },
        "vmap": {
            "times": vmap_times, 
            "mean": np.mean(vmap_times) if vmap_times else None,
            "std": np.std(vmap_times) if vmap_times else None,
            "profile": vmap_profile
        },
        "speedup": standard_mean / vmap_mean if can_calculate_speedup else None
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark standard vs. vmap implementations')
    parser.add_argument('-c', '--config', 
                      default=os.path.join(current_dir, 'config.yaml'),
                      help='Path to configuration file')
    parser.add_argument('--steps', type=int, default=100, 
                      help='Number of simulation steps')
    parser.add_argument('--runs', type=int, default=3, 
                      help='Number of runs per implementation')
    parser.add_argument('--output', default='benchmark_results', 
                      help='Output directory')
    parser.add_argument('--profile', action='store_true',
                      help='Enable profiling for both implementations')
    parser.add_argument('--use-compile', action='store_true',
                      help='Use torch.compile if available')
    
    args = parser.parse_args()
    
    run_benchmark(
        config_path=args.config,
        num_steps=args.steps,
        num_runs=args.runs,
        output_dir=args.output,
        profile=args.profile,
        use_compile=args.use_compile
    )