# agent_torch/models/predator_prey/vmap_simulation.py
import torch
import random
import time
import argparse
import os
import json
import matplotlib.pyplot as plt
import tracemalloc
import sys
from pathlib import Path

# Define simulation parameters
class SimParams:
    def __init__(self):
        self.grid_size = (18, 25)  # max_x, max_y
        self.num_prey = 40
        self.num_predators = 10
        self.num_steps = 100
        self.nutrition_value = 10.0
        self.prey_energy = 50.0
        self.pred_energy = 80.0
        self.prey_work = 1.0
        self.pred_work = 2.0
        self.regrowth_time = 20

def position_to_node(pos, max_y):
    """Convert (x,y) position to node index"""
    x, y = pos
    return (max_y * x) + y

def create_grid_network(shape):
    """Create a grid adjacency matrix"""
    max_x, max_y = shape
    size = max_x * max_y
    adjacency = torch.zeros((size, size))
    
    for x in range(max_x):
        for y in range(max_y):
            node = position_to_node((x, y), max_y)
            # Add connections to adjacent cells
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < max_x and 0 <= ny < max_y:
                    neighbor = position_to_node((nx, ny), max_y)
                    adjacency[node, neighbor] = 1
    
    return adjacency

def initialize_state(params):
    """Initialize the simulation state"""
    max_x, max_y = params.grid_size
    grid_size = max_x * max_y
    
    adjacency = create_grid_network(params.grid_size)
    
    prey_pos = torch.stack([
        torch.tensor([random.randint(0, max_x-1), random.randint(0, max_y-1)])
        for _ in range(params.num_prey)
    ])
    
    pred_pos = torch.stack([
        torch.tensor([random.randint(0, max_x-1), random.randint(0, max_y-1)])
        for _ in range(params.num_predators)
    ])
    
    prey_energy = torch.full((params.num_prey, 1), params.prey_energy)
    pred_energy = torch.full((params.num_predators, 1), params.pred_energy)
    
    grass_growth = torch.zeros(grid_size, 1)
    initial_grown = torch.randperm(grid_size)[:grid_size//3]
    grass_growth[initial_grown] = 1
    
    growth_countdown = torch.randint(0, params.regrowth_time, (grid_size, 1)).float()
    
    return {
        "prey_pos": prey_pos,
        "prey_energy": prey_energy,
        "pred_pos": pred_pos,
        "pred_energy": pred_energy,
        "adjacency": adjacency,
        "grass_growth": grass_growth,
        "growth_countdown": growth_countdown,
        "bounds": torch.tensor(params.grid_size),
        "prey_work": torch.tensor(params.prey_work),
        "pred_work": torch.tensor(params.pred_work),
        "nutrition": torch.tensor(params.nutrition_value),
        "regrowth_time": torch.tensor(params.regrowth_time)
    }

# Core vmap functions

def get_neighbors(positions, adjacency, bounds):
    """Find valid neighbors for each position"""
    max_x, max_y = bounds
    all_neighbors = []
    
    for pos in positions:
        x, y = pos
        node = position_to_node((x, y), max_y)
        
        # Skip invalid nodes
        if node >= adjacency.shape[0]:
            all_neighbors.append([])
            continue
            
        # Find connected nodes
        connected = adjacency[node].nonzero().squeeze(1)
        neighbors = []
        for idx in connected:
            ny = int(idx % max_y)
            nx = int((idx - ny) / max_y)
            neighbors.append(torch.tensor([nx, ny]))
            
        all_neighbors.append(neighbors)
    
    return all_neighbors

def move_agents(positions, energy, work, neighbors):
    """Move agents to random neighbors"""
    # Check which agents have energy to move
    has_energy = energy.squeeze(-1) > 0
    
    # Update energy using tensor operations (not conditionals)
    energy_flat = energy.squeeze(-1)
    new_energy = energy_flat - work
    
    # Handle neighbor selection
    next_positions = []
    for i, (pos, can_move, nbrs) in enumerate(zip(positions, has_energy, neighbors)):
        if can_move.item() and nbrs:  # Use item() to convert tensor to Python scalar
            next_positions.append(random.choice(nbrs))
        else:
            next_positions.append(pos)
    
    return torch.stack(next_positions), new_energy.view(-1, 1)

def grow_grass_batch(growth, countdown, regrowth_time):
    """Update grass growth using vmap"""
    # Decrease countdown
    new_countdown = countdown - 1
    
    # Set growth based on countdown - using tensor operations instead of if/else
    new_growth = (new_countdown <= 0).float()
    
    return new_growth, new_countdown

def eat_grass(prey_pos, prey_energy, grass_growth, growth_countdown, bounds, nutrition, regrowth_time):
    """Prey consume grass using vectorized operations"""
    max_x, max_y = bounds
    did_eat = torch.zeros(len(prey_pos), dtype=torch.bool)
    eaten_grass_nodes = []
    
    # Find grown grass
    grown_grass = (grass_growth == 1).squeeze(-1).nonzero().squeeze(-1)
    
    # Check which prey are at grown grass positions
    for node in grown_grass:
        # Convert to position
        y = int(node % max_y)
        x = int((node - y) / max_y)
        grass_pos = torch.tensor([x, y])
        
        # Check which prey are at this position using vmap
        check_prey_at_pos = torch.vmap(
            lambda prey_p: torch.all(prey_p == grass_pos),
            in_dims=0,
            out_dims=0
        )
        prey_at_pos = check_prey_at_pos(prey_pos)
        
        if torch.any(prey_at_pos):
            did_eat = did_eat | prey_at_pos
            eaten_grass_nodes.append(node)
    
    # Update prey energy using vmap and tensor operations
    energy_flat = prey_energy.squeeze(-1)
    nutrition_value = torch.ones_like(energy_flat) * nutrition
    
    # Apply nutrition to prey that ate grass
    energy_update = did_eat.float() * nutrition_value
    new_energy = energy_flat + energy_update
    
    # Update grass state
    new_grass_growth = grass_growth.clone()
    new_growth_countdown = growth_countdown.clone()
    
    for node in eaten_grass_nodes:
        new_grass_growth[node] = 0
        new_growth_countdown[node] = regrowth_time
    
    return new_energy.view(-1, 1), new_grass_growth, new_growth_countdown

def hunt_prey(pred_pos, prey_pos, pred_energy, prey_energy, nutrition):
    """Predators hunt prey using vectorized operations"""
    # Find which prey and predators interact
    hunted_prey = torch.zeros(len(prey_pos), dtype=torch.bool)
    hunter_preds = torch.zeros(len(pred_pos), dtype=torch.bool)
    
    for i, pred_position in enumerate(pred_pos):
        # Use vmap to check all prey at once
        check_prey_matches = torch.vmap(
            lambda prey_p: torch.all(prey_p == pred_position),
            in_dims=0,
            out_dims=0
        )
        prey_matches = check_prey_matches(prey_pos)
        
        if torch.any(prey_matches):
            hunted_prey = hunted_prey | prey_matches
            hunter_preds[i] = True
    
    # Update prey energy using tensor operations
    prey_energy_flat = prey_energy.squeeze(-1)
    # Zero out energy for hunted prey
    hunted_mask = (~hunted_prey).float()
    new_prey_energy = prey_energy_flat * hunted_mask
    
    # Update predator energy
    pred_energy_flat = pred_energy.squeeze(-1)
    # Add nutrition to hunters
    nutrition_gain = hunter_preds.float() * nutrition
    new_pred_energy = pred_energy_flat + nutrition_gain
    
    return new_prey_energy.view(-1, 1), new_pred_energy.view(-1, 1)

def simulation_step(state, params, timing_data=None, memory_data=None, step_num=None):
    """Run one step of the simulation using vmap operations"""
    # Extract current state
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
    
    # Track timing if requested
    timings = {}
    memory_usage = {}
    
    # 1. Find neighbors for movement
    if timing_data is not None:
        start_time = time.time()
    
    prey_neighbors = get_neighbors(prey_pos, adjacency, bounds)
    pred_neighbors = get_neighbors(pred_pos, adjacency, bounds)
    
    if timing_data is not None:
        timings['get_neighbors'] = time.time() - start_time
    
    # 2. Move agents
    if timing_data is not None:
        start_time = time.time()
    
    prey_pos, prey_energy = move_agents(prey_pos, prey_energy, prey_work, prey_neighbors)
    pred_pos, pred_energy = move_agents(pred_pos, pred_energy, pred_work, pred_neighbors)
    
    if timing_data is not None:
        timings['move_agents'] = time.time() - start_time
    
    # 3. Grow grass
    if timing_data is not None:
        start_time = time.time()
    
    grass_growth_flat = grass_growth.squeeze(-1)
    growth_countdown_flat = growth_countdown.squeeze(-1)
    new_growth, new_countdown = grow_grass_batch(grass_growth_flat, growth_countdown_flat, regrowth_time)
    
    # Reshape back to column vectors
    grass_growth = new_growth.view(-1, 1)
    growth_countdown = new_countdown.view(-1, 1)
    
    if timing_data is not None:
        timings['grow_grass'] = time.time() - start_time
    
    # 4. Prey eat grass
    if timing_data is not None:
        start_time = time.time()
    
    prey_energy, grass_growth, growth_countdown = eat_grass(
        prey_pos, prey_energy, grass_growth, growth_countdown, bounds, nutrition, regrowth_time
    )
    
    if timing_data is not None:
        timings['eat_grass'] = time.time() - start_time
    
    # 5. Predators hunt prey
    if timing_data is not None:
        start_time = time.time()
    
    prey_energy, pred_energy = hunt_prey(pred_pos, prey_pos, pred_energy, prey_energy, nutrition)
    
    if timing_data is not None:
        timings['hunt_prey'] = time.time() - start_time
    
    # Track memory usage if requested
    if memory_data is not None and step_num is not None:
        current, peak = tracemalloc.get_traced_memory()
        memory_usage['current'] = current / 1024 / 1024  # MB
        memory_usage['peak'] = peak / 1024 / 1024  # MB
    
    # Save timing and memory data if requested
    if timing_data is not None and step_num is not None:
        timing_data[f'step_{step_num}'] = timings
    
    if memory_data is not None and step_num is not None:
        memory_data[f'step_{step_num}'] = memory_usage
    
    # Update state
    state["prey_pos"] = prey_pos
    state["prey_energy"] = prey_energy
    state["pred_pos"] = pred_pos
    state["pred_energy"] = pred_energy
    state["grass_growth"] = grass_growth
    state["growth_countdown"] = growth_countdown
    
    return state

def run_simulation(steps=100, prey=40, predators=10, grid=(18, 25), output_dir="simulation_results", 
                  animate=False, profile_memory=False):
    """Run the full simulation with the vmap implementation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create custom parameters
    params = SimParams()
    params.grid_size = grid
    params.num_prey = prey
    params.num_predators = predators
    params.num_steps = steps
    
    # Initialize
    state = initialize_state(params)
    
    # Track stats
    stats = {
        "prey_count": [],
        "predator_count": [],
        "grass_count": []
    }
    
    # For profiling
    timing_data = {} if profile_memory else None
    memory_data = {} if profile_memory else None
    
    # Start tracking memory if requested
    if profile_memory:
        tracemalloc.start()
    
    # Run simulation
    start_time = time.time()
    for step in range(params.num_steps):
        # Update state
        state = simulation_step(state, params, timing_data, memory_data, step)
        
        # Count living agents and grown grass
        living_prey = (state["prey_energy"] > 0).sum().item()
        living_predators = (state["pred_energy"] > 0).sum().item()
        grown_grass = (state["grass_growth"] == 1).sum().item()
        
        # Record stats
        stats["prey_count"].append(living_prey)
        stats["predator_count"].append(living_predators)
        stats["grass_count"].append(grown_grass)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Simulation completed in {execution_time:.4f} seconds")
    
    # Stop tracking memory if requested
    if profile_memory:
        tracemalloc.stop()
    
    # Create results dictionary
    results = {
        'execution_time': execution_time,
        'steps': steps,
        'prey': prey,
        'predators': predators,
        'grid_size': list(grid),
        'stats': stats
    }
    
    # Add profiling data if collected
    if timing_data:
        results['timing_data'] = timing_data
    
    if memory_data:
        results['memory_data'] = memory_data
    
    # Save results
    with open(os.path.join(output_dir, 'simulation_results.json'), 'w') as f:
        json.dump(results, f)
    
    # Plot results if requested
    if animate:
        plt.figure(figsize=(10, 6))
        plt.plot(stats["prey_count"], label="Prey", color="blue")
        plt.plot(stats["predator_count"], label="Predators", color="red")
        plt.plot(stats["grass_count"], label="Grass", color="green")
        plt.xlabel("Step")
        plt.ylabel("Count")
        plt.title("Predator-Prey Population Dynamics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "simulation_results.png"))
        plt.show()
    
    # Plot profiling data if collected
    if timing_data:
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        steps = list(range(params.num_steps))
        substeps = next(iter(timing_data.values())).keys()
        
        for substep in substeps:
            values = [timing_data[f'step_{i}'][substep] for i in steps]
            plt.plot(steps, values, label=substep)
        
        plt.xlabel("Step")
        plt.ylabel("Time (seconds)")
        plt.title("Execution Time by Substep")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "timing_profile.png"))
    
    if memory_data:
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        steps = list(range(params.num_steps))
        current_memory = [memory_data[f'step_{i}']['current'] for i in steps]
        peak_memory = [memory_data[f'step_{i}']['peak'] for i in steps]
        
        plt.plot(steps, current_memory, label="Current Memory", color="blue")
        plt.plot(steps, peak_memory, label="Peak Memory", color="red")
        
        plt.xlabel("Step")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Memory Usage During Simulation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "memory_profile.png"))
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run predator-prey simulation with vmap')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to run')
    parser.add_argument('--prey', type=int, default=40, help='Number of prey')
    parser.add_argument('--predators', type=int, default=10, help='Number of predators')
    parser.add_argument('--grid', type=str, default='18,25', help='Grid size (x,y)')
    parser.add_argument('--output', default='simulation_results', help='Output directory')
    parser.add_argument('--animate', action='store_true', help='Create animation and show plot')
    parser.add_argument('--profile-memory', action='store_true', help='Profile memory usage')
    
    args = parser.parse_args()
    
    # Parse grid size
    grid_x, grid_y = map(int, args.grid.split(','))
    
    # Run simulation
    run_simulation(
        steps=args.steps,
        prey=args.prey,
        predators=args.predators,
        grid=(grid_x, grid_y),
        output_dir=args.output,
        animate=args.animate,
        profile_memory=args.profile_memory
    )