#!/usr/bin/env python3
"""
Run the boids flocking simulation.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from agent_torch.populations import sample2
from agent_torch.examples.models import boids
from agent_torch.core.environment import envs

def run_boids_simulation():
    """Run the boids flocking simulation."""
    print("Starting boids flocking simulation...")
    
    # Create simulation runner
    runner = envs.create(
        model=boids,
        population=sample2
    )
    
    # Get simulation parameters
    sim_steps = runner.config["simulation_metadata"]["num_steps_per_episode"]
    num_episodes = runner.config["simulation_metadata"]["num_episodes"]
    num_agents = runner.config["simulation_metadata"]["num_agents"]
    
    print(f"Simulation parameters:")
    print(f"  - Agents: {num_agents}")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Steps per episode: {sim_steps}")
    
    # Store trajectory for visualization
    trajectory = []
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\nRunning episode {episode + 1}/{num_episodes}...")
        
        if episode > 0:
            runner.reset()
        
        # Store initial positions
        positions = runner.state["agents"]["boids"]["position"]
        velocities = runner.state["agents"]["boids"]["velocity"]
        trajectory.append({
            "step": 0,
            "positions": positions.clone().cpu(),
            "velocities": velocities.clone().cpu()
        })
        
        # Run simulation steps
        for step in range(sim_steps):
            runner.step(1)  # Step once
            
            # Store positions every 10 steps for visualization
            if step % 10 == 0 or step == sim_steps - 1:
                positions = runner.state["agents"]["boids"]["position"]
                velocities = runner.state["agents"]["boids"]["velocity"]
                trajectory.append({
                    "step": step + 1,
                    "positions": positions.clone().cpu(),
                    "velocities": velocities.clone().cpu()
                })
                
                # Print some statistics
                avg_position = positions.mean(dim=0)
                avg_speed = torch.norm(velocities, dim=1).mean()
                print(f"  Step {step + 1:3d}: Avg position: ({avg_position[0]:.1f}, {avg_position[1]:.1f}), Avg speed: {avg_speed:.2f}")
    
    print("\nSimulation completed!")
    
    # Visualize the final state
    visualize_boids(trajectory[-1], runner.config)
    
    return runner, trajectory

def visualize_boids(data, config):
    """Visualize the boids positions and velocities."""
    positions = data["positions"]
    velocities = data["velocities"]
    
    # Get environment bounds
    bounds = config["state"]["environment"]["bounds"]["value"]
    
    plt.figure(figsize=(12, 8))
    
    # Plot positions as dots
    plt.scatter(positions[:, 0], positions[:, 1], c='blue', s=20, alpha=0.7, label='Boids')
    
    # Plot velocity vectors
    scale = 10  # Scale factor for arrow length
    plt.quiver(positions[:, 0], positions[:, 1], 
              velocities[:, 0], velocities[:, 1],
              angles='xy', scale_units='xy', scale=1/scale, 
              color='red', alpha=0.6, width=0.003)
    
    plt.xlim(0, bounds[0])
    plt.ylim(0, bounds[1])
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Boids Flocking Simulation - Step {data["step"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save plot
    plt.savefig('boids_simulation.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'boids_simulation.png'")
    plt.show()

def analyze_flocking_behavior(trajectory):
    """Analyze the flocking behavior over time."""
    steps = []
    cohesion_scores = []
    alignment_scores = []
    
    for data in trajectory:
        positions = data["positions"]
        velocities = data["velocities"]
        
        # Cohesion: measure spread of positions (lower = more cohesive)
        center = positions.mean(dim=0)
        distances_to_center = torch.norm(positions - center, dim=1)
        cohesion = distances_to_center.mean().item()
        
        # Alignment: measure how aligned velocities are (higher = more aligned)
        vel_norms = torch.norm(velocities, dim=1, keepdim=True)
        normalized_vels = velocities / (vel_norms + 1e-8)
        pairwise_dots = torch.mm(normalized_vels, normalized_vels.T)
        alignment = pairwise_dots.mean().item()
        
        steps.append(data["step"])
        cohesion_scores.append(cohesion)
        alignment_scores.append(alignment)
    
    # Plot analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(steps, cohesion_scores, 'b-', linewidth=2)
    ax1.set_ylabel('Cohesion (avg distance to center)')
    ax1.set_title('Flocking Behavior Analysis')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, alignment_scores, 'r-', linewidth=2)
    ax2.set_ylabel('Alignment (velocity similarity)')
    ax2.set_xlabel('Simulation Step')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('boids_analysis.png', dpi=150, bbox_inches='tight')
    print("Analysis saved as 'boids_analysis.png'")
    plt.show()

if __name__ == "__main__":
    try:
        runner, trajectory = run_boids_simulation()
        print("\nAnalyzing flocking behavior...")
        analyze_flocking_behavior(trajectory)
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc() 