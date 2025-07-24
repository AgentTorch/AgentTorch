#!/usr/bin/env python3
"""
Run the boids flocking simulation - simplified version without visualization.
"""

import torch
from agent_torch.populations import sample2
from agent_torch.examples.models import boids
from agent_torch.core.environment import envs

def run_boids_simulation_simple():
    """Run the boids flocking simulation without visualization."""
    print("Starting boids flocking simulation...")
    
    try:
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
        
        # Run episodes
        for episode in range(num_episodes):
            print(f"\nRunning episode {episode + 1}/{num_episodes}...")
            
            if episode > 0:
                runner.reset()
            
            # Get initial state
            positions = runner.state["agents"]["boids"]["position"]
            velocities = runner.state["agents"]["boids"]["velocity"]
            
            print(f"Initial state:")
            print(f"  Position shape: {positions.shape}")
            print(f"  Velocity shape: {velocities.shape}")
            print(f"  Avg position: ({positions.mean(dim=0)[0]:.1f}, {positions.mean(dim=0)[1]:.1f})")
            print(f"  Avg speed: {torch.norm(velocities, dim=1).mean():.2f}")
            print(f"  Position range: ({positions.min(dim=0)[0][0]:.0f}-{positions.max(dim=0)[0][0]:.0f}, {positions.min(dim=0)[0][1]:.0f}-{positions.max(dim=0)[0][1]:.0f})")
            
            # Run simulation steps
            for step in range(min(sim_steps, 50)):  # Limit to 50 steps for testing
                runner.step(1)  # Step once
                
                # Print statistics every 10 steps
                if step % 10 == 0 or step == min(sim_steps, 50) - 1:
                    positions = runner.state["agents"]["boids"]["position"]
                    velocities = runner.state["agents"]["boids"]["velocity"]
                    
                    avg_position = positions.mean(dim=0)
                    avg_speed = torch.norm(velocities, dim=1).mean()
                    min_pos = positions.min(dim=0)[0]
                    max_pos = positions.max(dim=0)[0]
                    
                    print(f"  Step {step + 1:3d}: Avg pos: ({avg_position[0]:.1f}, {avg_position[1]:.1f}), "
                          f"Avg speed: {avg_speed:.2f}, Range: ({min_pos[0]:.0f}-{max_pos[0]:.0f}, {min_pos[1]:.0f}-{max_pos[1]:.0f})")
        
        print("\nSimulation completed successfully!")
        return runner
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    runner = run_boids_simulation_simple() 