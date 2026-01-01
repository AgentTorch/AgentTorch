#!/usr/bin/env python3
"""
Boids Flocking Simulation with Visualization

This script runs the boids simulation with visualization using matplotlib.
It demonstrates emergent flocking behavior and collective movement patterns.

Usage:
    python -m agent_torch.examples.run_boids_sim
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from agent_torch.populations import sample2
from agent_torch.examples.models import boids
from agent_torch.core.environment import envs


class BoidsVisualizer:
    """Handles visualization of the boids simulation."""
    
    def __init__(self, bounds: Tuple[float, float] = (800, 600)):
        self.bounds = bounds
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.setup_plot()
    
    def setup_plot(self):
        """Setup the matplotlib plot."""
        self.ax.set_xlim(0, self.bounds[0])
        self.ax.set_ylim(0, self.bounds[1])
        self.ax.set_aspect('equal')
        self.ax.set_title('Boids Flocking Simulation - Emergent Collective Behavior', 
                         fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X Position', fontsize=12)
        self.ax.set_ylabel('Y Position', fontsize=12)
        self.ax.grid(True, alpha=0.3)
    
    def plot_final_state(self, positions: torch.Tensor, velocities: torch.Tensor, step: int):
        """Plot the final state of the simulation."""
        # Convert tensors to numpy (handle gradients properly)
        pos_np = positions.detach().cpu().numpy()
        vel_np = velocities.detach().cpu().numpy()
        
        # Clear and setup plot
        self.ax.clear()
        self.setup_plot()
        
        # Plot agents as points with velocity vectors
        self.ax.scatter(pos_np[:, 0], pos_np[:, 1], 
                       c='blue', s=30, alpha=0.8, edgecolors='darkblue', 
                       linewidth=0.5, label='Boids')
        
        # Plot velocity vectors (scaled for visibility)
        scale = 15.0
        self.ax.quiver(pos_np[:, 0], pos_np[:, 1], 
                      vel_np[:, 0] * scale, vel_np[:, 1] * scale,
                      color='red', alpha=0.7, width=0.003, headwidth=3,
                      label='Velocity')
        
        # Add step counter and statistics
        avg_speed = np.linalg.norm(vel_np, axis=1).mean()
        center = pos_np.mean(axis=0)
        
        self.ax.text(0.02, 0.98, f'Final Step: {step}', 
                    transform=self.ax.transAxes, fontsize=14, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        self.ax.text(0.02, 0.02, 
                    f'Agents: {len(pos_np)}\nAvg Speed: {avg_speed:.2f}\nCenter: ({center[0]:.1f}, {center[1]:.1f})', 
                    transform=self.ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Add legend
        self.ax.legend(loc='upper right')


def run_boids_simulation_with_viz():
    """Run the boids simulation with visualization."""
    print("Starting boids flocking simulation with visualization...")
    
    try:
        # Create simulation runner (same as simple version)
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
            
            # Get initial state (same as simple version)
            positions = runner.state["agents"]["boids"]["position"]
            velocities = runner.state["agents"]["boids"]["velocity"]
            
            print(f"Initial state:")
            print(f"  Position shape: {positions.shape}")
            print(f"  Velocity shape: {velocities.shape}")
            print(f"  Avg position: ({positions.mean(dim=0)[0]:.1f}, {positions.mean(dim=0)[1]:.1f})")
            print(f"  Avg speed: {torch.norm(velocities, dim=1).mean():.2f}")
            print(f"  Position range: ({positions.min(dim=0)[0][0]:.0f}-{positions.max(dim=0)[0][0]:.0f}, {positions.min(dim=0)[0][1]:.0f}-{positions.max(dim=0)[0][1]:.0f})")
            
            # Run simulation steps (same logic as simple version)
            for step in range(sim_steps):
                runner.step(1)  # Step once
                
                # Print statistics every 10 steps
                if step % 10 == 0 or step == sim_steps - 1:
                    positions = runner.state["agents"]["boids"]["position"]
                    velocities = runner.state["agents"]["boids"]["velocity"]
                    
                    avg_position = positions.mean(dim=0)
                    avg_speed = torch.norm(velocities, dim=1).mean()
                    min_pos = positions.min(dim=0)[0]
                    max_pos = positions.max(dim=0)[0]
                    
                    print(f"  Step {step + 1:3d}: Avg pos: ({avg_position[0]:.1f}, {avg_position[1]:.1f}), "
                          f"Avg speed: {avg_speed:.2f}, Range: ({min_pos[0]:.0f}-{max_pos[0]:.0f}, {min_pos[1]:.0f}-{max_pos[1]:.0f})")
        
        print("\nSimulation completed!")
        
        # Visualize final state
        print("Creating visualization...")
        final_positions = runner.state["agents"]["boids"]["position"]
        final_velocities = runner.state["agents"]["boids"]["velocity"]
        bounds = runner.config["state"]["environment"]["bounds"]["value"]
        
        # Create visualizer and plot
        visualizer = BoidsVisualizer(bounds=(bounds[0], bounds[1]))
        visualizer.plot_final_state(final_positions, final_velocities, sim_steps)
        
        # Show plot
        plt.tight_layout()
        plt.savefig('boids_final_state.png', dpi=150, bbox_inches='tight')
        print("Visualization saved as 'boids_final_state.png'")
        plt.show()
        
        print("Visualization completed successfully!")
        
    except Exception as e:
        print(f"Failed to run simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_boids_simulation_with_viz() 