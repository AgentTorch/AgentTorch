#!/usr/bin/env python
"""
Run the predator-prey simulation using vectorized operations.

This script provides a command-line interface to run the predator-prey simulation
with the vectorized implementation, allowing you to set parameters and visualize results.
"""
import os
import sys
import argparse
import time
import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

# Add the parent directory to sys.path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from agent_torch.core import Registry, Runner, VectorizedRunner
from agent_torch.core.helpers import read_config, read_from_file, grid_network

# Import both standard and vectorized substeps with absolute paths
from agent_torch.models.predator_prey.substeps import *
from agent_torch.models.predator_prey.vmap_substeps import *
from agent_torch.models.predator_prey.helpers.map import map_network
from agent_torch.models.predator_prey.helpers.random import random_float, random_int

try:
    from agent_torch.models.predator_prey.plot import Plot
except ImportError:
    Plot = None
    print("Warning: Plot module not found. Visualization will be disabled.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run predator-prey simulation with vectorized implementation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_vmap.yaml",
        help="Path to YAML config file (default: config_vmap.yaml)",
    )
    parser.add_argument("--episodes", type=int, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, help="Number of steps per episode")
    parser.add_argument("--predators", type=int, help="Override number of predators")
    parser.add_argument("--prey", type=int, help="Override number of prey")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for visualizations (default: results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run the simulation on (default: cpu)",
    )
    return parser.parse_args()


def custom_read_from_file(shape, params):
    """
    Custom file reader that handles relative paths correctly.
    """
    file_path = params["file_path"]

    # Make path absolute if needed
    if not os.path.isabs(file_path):
        file_path = os.path.join(current_dir, file_path)

    print(f"Reading file: {file_path}")

    if file_path.endswith("csv"):
        import pandas as pd

        data = pd.read_csv(file_path)

    data_values = data.values
    assert data_values.shape == tuple(
        shape
    ), f"Shape mismatch: {data_values.shape} vs expected {shape}"

    data_tensor = torch.tensor(data_values, dtype=torch.float32)
    return data_tensor


def setup_registry():
    """Set up the registry with all necessary functions."""
    registry = Registry()
    registry.register(custom_read_from_file, "read_from_file", "initialization")
    registry.register(grid_network, "grid", key="network")
    registry.register(map_network, "map", key="network")
    registry.register(random_float, "random_float", "initialization")
    registry.register(random_int, "random_int", "initialization")
    return registry


def run_simulation(config_path, args):
    """Run the simulation with the given config and args."""
    # Read configuration file
    if not os.path.isabs(config_path):
        config_path = os.path.join(current_dir, config_path)

    print(f"Reading config from: {config_path}")
    config = read_config(config_path)

    # Override config with command-line arguments
    if args.episodes:
        config["simulation_metadata"]["num_episodes"] = args.episodes
    if args.steps:
        config["simulation_metadata"]["num_steps_per_episode"] = args.steps
    if args.predators:
        config["simulation_metadata"]["num_predators"] = args.predators
    if args.prey:
        config["simulation_metadata"]["num_prey"] = args.prey
    if args.visualize:
        config["simulation_metadata"]["visualize"] = True
    if args.device:
        config["simulation_metadata"]["device"] = args.device

    # Get simulation parameters from config
    metadata = config.get("simulation_metadata")
    num_episodes = metadata.get("num_episodes")
    num_steps_per_episode = metadata.get("num_steps_per_episode")
    visualize = metadata.get("visualize")

    # Set up registry and runner
    registry = setup_registry()

    # Use vectorized runner
    print(":: initializing vectorized runner")

    # Create and initialize runner
    runner = VectorizedRunner(config, registry)
    runner.init()

    print(":: simulation started")

    # Set up visualization if enabled
    if visualize and Plot is not None:
        # Create output directory if it doesn't exist
        output_dir = os.path.join(current_dir, args.output)
        plots_dir = os.path.join(output_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        visual = Plot(metadata.get("max_x"), metadata.get("max_y"))
    else:
        visual = None
        visualize = False

    # Collect statistics
    stats = {
        "episode": [],
        "step": [],
        "predators_alive": [],
        "prey_alive": [],
        "grass_grown": [],
        "time": [],
    }

    # Run simulation episodes
    for episode in range(num_episodes):
        print(f":: starting episode {episode+1}/{num_episodes}")
        runner.reset()

        # Run steps in each episode
        for step in trange(
            num_steps_per_episode, desc=f"Episode {episode+1}/{num_episodes}"
        ):
            # Track step time
            start_time = time.time()

            # Run simulation step
            runner.step(1)

            # Record statistics
            current_state = runner.state

            # Get alive counts
            pred_alive = (
                (current_state["agents"]["predator"]["energy"] > 0).sum().item()
            )
            prey_alive = (current_state["agents"]["prey"]["energy"] > 0).sum().item()
            grass_grown = (
                (current_state["objects"]["grass"]["growth_stage"] == 1).sum().item()
            )

            # Store stats
            stats["episode"].append(episode)
            stats["step"].append(step)
            stats["predators_alive"].append(pred_alive)
            stats["prey_alive"].append(prey_alive)
            stats["grass_grown"].append(grass_grown)
            stats["time"].append(time.time() - start_time)

            # Visualize if enabled
            if visualize and visual:
                visual.capture(step, current_state)

        # Create animation if visualization is enabled
        if visualize and visual:
            visual.compile(episode)

    print(":: simulation completed")

    # Plot final statistics
    plot_statistics(stats, output_dir=os.path.join(current_dir, args.output))

    return runner, stats


def plot_statistics(stats, output_dir="results"):
    """Plot and save statistics from the simulation run."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot population counts
    episodes = np.array(stats["episode"])
    steps = np.array(stats["step"])
    step_numbers = episodes * max(steps) + steps

    # Population over time
    axs[0, 0].plot(step_numbers, stats["predators_alive"], "r-", label="Predators")
    axs[0, 0].plot(step_numbers, stats["prey_alive"], "b-", label="Prey")
    axs[0, 0].set_title("Population Over Time")
    axs[0, 0].set_xlabel("Simulation Step")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Grass grown over time
    axs[0, 1].plot(step_numbers, stats["grass_grown"], "g-", label="Grown Grass")
    axs[0, 1].set_title("Grass Growth Over Time")
    axs[0, 1].set_xlabel("Simulation Step")
    axs[0, 1].set_ylabel("Count")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Step execution time
    axs[1, 0].plot(step_numbers, stats["time"], "k-")
    axs[1, 0].set_title("Step Execution Time")
    axs[1, 0].set_xlabel("Simulation Step")
    axs[1, 0].set_ylabel("Time (seconds)")
    axs[1, 0].grid(True)

    # Population ratio
    prey_pred_ratio = np.array(stats["prey_alive"]) / np.maximum(
        np.array(stats["predators_alive"]), 1
    )
    axs[1, 1].plot(step_numbers, prey_pred_ratio, "purple")
    axs[1, 1].set_title("Prey/Predator Ratio")
    axs[1, 1].set_xlabel("Simulation Step")
    axs[1, 1].set_ylabel("Ratio")
    axs[1, 1].grid(True)

    # Add overall title
    fig.suptitle(
        "Predator-Prey Simulation Statistics (Vectorized Implementation)", fontsize=16
    )

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "statistics_vectorized.png"))
    plt.close()

    # Save raw data
    import json

    with open(os.path.join(output_dir, "statistics_vectorized.json"), "w") as f:
        json.dump(stats, f)

    print(f"Statistics saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    runner, stats = run_simulation(args.config, args)
