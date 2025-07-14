#!/usr/bin/env python
"""
Benchmark comparing standard vs. vectorized implementations of the predator-prey model.

This script runs benchmark tests comparing the performance of standard vs. 
vectorized implementations of the predator-prey model. It measures initialization
and simulation times for both implementations and reports speedups.
"""
import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from agent_torch.core import Registry, Runner, VectorizedRunner
from agent_torch.core.helpers import read_config, read_from_file, grid_network

# Import all substeps - standard and vectorized
from agent_torch.models.predator_prey.substeps import *
from agent_torch.models.predator_prey.vmap_substeps import *
from agent_torch.models.predator_prey.helpers.map import map_network
from agent_torch.models.predator_prey.helpers.random import random_float, random_int


def parse_args():
    """Parse command-line arguments for the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark standard vs. vectorized AgentTorch implementations"
    )
    parser.add_argument(
        "--std-config",
        type=str,
        default="config.yaml",
        help="Path to standard config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--vmap-config",
        type=str,
        default="config_vmap.yaml",
        help="Path to vectorized config YAML (default: config_vmap.yaml)",
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of simulation steps (default: 10)"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of benchmark runs (default: 3)"
    )
    parser.add_argument(
        "--predators", type=int, default=40, help="Number of predators (default: 40)"
    )
    parser.add_argument(
        "--prey", type=int, default=80, help="Number of prey (default: 80)"
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Run scale test with different population sizes",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run simulations on (default: cpu)",
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

    if file_path.endswith("csv"):
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


def run_benchmark(runner_class, config, registry, num_runs=3, num_steps=10):
    """
    Run a benchmark for the given runner type.

    Args:
        runner_class: Runner class to benchmark
        config: Configuration dictionary
        registry: Registry with registered functions
        num_runs: Number of benchmark runs to perform
        num_steps: Number of simulation steps per run

    Returns:
        dict: Dictionary with benchmark results
    """
    runner_name = runner_class.__name__
    print(f"\n=== {runner_name} Benchmark ===")

    # Track timing results
    init_times = []
    run_times = []
    mem_usage = []
    step_times = []

    # Run multiple simulations to get statistics
    for run in range(num_runs):
        # Track memory usage
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Measure initialization time
        init_start = time.time()
        runner = runner_class(config, registry)
        runner.init()
        init_end = time.time()
        init_time = init_end - init_start
        init_times.append(init_time)

        # Measure simulation time with per-step timing
        run_start = time.time()
        run_step_times = []

        for _ in range(num_steps):
            step_start = time.time()
            runner.step(1)
            step_end = time.time()
            run_step_times.append(step_end - step_start)

        run_end = time.time()
        run_time = run_end - run_start
        run_times.append(run_time)
        step_times.append(run_step_times)

        # Track memory usage
        end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        mem_used = end_mem - start_mem if end_mem > start_mem else 0
        mem_usage.append(mem_used)

        print(
            f"Run {run+1}: initialization {init_time:.4f}s, simulation {run_time:.4f}s"
        )

    # Calculate statistics
    results = {
        "runner": runner_name,
        "initialization": {
            "mean": np.mean(init_times),
            "std": np.std(init_times),
            "min": np.min(init_times),
            "max": np.max(init_times),
            "all_runs": init_times,
        },
        "simulation": {
            "mean": np.mean(run_times),
            "std": np.std(run_times),
            "min": np.min(run_times),
            "max": np.max(run_times),
            "all_runs": run_times,
        },
        "step_times": {
            "mean": np.mean([np.mean(steps) for steps in step_times]),
            "std": np.std([np.mean(steps) for steps in step_times]),
            "all_steps": step_times,
        },
        "memory": {"mean": np.mean(mem_usage), "std": np.std(mem_usage)},
    }

    return results


def run_scale_test(config_std, config_vmap, registry, args):
    """
    Run scale tests with different population sizes.

    Args:
        config_std: Standard configuration
        config_vmap: Vectorized configuration
        registry: Registry with registered functions
        args: Command-line arguments

    Returns:
        dict: Dictionary with scale test results
    """
    # Define different scales to test
    scales = [
        {"predators": 10, "prey": 20},
        {"predators": 20, "prey": 40},
        {"predators": 40, "prey": 80},
        {"predators": 80, "prey": 160},
        {"predators": 160, "prey": 320},
    ]

    results = {"scales": scales, "standard": [], "vectorized": []}

    for scale in scales:
        print(
            f"\n=== Scale Test: {scale['predators']} predators, {scale['prey']} prey ==="
        )

        # Update configurations with current scale
        config_std_copy = config_std.copy()
        config_vmap_copy = config_vmap.copy()

        for config in [config_std_copy, config_vmap_copy]:
            config["simulation_metadata"]["num_predators"] = scale["predators"]
            config["simulation_metadata"]["num_prey"] = scale["prey"]

        # Run benchmarks
        std_result = run_benchmark(
            Runner, config_std_copy, registry, num_runs=1, num_steps=args.steps
        )
        vmap_result = run_benchmark(
            VectorizedRunner,
            config_vmap_copy,
            registry,
            num_runs=1,
            num_steps=args.steps,
        )

        # Store results
        results["standard"].append(
            {
                "init_time": std_result["initialization"]["mean"],
                "sim_time": std_result["simulation"]["mean"],
                "step_time": std_result["step_times"]["mean"],
            }
        )

        results["vectorized"].append(
            {
                "init_time": vmap_result["initialization"]["mean"],
                "sim_time": vmap_result["simulation"]["mean"],
                "step_time": vmap_result["step_times"]["mean"],
            }
        )

    return results


def plot_benchmark_results(results, output_dir="benchmark_results"):
    """Plot and save benchmark results."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract data
    std_init = results["standard"]["initialization"]["mean"]
    vmap_init = results["vectorized"]["initialization"]["mean"]

    std_sim = results["standard"]["simulation"]["mean"]
    vmap_sim = results["vectorized"]["simulation"]["mean"]

    std_step = results["standard"]["step_times"]["mean"]
    vmap_step = results["vectorized"]["step_times"]["mean"]

    # Calculate speedups
    init_speedup = std_init / vmap_init if vmap_init > 0 else 0
    sim_speedup = std_sim / vmap_sim if vmap_sim > 0 else 0
    step_speedup = std_step / vmap_step if vmap_step > 0 else 0

    # Create figure with bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up data
    labels = ["Initialization", "Simulation", "Average Step"]
    std_times = [std_init, std_sim, std_step]
    vmap_times = [vmap_init, vmap_sim, vmap_step]

    # Set up bar chart
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, std_times, width, label="Standard")
    ax.bar(x + width / 2, vmap_times, width, label="Vectorized")

    # Add speedup annotations
    for i, (std, vmap) in enumerate(zip(std_times, vmap_times)):
        speedup = std / vmap if vmap > 0 else 0
        ax.annotate(
            f"{speedup:.2f}x",
            xy=(x[i], min(std, vmap) / 2),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Customize chart
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Performance Comparison: Standard vs. Vectorized Implementation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_comparison.png"))
    plt.close()

    return init_speedup, sim_speedup, step_speedup


def plot_scale_test_results(scale_results, output_dir="benchmark_results"):
    """Plot and save scale test results."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract data
    scales = [
        f"{s['predators']} pred\n{s['prey']} prey" for s in scale_results["scales"]
    ]

    std_init_times = [r["init_time"] for r in scale_results["standard"]]
    vmap_init_times = [r["init_time"] for r in scale_results["vectorized"]]

    std_sim_times = [r["sim_time"] for r in scale_results["standard"]]
    vmap_sim_times = [r["sim_time"] for r in scale_results["vectorized"]]

    std_step_times = [r["step_time"] for r in scale_results["standard"]]
    vmap_step_times = [r["step_time"] for r in scale_results["vectorized"]]

    # Calculate speedups
    init_speedups = [
        s / v if v > 0 else 0 for s, v in zip(std_init_times, vmap_init_times)
    ]
    sim_speedups = [
        s / v if v > 0 else 0 for s, v in zip(std_sim_times, vmap_sim_times)
    ]
    step_speedups = [
        s / v if v > 0 else 0 for s, v in zip(std_step_times, vmap_step_times)
    ]

    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot initialization times
    axs[0, 0].plot(scales, std_init_times, "o-", label="Standard")
    axs[0, 0].plot(scales, vmap_init_times, "o-", label="Vectorized")
    axs[0, 0].set_title("Initialization Time")
    axs[0, 0].set_ylabel("Time (seconds)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot simulation times
    axs[0, 1].plot(scales, std_sim_times, "o-", label="Standard")
    axs[0, 1].plot(scales, vmap_sim_times, "o-", label="Vectorized")
    axs[0, 1].set_title("Total Simulation Time")
    axs[0, 1].set_ylabel("Time (seconds)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot average step times
    axs[1, 0].plot(scales, std_step_times, "o-", label="Standard")
    axs[1, 0].plot(scales, vmap_step_times, "o-", label="Vectorized")
    axs[1, 0].set_title("Average Step Time")
    axs[1, 0].set_ylabel("Time (seconds)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot speedups
    axs[1, 1].plot(scales, init_speedups, "o-", label="Initialization")
    axs[1, 1].plot(scales, sim_speedups, "o-", label="Simulation")
    axs[1, 1].plot(scales, step_speedups, "o-", label="Step")
    axs[1, 1].set_title("Speedup Factors (Standard / Vectorized)")
    axs[1, 1].set_ylabel("Speedup (x times)")
    axs[1, 1].axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Add overall title
    fig.suptitle(
        "Performance Scaling: Standard vs. Vectorized Implementation", fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "scale_test_results.png"))
    plt.close()


def main():
    """Run the benchmark comparing standard vs. vectorized implementations."""
    args = parse_args()

    # Enable PyTorch optimizations
    torch.set_float32_matmul_precision("high")

    # Setup paths to config files
    std_config_path = os.path.join(current_dir, args.std_config)
    vmap_config_path = os.path.join(current_dir, args.vmap_config)

    # Read configurations
    print(f"Reading standard config from: {std_config_path}")
    std_config = read_config(std_config_path, register_resolvers=True)

    print(f"Reading vectorized config from: {vmap_config_path}")
    vmap_config = read_config(vmap_config_path, register_resolvers=False)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(current_dir, args.out)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Override population sizes if specified
    for config in [std_config, vmap_config]:
        if args.predators:
            config["simulation_metadata"]["num_predators"] = args.predators
        if args.prey:
            config["simulation_metadata"]["num_prey"] = args.prey
        config["simulation_metadata"]["visualize"] = False
        config["simulation_metadata"]["device"] = args.device

    # Setup registry
    registry = setup_registry()

    # If scale test is requested, run scale tests
    if args.scale:
        print("\n=== Running Scale Tests ===")
        scale_results = run_scale_test(std_config, vmap_config, registry, args)
        plot_scale_test_results(scale_results, output_dir)

        # Save scale test results
        with open(os.path.join(output_dir, "scale_test_results.json"), "w") as f:
            json.dump(scale_results, f, indent=2)

        print(f"\nScale test results saved to {output_dir}")

    # Otherwise, run standard benchmarks
    else:
        # Run standard implementation benchmark
        standard_results = run_benchmark(
            Runner, std_config, registry, args.runs, args.steps
        )

        # Run vectorized implementation benchmark
        vectorized_results = run_benchmark(
            VectorizedRunner, vmap_config, registry, args.runs, args.steps
        )

        # Prepare complete results
        results = {
            "standard": standard_results,
            "vectorized": vectorized_results,
            "config": {
                "steps": args.steps,
                "runs": args.runs,
                "num_predators": std_config["simulation_metadata"]["num_predators"],
                "num_prey": std_config["simulation_metadata"]["num_prey"],
                "device": args.device,
            },
        }

        # Plot results
        init_speedup, sim_speedup, step_speedup = plot_benchmark_results(
            results, output_dir
        )

        # Calculate speedups and add to results
        results["speedup"] = {
            "initialization": init_speedup,
            "simulation": sim_speedup,
            "step": step_speedup,
        }

        # Save results
        with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        print("\n=== Benchmark Summary ===")
        print(f"Initialization:")
        print(
            f"  Standard:   {standard_results['initialization']['mean']:.4f} ± {standard_results['initialization']['std']:.4f} seconds"
        )
        print(
            f"  Vectorized: {vectorized_results['initialization']['mean']:.4f} ± {vectorized_results['initialization']['std']:.4f} seconds"
        )
        print(f"  Speedup:    {init_speedup:.2f}x")

        print(f"\nSimulation:")
        print(
            f"  Standard:   {standard_results['simulation']['mean']:.4f} ± {standard_results['simulation']['std']:.4f} seconds"
        )
        print(
            f"  Vectorized: {vectorized_results['simulation']['mean']:.4f} ± {vectorized_results['simulation']['std']:.4f} seconds"
        )
        print(f"  Speedup:    {sim_speedup:.2f}x")

        print(f"\nAverage Step:")
        print(
            f"  Standard:   {standard_results['step_times']['mean']:.4f} ± {standard_results['step_times']['std']:.4f} seconds"
        )
        print(
            f"  Vectorized: {vectorized_results['step_times']['mean']:.4f} ± {vectorized_results['step_times']['std']:.4f} seconds"
        )
        print(f"  Speedup:    {step_speedup:.2f}x")

        print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
