"""
Distributed version of run_movement_sim.py

This shows how to convert existing AgentTorch simulations to run on multiple GPUs
with minimal code changes.

Compare this to run_movement_sim.py - only 3 lines changed!
"""

print("--- RUNNING DISTRIBUTED MOVEMENT SIMULATION ---")
import warnings

warnings.filterwarnings("ignore")

from agent_torch.populations import sample2
from agent_torch.examples.models import movement
from agent_torch.core.environment import envs


def run_movement_simulation_distributed(world_size=2):
    """Run the movement simulation using distributed execution."""
    print(f"\n=== Running Distributed Movement Simulation on {world_size} GPUs ===")

    # Create the runner using envs.create - ONLY CHANGE: added distributed=True
    print("\nCreating distributed simulation runner...")
    runner = envs.create(
        model=movement, 
        population=sample2, 
        distributed=True,           # <-- ONLY CHANGE 1: Enable distributed
        world_size=world_size       # <-- ONLY CHANGE 2: Specify GPU count
    )

    # Get simulation parameters from config (SAME AS BEFORE)
    sim_steps = runner.config["simulation_metadata"]["num_steps_per_episode"]
    num_episodes = runner.config["simulation_metadata"]["num_episodes"]

    print(f"\nSimulation parameters:")
    print(f"- Steps per episode: {sim_steps}")
    print(f"- Number of episodes: {num_episodes}")
    print(f"- Number of agents: {runner.config['simulation_metadata']['num_agents']}")
    print(f"- GPUs: {world_size}")

    # Initialize runner (SAME AS BEFORE)
    runner.init()

    # Run all episodes (SAME AS BEFORE)
    print("\nRunning distributed simulation episodes...")
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")

        # Reset state at the start of each episode (SAME AS BEFORE)
        if episode > 0:
            runner.reset()

        # Run one episode (SAME AS BEFORE)
        runner.step(sim_steps)

        # Print statistics (SAME AS BEFORE)
        if runner.state and "agents" in runner.state:
            final_state = runner.state
            positions = final_state["agents"]["citizens"]["position"]
            print(f"- Average position: {positions.mean(dim=0)}")
            print(f"- Position tensor shape: {positions.shape}")

    print("\nDistributed simulation completed!")
    return runner


def compare_single_vs_distributed():
    """Compare single GPU vs distributed performance."""
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON: Single GPU vs Distributed")
    print("="*60)
    
    import time
    
    # Test 1: Single GPU (existing approach)
    print("\n1. Single GPU Simulation:")
    start_time = time.time()
    
    # Original code - exactly as before
    runner_single = envs.create(model=movement, population=sample2)
    runner_single.init()
    runner_single.step(20)  # 20 steps
    
    single_time = time.time() - start_time
    single_positions = runner_single.state["agents"]["citizens"]["position"]
    
    print(f"   ✅ Completed in {single_time:.2f} seconds")
    print(f"   📍 Final positions shape: {single_positions.shape}")
    print(f"   📊 Average position: {single_positions.mean(dim=0)}")
    
    # Test 2: Distributed (2 GPUs)
    print("\n2. Distributed Simulation (2 GPUs):")
    start_time = time.time()
    
    # New distributed code - minimal changes
    runner_distributed = envs.create(
        model=movement, 
        population=sample2, 
        distributed=True,
        world_size=2
    )
    runner_distributed.init()
    runner_distributed.step(20)  # 20 steps
    
    distributed_time = time.time() - start_time
    
    if runner_distributed.state and "agents" in runner_distributed.state:
        distributed_positions = runner_distributed.state["agents"]["citizens"]["position"]
        print(f"   ✅ Completed in {distributed_time:.2f} seconds")
        print(f"   📍 Final positions shape: {distributed_positions.shape}")
        print(f"   📊 Average position: {distributed_positions.mean(dim=0)}")
        
        # Performance metrics
        if single_time > 0:
            speedup = single_time / distributed_time
            print(f"   🚀 Speedup: {speedup:.2f}x")
    else:
        print("   ❌ Distributed simulation failed or returned no state")


def run_with_custom_config():
    """Show how to use distributed execution with custom configurations."""
    
    print("\n" + "="*50)
    print("CUSTOM DISTRIBUTED CONFIGURATION")
    print("="*50)
    
    # Custom distributed settings
    distributed_config = {
        "strategy": "data_parallel",
        "sync_frequency": 3,  # Sync every 3 steps
    }
    
    runner = envs.create(
        model=movement,
        population=sample2,
        distributed=True,
        world_size=2,
        distributed_config=distributed_config  # <-- CHANGE 3: Custom config
    )
    
    print(f"✅ Created runner with custom distributed config")
    print(f"   Strategy: {distributed_config['strategy']}")
    print(f"   Sync frequency: {distributed_config['sync_frequency']}")
    
    runner.init()
    runner.step(10)
    
    if runner.state and "agents" in runner.state:
        positions = runner.state["agents"]["citizens"]["position"]
        print(f"   📍 Final shape: {positions.shape}")
        print(f"   📊 Average position: {positions.mean(dim=0)}")


def main():
    import argparse
    import torch
    from agent_torch.examples.setup_movement_sim import setup_movement_simulation
    
    parser = argparse.ArgumentParser(description="Distributed Movement Simulation")
    parser.add_argument("--agents", type=int, default=1000, help="Number of agents")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps")
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This demo requires GPU support.")
        return
    
    num_gpus = torch.cuda.device_count()
    world_size = args.gpus if args.gpus else num_gpus
    world_size = min(world_size, num_gpus)
    
    print(f"🖥️  Available GPUs: {num_gpus}")
    print(f"🎯 Using: {args.agents:,} agents on {world_size} GPU(s)")
    
    # Create custom config with specified agent count
    config_builder = setup_movement_simulation()
    config_builder.config["simulation_metadata"]["num_agents"] = args.agents
    config_builder.config["state"]["agents"]["citizens"]["number"] = args.agents
    config_builder.config["state"]["agents"]["citizens"]["properties"]["position"]["shape"] = [args.agents, 2]
    
    try:
        # Create distributed runner
        runner = envs.create(
            model=movement,
            population=sample2,
            distributed=(world_size > 1),
            world_size=world_size
        )
        
        # Apply custom config
        runner.config.update(config_builder.config)
        
        print(f"✅ Runner created: {args.agents:,} agents, {world_size} GPUs")
        
        # Run simulation
        runner.init()
        runner.step(args.steps)
        
        if runner.state and "agents" in runner.state:
            positions = runner.state["agents"]["citizens"]["position"]
            print(f"📍 Final shape: {positions.shape}")
            print(f"📊 Average position: {positions.mean(dim=0)}")
            print("✅ Simulation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 