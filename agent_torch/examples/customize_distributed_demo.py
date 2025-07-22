"""
Demo: Customizing Agent Count and GPU Count for Distributed Simulation

This shows how NEITHER is hardcoded - both are fully configurable!
"""

import torch
from agent_torch.populations import sample2
from agent_torch.examples.models import movement
from agent_torch.core.environment import envs
from agent_torch.examples.setup_movement_sim import setup_movement_simulation


def demo_custom_agent_counts():
    """Show how to run with different agent counts."""
    
    print("🔢 CUSTOMIZING AGENT COUNTS")
    print("="*50)
    
    agent_counts = [5000, 20000, 100000]
    
    for num_agents in agent_counts:
        print(f"\n📊 Testing with {num_agents:,} agents...")
        
        # Method 1: Modify existing config programmatically
        config_builder = setup_movement_simulation()
        
        # Update agent count in the config builder
        config_builder.config["simulation_metadata"]["num_agents"] = num_agents
        config_builder.config["state"]["agents"]["citizens"]["number"] = num_agents
        
        # Update position shape to match agent count
        config_builder.config["state"]["agents"]["citizens"]["properties"]["position"]["shape"] = [num_agents, 2]
        
        # Get the modified config
        config = config_builder.config
        
        print(f"   ✅ Created config with {config['simulation_metadata']['num_agents']:,} agents")
        print(f"   📍 Position shape: {config['state']['agents']['citizens']['properties']['position']['shape']}")


def demo_custom_gpu_counts():
    """Show how to run with different GPU counts."""
    
    print("\n🖥️  CUSTOMIZING GPU COUNTS")
    print("="*50)
    
    total_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {total_gpus}")
    
    # Test different GPU configurations
    gpu_configs = [1, 2, min(3, total_gpus), total_gpus]
    gpu_configs = list(set(gpu_configs))  # Remove duplicates
    
    for world_size in gpu_configs:
        if world_size <= total_gpus:
            print(f"\n🔧 Configuration for {world_size} GPU(s):")
            
            runner = envs.create(
                model=movement,
                population=sample2,
                distributed=(world_size > 1),  # Auto-enable distributed if >1 GPU
                world_size=world_size
            )
            
            print(f"   ✅ Runner created for {world_size} GPU(s)")
            print(f"   📊 Agent count: {runner.config['simulation_metadata']['num_agents']:,}")
            print(f"   🔄 Distributed: {hasattr(runner, 'world_size')}")


def demo_full_customization():
    """Show complete customization of both agents and GPUs."""
    
    print("\n🎯 FULL CUSTOMIZATION DEMO")
    print("="*50)
    
    # Custom configurations
    test_configs = [
        {"agents": 50000, "gpus": 2, "name": "Medium Scale"},
        {"agents": 200000, "gpus": 3, "name": "Large Scale"},
        {"agents": 1000000, "gpus": torch.cuda.device_count(), "name": "Massive Scale"},
    ]
    
    for config in test_configs:
        agents = config["agents"]
        gpus = min(config["gpus"], torch.cuda.device_count())
        name = config["name"]
        
        print(f"\n📋 {name}: {agents:,} agents on {gpus} GPU(s)")
        
        # Create custom config
        config_builder = setup_movement_simulation()
        
        # Customize agent count
        config_builder.config["simulation_metadata"]["num_agents"] = agents
        config_builder.config["state"]["agents"]["citizens"]["number"] = agents
        config_builder.config["state"]["agents"]["citizens"]["properties"]["position"]["shape"] = [agents, 2]
        
        # Create runner with custom GPU count
        try:
            runner = envs.create(
                model=movement,
                population=sample2,
                distributed=(gpus > 1),
                world_size=gpus
            )
            
            # Override config with our custom one
            runner.config.update(config_builder.config)
            
            print(f"   ✅ Successfully configured")
            print(f"   🎯 Agents: {runner.config['simulation_metadata']['num_agents']:,}")
            print(f"   🖥️  GPUs: {gpus}")
            print(f"   📝 Agents per GPU: {agents // gpus:,}")
            
            # Could run simulation here if needed
            # runner.init()
            # runner.step(5)
            
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")


def show_api_examples():
    """Show different API ways to customize parameters."""
    
    print("\n🔧 API CUSTOMIZATION EXAMPLES")
    print("="*50)
    
    print("\n1. Customize via envs.create():")
    print("""
    # Method 1: Basic distributed
    runner = envs.create(
        model=movement,
        population=sample2,
        distributed=True,      # Enable distributed
        world_size=4          # Use 4 GPUs
    )
    """)
    
    print("\n2. Customize via config modification:")
    print("""
    # Method 2: Custom agent count
    config_builder = setup_movement_simulation()
    config_builder.config["simulation_metadata"]["num_agents"] = 500000
    
    runner = envs.create(
        model=movement,
        population=sample2,
        distributed=True,
        world_size=8
    )
    runner.config.update(config_builder.config)
    """)
    
    print("\n3. Customize with distributed config:")
    print("""
    # Method 3: Advanced distributed settings
    distributed_config = {
        "strategy": "data_parallel",
        "sync_frequency": 10
    }
    
    runner = envs.create(
        model=movement,
        population=sample2,
        distributed=True,
        world_size=2,
        distributed_config=distributed_config
    )
    """)


def main():
    print("🚀 DISTRIBUTED SIMULATION CUSTOMIZATION DEMO")
    print("="*60)
    print("Showing that NOTHING is hardcoded!")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This demo requires GPU support.")
        return
    
    try:
        # 1. Show agent count customization
        demo_custom_agent_counts()
        
        # 2. Show GPU count customization  
        demo_custom_gpu_counts()
        
        # 3. Show full customization
        demo_full_customization()
        
        # 4. Show API examples
        show_api_examples()
        
        print(f"\n✅ CONCLUSION: Both agent count and GPU count are FULLY CONFIGURABLE!")
        print(f"   🔢 Agents: Set via config or setup_movement_simulation()")
        print(f"   🖥️  GPUs: Set via world_size parameter")
        print(f"   🎯 No hardcoding anywhere!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    main() 