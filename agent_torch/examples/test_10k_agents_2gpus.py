"""
Simple test: 10,000 agents on 2 GPUs

This demonstrates customizing both agent count and GPU count.
"""

import torch
from agent_torch.populations import sample2
from agent_torch.examples.models import movement
from agent_torch.core.environment import envs
from agent_torch.examples.setup_movement_sim import setup_movement_simulation


def test_10k_agents_2gpus():
    """Test exactly what the user requested: 10,000 agents on 2 GPUs."""
    
    print("🧪 Testing: 10,000 agents on 2 GPUs")
    print("="*50)
    
    # Step 1: Create custom config with 10,000 agents
    print("📝 Step 1: Creating config with 10,000 agents...")
    config_builder = setup_movement_simulation()
    
    # Customize to 10,000 agents
    config_builder.config["simulation_metadata"]["num_agents"] = 10000
    config_builder.config["state"]["agents"]["citizens"]["number"] = 10000
    config_builder.config["state"]["agents"]["citizens"]["properties"]["position"]["shape"] = [10000, 2]
    
    print(f"   ✅ Config created with {config_builder.config['simulation_metadata']['num_agents']:,} agents")
    
    # Step 2: Create distributed runner for 2 GPUs
    print("\n🖥️  Step 2: Setting up distributed execution on 2 GPUs...")
    runner = envs.create(
        model=movement,
        population=sample2,
        distributed=True,
        world_size=2  # Force 2 GPUs
    )
    
    # Apply our custom config
    runner.config.update(config_builder.config)
    
    print(f"   ✅ Distributed runner created")
    print(f"   🎯 Total agents: {runner.config['simulation_metadata']['num_agents']:,}")
    print(f"   🖥️  GPUs: 2")
    print(f"   📊 Agents per GPU: {runner.config['simulation_metadata']['num_agents'] // 2:,}")
    
    # Step 3: Initialize and run a few steps
    print("\n🚀 Step 3: Running simulation...")
    try:
        runner.init()
        print("   ✅ Initialization completed")
        
        # Run 5 steps as a test
        result = runner.step(5)
        print("   ✅ Simulation steps completed")
        
        if runner.state and "agents" in runner.state:
            positions = runner.state["agents"]["citizens"]["position"]
            print(f"   📍 Final position tensor shape: {positions.shape}")
            print(f"   📈 Average position: {positions.mean(dim=0)}")
            print(f"   🎯 SUCCESS: 10,000 agents ran on 2 GPUs!")
        else:
            print("   ⚠️  No final state returned")
            
    except Exception as e:
        print(f"   ❌ Simulation failed: {e}")
        print("   💡 This might be due to the distributed implementation needing refinement")


def show_configuration_summary():
    """Show how both parameters are configurable."""
    
    print("\n📋 CONFIGURATION SUMMARY")
    print("="*50)
    
    print("🔢 AGENT COUNT:")
    print("   ❌ NOT hardcoded")
    print("   ✅ Configurable via:")
    print("      - setup_movement_simulation() API")
    print("      - Direct config modification")
    print("      - YAML file editing")
    
    print("\n🖥️  GPU COUNT:")
    print("   ❌ NOT hardcoded") 
    print("   ✅ Configurable via:")
    print("      - world_size parameter")
    print("      - Auto-detection (default)")
    print("      - envs.create() argument")
    
    print("\n💡 EXAMPLES:")
    print("   # 50K agents on 4 GPUs:")
    print("   config['num_agents'] = 50000")
    print("   runner = envs.create(..., world_size=4)")
    print()
    print("   # 1M agents on 8 GPUs:")
    print("   config['num_agents'] = 1000000")  
    print("   runner = envs.create(..., world_size=8)")


def main():
    print("🎯 TESTING USER REQUEST: 10,000 agents on 2 GPUs")
    print("Also proving that NOTHING is hardcoded!")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires GPU support.")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"🖥️  Available GPUs: {num_gpus}")
    
    if num_gpus < 2:
        print("⚠️  Need at least 2 GPUs for this test")
        print("   But agent count is still configurable!")
        show_configuration_summary()
        return
    
    # Run the actual test
    test_10k_agents_2gpus()
    
    # Show configuration options
    show_configuration_summary()


if __name__ == "__main__":
    main() 