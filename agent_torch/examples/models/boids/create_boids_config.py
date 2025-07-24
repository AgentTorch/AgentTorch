#!/usr/bin/env python3
"""
Boids Model Configuration Generator

This script generates the boids model configuration using AgentTorch's Config API.
It creates both the YAML configuration file and substep implementation templates.

Usage:
    python create_boids_config.py

Generated files:
    - yamls/config.yaml: Model configuration
    - substeps/*.py: Substep implementation templates (if they don't exist)

The configuration includes:
    - 100 boids with position/velocity properties
    - Environment bounds and flocking parameters
    - Two main substeps: FlockingBehavior and MovementUpdate
    - Learnable parameters for customization
"""

from agent_torch.config import (
    ConfigBuilder,
    StateBuilder,
    AgentBuilder,
    PropertyBuilder,
    EnvironmentBuilder,
    SubstepBuilderWithImpl,
    PolicyBuilder,
    TransitionBuilder
)

def create_boids_config():
    """Create the boids flocking simulation configuration."""
    print("Creating boids model configuration...")
    
    # Initialize config builder
    config = ConfigBuilder()
    
    # 1. Set simulation metadata
    metadata = {
        "num_agents": 100,
        "num_episodes": 1,
        "num_steps_per_episode": 500,
        "num_substeps_per_step": 2,  # Two substeps: flocking + movement
        "device": "cpu",
        "calibration": False
    }
    config.set_metadata(metadata)
    
    # 2. Build state configuration
    state_builder = StateBuilder()
    
    # Add boid agents with simple initial values (will be randomized by initialization substep)
    agent_builder = AgentBuilder("boids", metadata["num_agents"])
    
    # Position property (will be randomized)
    position = PropertyBuilder("position")\
        .set_dtype("float")\
        .set_shape([metadata["num_agents"], 2])\
        .set_value([400.0, 300.0])  # Default center position
    agent_builder.add_property(position)
    
    # Velocity property (will be randomized)
    velocity = PropertyBuilder("velocity")\
        .set_dtype("float")\
        .set_shape([metadata["num_agents"], 2])\
        .set_value([1.0, 0.0])  # Default rightward velocity
    agent_builder.add_property(velocity)
    
    state_builder.add_agent("boids", agent_builder)
    
    # Add environment variables
    env_builder = EnvironmentBuilder()
    
    # World boundaries
    bounds = PropertyBuilder("bounds")\
        .set_dtype("float")\
        .set_shape([2])\
        .set_value([800.0, 600.0])
    env_builder.add_variable(bounds)
    
    # Boids parameters
    perception_radius = PropertyBuilder("perception_radius")\
        .set_dtype("float")\
        .set_value(50.0)
    env_builder.add_variable(perception_radius)
    
    separation_distance = PropertyBuilder("separation_distance")\
        .set_dtype("float")\
        .set_value(25.0)
    env_builder.add_variable(separation_distance)
    
    max_speed = PropertyBuilder("max_speed")\
        .set_dtype("float")\
        .set_value(4.0)
    env_builder.add_variable(max_speed)
    
    max_force = PropertyBuilder("max_force")\
        .set_dtype("float")\
        .set_value(0.1)
    env_builder.add_variable(max_force)
    
    state_builder.set_environment(env_builder)
    config.set_state(state_builder.to_dict())
    
    # 3. Create Substep 0: Flocking Behavior
    flocking_substep = SubstepBuilderWithImpl(
        name="FlockingBehavior",
        description="Calculate flocking forces (separation, alignment, cohesion)",
        output_dir="substeps"  # Use relative path
    )
    flocking_substep.add_active_agent("boids")
    flocking_substep.config["observation"] = {"boids": None}
    
    # Flocking policy - calculate steering forces
    flocking_policy = PolicyBuilder()
    
    # Learnable initialization parameters  
    speed_range = PropertyBuilder.create_argument(
        name="Speed range for random velocities",
        value=[0.5, 1.5],
        shape=[2],
        learnable=True
    ).config
    
    position_margin = PropertyBuilder.create_argument(
        name="Margin from boundaries for positions", 
        value=50.0,
        learnable=True
    ).config
    
    # Learnable weights for the three flocking forces
    separation_weight = PropertyBuilder.create_argument(
        name="Separation weight",
        value=1.5,
        learnable=True
    ).config
    
    alignment_weight = PropertyBuilder.create_argument(
        name="Alignment weight", 
        value=1.0,
        learnable=True
    ).config
    
    cohesion_weight = PropertyBuilder.create_argument(
        name="Cohesion weight",
        value=1.0,
        learnable=True
    ).config
    
    flocking_policy.add_policy(
        "calculate_flocking_forces",
        "CalculateFlockingForces",
        {
            "position": "agents/boids/position",
            "velocity": "agents/boids/velocity",
            "perception_radius": "environment/perception_radius",
            "separation_distance": "environment/separation_distance"
        },
        ["steering_force"],
        {
            "speed_range": speed_range,
            "position_margin": position_margin,
            "separation_weight": separation_weight,
            "alignment_weight": alignment_weight, 
            "cohesion_weight": cohesion_weight
        }
    )
    flocking_substep.set_policy("boids", flocking_policy)
    
    # Flocking transition - update velocity based on steering force
    flocking_transition = TransitionBuilder()
    
    max_force_param = PropertyBuilder.create_argument(
        name="Maximum steering force",
        value=0.1,
        learnable=True
    ).config
    
    flocking_transition.add_transition(
        "update_velocity",
        "UpdateVelocity", 
        {
            "velocity": "agents/boids/velocity",
            "max_force": "environment/max_force"
        },
        ["velocity"],
        {"max_force": max_force_param}
    )
    flocking_substep.set_transition(flocking_transition)
    
    # 4. Create Substep 1: Movement Update
    movement_substep = SubstepBuilderWithImpl(
        name="MovementUpdate",
        description="Update positions and handle boundaries",
        output_dir="substeps"  # Use relative path
    )
    movement_substep.add_active_agent("boids")
    movement_substep.config["observation"] = {"boids": None}
    
    # Movement policy - apply speed limits
    movement_policy = PolicyBuilder()
    
    max_speed_param = PropertyBuilder.create_argument(
        name="Maximum speed",
        value=4.0,
        learnable=True
    ).config
    
    movement_policy.add_policy(
        "limit_speed",
        "LimitSpeed",
        {
            "velocity": "agents/boids/velocity",
            "max_speed": "environment/max_speed"
        },
        ["limited_velocity"],
        {"max_speed": max_speed_param}
    )
    movement_substep.set_policy("boids", movement_policy)
    
    # Movement transition - update position and handle boundaries  
    movement_transition = TransitionBuilder()
    
    bounds_param = PropertyBuilder.create_argument(
        name="World boundaries",
        value=[800.0, 600.0],
        shape=[2],
        learnable=False
    ).config
    
    movement_transition.add_transition(
        "update_position",
        "UpdatePosition",
        {
            "position": "agents/boids/position",
            "bounds": "environment/bounds"
        },
        ["position"],
        {"bounds": bounds_param}
    )
    movement_substep.set_transition(movement_transition)
    
    # 5. Add substeps to config and generate implementations
    config.add_substep("0", flocking_substep)
    config.add_substep("1", movement_substep)
    
    # Generate the substep implementation files
    print("Generating flocking behavior substep implementations...")
    flocking_files = flocking_substep.generate_implementations()
    print(f"Generated files: {flocking_files}")
    
    print("Generating movement update substep implementations...")
    movement_files = movement_substep.generate_implementations()
    print(f"Generated files: {movement_files}")
    
    # Save configuration
    config_path = "agent_torch/examples/models/boids/yamls/config.yaml"
    config.save_yaml(config_path)
    print(f"Configuration saved to: {config_path}")
    
    return config

if __name__ == "__main__":
    config = create_boids_config()
    print("Boids model configuration and substep templates created successfully!") 