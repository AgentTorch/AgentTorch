from pathlib import Path
from agent_torch.config import (
    ConfigBuilder,
    StateBuilder,
    AgentBuilder,
    PropertyBuilder,
    EnvironmentBuilder,
    SubstepBuilder,
    PolicyBuilder,
    TransitionBuilder,
)


def setup_movement_simulation():
    """Setup a complete movement simulation structure."""

    # Create config builder
    config = ConfigBuilder()

    # Set simulation metadata - these are all required arguments
    metadata = {
        "num_agents": 1000,
        "num_episodes": 10,
        "num_steps_per_episode": 20,
        "num_substeps_per_step": 1,
        "device": "cpu",
        "calibration": False,
    }
    config.set_metadata(metadata)

    # Build state
    state_builder = StateBuilder()

    # Add agent
    agent_builder = AgentBuilder("citizens", metadata["num_agents"])

    # Add position property
    position = (
        PropertyBuilder("position")
        .set_dtype("float")
        .set_shape([metadata["num_agents"], 2])
        .set_value([0.0, 0.0])
    )

    agent_builder.add_property(position)
    state_builder.add_agent("citizens", agent_builder)

    # Add environment bounds
    env_builder = EnvironmentBuilder()
    bounds = (
        PropertyBuilder("bounds")
        .set_dtype("float")
        .set_shape([2])
        .set_value([100.0, 100.0])
    )
    env_builder.add_variable(bounds)
    state_builder.set_environment(env_builder)

    # Set state in config
    config.set_state(state_builder.to_dict())

    # Create movement substep
    movement = SubstepBuilder("Movement", "Agent movement simulation")
    movement.add_active_agent("citizens")

    # Set observation structure
    movement.config["observation"] = {"citizens": None}

    # Add movement policy
    policy = PolicyBuilder()
    step_size = PropertyBuilder.create_argument(
        name="Step size parameter", value=1.0, learnable=True
    ).config  # Use .config instead of .to_dict()

    policy.add_policy(
        "move",
        "RandomMove",
        {"position": "agents/citizens/position"},
        ["direction"],
        {"step_size": step_size},
    )
    movement.set_policy("citizens", policy)

    # Add movement transition
    transition = TransitionBuilder()
    bounds_param = PropertyBuilder.create_argument(
        name="Environment bounds", value=[100.0, 100.0], shape=[2], learnable=True
    ).config  # Use .config instead of .to_dict()

    transition.add_transition(
        "update_position",
        "UpdatePosition",
        {"position": "agents/citizens/position"},
        ["position"],
        {"bounds": bounds_param},
    )
    movement.set_transition(transition)

    # Add substep to config
    config.add_substep("0", movement)

    # Save the config
    examples_dir = Path(__file__).parent
    model_dir = examples_dir / "models" / "movement"
    config_path = model_dir / "yamls" / "config.yaml"
    config.save_yaml(str(config_path))
    print(f"\nGenerated config file: {config_path}")

    return config


if __name__ == "__main__":
    setup_movement_simulation()
