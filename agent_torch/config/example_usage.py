from state_builder import (
    StateBuilder,
    AgentBuilder,
    PropertyBuilder,
    EnvironmentBuilder,
    NetworkBuilder,
)
from substep_builder import (
    ConfigBuilder,
    SubstepBuilder,
    PolicyBuilder,
    TransitionBuilder,
)


def create_covid_config():
    # Create the main config builder
    config = ConfigBuilder()

    # Set simulation metadata
    metadata = {
        "ALIGN_LLM": True,
        "EXECUTION_MODE": "heuristic",
        "NUM_TRAIN_WEEKS": 2,
        "NUM_WEEKS": 3,
        "num_agents": 1000,
        "population_dir": "/path/to/population",
        # ... other metadata ...
    }
    config.set_metadata(metadata)

    # Build state configuration
    state_builder = StateBuilder()

    # Add citizen agent
    citizen_builder = AgentBuilder("citizens", metadata["num_agents"])

    # Add agent properties
    age_prop = (
        PropertyBuilder("age")
        .set_dtype("int")
        .set_shape([metadata["num_agents"], 1])
        .set_initialization(
            "load_population_attribute",
            {
                "file_path": {"value": "${simulation_metadata.age_group_file}"},
                "attribute": {"value": "age"},
            },
        )
    )

    disease_stage_prop = (
        PropertyBuilder("disease_stage")
        .set_dtype("int")
        .set_shape([metadata["num_agents"], 1])
        .set_initialization(
            "read_from_file",
            {"file_path": {"value": "${simulation_metadata.disease_stage_file}"}},
        )
    )

    citizen_builder.add_property(age_prop).add_property(disease_stage_prop)
    state_builder.add_agent("citizens", citizen_builder)

    # Add environment variables
    env_builder = EnvironmentBuilder()
    sf_infector = (
        PropertyBuilder("SFInfector")
        .set_dtype("float")
        .set_shape([5])
        .set_value([0.0, 0.33, 0.72, 0.0, 0.0])
    )
    env_builder.add_variable(sf_infector)
    state_builder.set_environment(env_builder)

    # Add network configuration
    network_builder = NetworkBuilder()
    network_builder.add_network(
        "infection_network",
        "network_from_file",
        {"file_path": "${simulation_metadata.infection_network_file}"},
    )
    state_builder.set_network(network_builder)

    # Set state in config
    config.set_state(state_builder.to_dict())

    # Add substeps
    # Transmission substep
    transmission_step = SubstepBuilder("Transmission", "Transmission of new infections")
    transmission_step.add_active_agent("citizens")

    # Add policy
    policy_builder = PolicyBuilder()
    policy_builder.add_policy(
        "make_isolation_decision",
        "MakeIsolationDecision",
        {"age": "agents/citizens/age"},
        ["isolation_decision"],
        {"align_vector": {"dtype": "float", "value": 0.3, "shape": [6]}},
    )
    transmission_step.set_policy("citizens", policy_builder)

    # Add transition
    transition_builder = TransitionBuilder()
    transition_builder.add_transition(
        "new_transmission",
        "NewTransmission",
        {
            "disease_stage": "agents/citizens/disease_stage",
            "age": "agents/citizens/age",
            # ... other inputs ...
        },
        ["disease_stage", "next_stage_time", "infected_time", "daily_infected"],
        {"R2": {"value": 4.75, "learnable": True}},
    )
    transmission_step.set_transition(transition_builder)

    config.add_substep("0", transmission_step)

    # Save the config
    config.save_yaml("config.yaml")


if __name__ == "__main__":
    create_covid_config()
