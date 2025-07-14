import os
from pathlib import Path
from agent_torch.config import (
    StateBuilder,
    AgentBuilder,
    PropertyBuilder,
    EnvironmentBuilder,
    NetworkBuilder,
    ConfigBuilder,
    PolicyBuilder,
    TransitionBuilder,
    SubstepBuilderWithImpl,
    ObservationBuilder,
)


class RegistryBuilder:
    """Helper class to track and generate registry setup code."""

    def __init__(self):
        self.components = []
        self.utils = []

    def add_component(self, name, import_path, component_type, key):
        self.components.append(
            {"name": name, "import": import_path, "type": component_type, "key": key}
        )

    def add_util(self, name, import_path, key):
        self.utils.append({"name": name, "import": import_path, "key": key})

    def generate_init_code(self):
        """Generate code for __init__.py"""
        code = [
            '"""COVID-19 simulation model."""',
            "from agent_torch.core import Registry",
            "from agent_torch.core.helpers import *\n",
            "from .substeps import *\n",
            "# Create and populate registry as a module-level variable",
            "registry = Registry()",
        ]

        return "\n".join(code)


def create_substeps_structure(base_dir):
    """Create substeps directory with a flat file structure."""
    # Create main substeps directory
    substeps_dir = base_dir / "substeps"
    substeps_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py
    init_file = substeps_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Substep implementations for COVID simulation."""\n')

    return substeps_dir


def setup_covid_simulation():
    """Setup a complete COVID simulation structure in the examples directory."""

    # Get the examples directory path
    examples_dir = Path(__file__).parent

    # Create model directory structure
    model_dir = examples_dir / "models" / "covid"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create substeps directory
    substeps_dir = create_substeps_structure(model_dir)

    # Create yamls directory
    yamls_dir = model_dir / "yamls"
    yamls_dir.mkdir(exist_ok=True)

    # Initialize registry builder
    registry = RegistryBuilder()

    # Create the main config builder
    config = ConfigBuilder()

    # Set simulation metadata
    metadata = {
        "ALIGN_LLM": True,
        "EXECUTION_MODE": "heuristic",
        "EXPOSED_TO_INFECTED_TIME": 5,
        "EXPOSED_VAR": 1,
        "INCLUDE_WEEK_COUNT": True,
        "INFECTED_TO_RECOVERED_TIME": 5,
        "INFECTED_VAR": 2,
        "INFINITY_TIME": 130,
        "MORTALITY_VAR": 4,
        "NEIGHBORHOOD": "Astoria",
        "NUM_TRAIN_WEEKS": 2,
        "NUM_WEEKS": 3,
        "device": "cpu",
        "calibration": False,
        "OPENAI_API_KEY": None,
        "RECOVERED_TO_SUSCEPTIBLE_TIME": 100,
        "RECOVERED_VAR": 3,
        "RESCALE_CONFIG": 0,
        "START_WEEK": 202048,
        "SUSCEPTIBLE_VAR": 0,
        "num_agents": 1000,
        "population_dir": "sample2",
        "age_group_file": "${simulation_metadata.population_dir}/age.pickle",
        "disease_stage_file": "${simulation_metadata.population_dir}/disease_stages.csv",
        "infection_network_file": "${simulation_metadata.population_dir}/mobility_networks/0.csv",
        "num_episodes": 5,
        "num_steps_per_episode": 21,
        "num_substeps_per_step": 2,
    }
    config.set_metadata(metadata)

    # Build state configuration and track registry components
    state_builder = StateBuilder()

    # Add citizen agent
    citizen_builder = AgentBuilder("citizens", metadata["num_agents"])

    # Add agent properties and track initialization functions
    age_prop = (
        PropertyBuilder("age")
        .set_dtype("int")
        .set_shape([metadata["num_agents"], 1])
        .set_learnable(False)
        .set_value(20)
    )  #     .set_initialization("load_population_attribute", {
    #         "file_path": {"value": "${simulation_metadata.age_group_file}"},
    #         "attribute": {"value": "age"}
    #     })
    # registry.add_util("load_population_attribute", ".substeps.utils", "initialization")

    disease_stage_prop = (
        PropertyBuilder("disease_stage")
        .set_dtype("int")
        .set_shape([metadata["num_agents"], 1])
        .set_learnable(False)
        .set_value(0)
    )  # .set_initialization("read_from_file", {
    #     "file_path": {"value": "${simulation_metadata.disease_stage_file}"}
    # })
    # registry.add_util("read_from_file", ".substeps.utils", "initialization")

    citizen_builder.add_property(age_prop).add_property(disease_stage_prop)
    state_builder.add_agent("citizens", citizen_builder)

    # Add environment variables
    env_builder = EnvironmentBuilder()
    sf_infector = (
        PropertyBuilder("SFInfector")
        .set_dtype("float")
        .set_shape([5])
        .set_learnable(False)
        .set_value([0.0, 0.33, 0.72, 0.0, 0.0])
    )
    env_builder.add_variable(sf_infector)
    state_builder.set_environment(env_builder)

    # Add network configuration
    # network_builder = NetworkBuilder()
    # network_builder.add_network(
    #     "infection_network",
    #     "network_from_file",
    #     {"file_path": "${simulation_metadata.infection_network_file}"}
    # )
    # state_builder.set_network(network_builder)

    # Set state in config
    config.set_state(state_builder.to_dict())

    # Create first substep (Disease Transmission) using SubstepBuilderWithImpl
    transmission_substep = SubstepBuilderWithImpl(
        name="Transmission",
        description="Disease transmission including isolation decisions",
        output_dir=str(substeps_dir),
    )
    transmission_substep.add_active_agent("citizens")

    # Set empty observation in config
    transmission_substep.config["observation"] = {"citizens": None}

    # Add policy for isolation decision
    policy_builder = PolicyBuilder()
    align_vector = PropertyBuilder.create_argument(
        name="align LLM agents to the populations", value=0.3, shape=[6], learnable=True
    )
    policy_builder.add_policy(
        "make_isolation_decision",
        "MakeIsolationDecision",
        {"age": "agents/citizens/age"},
        ["isolation_decision"],
        {"align_vector": align_vector},
    )
    transmission_substep.add_policy("citizens", policy_builder)

    # Add transition for transmission
    transition_builder = TransitionBuilder()
    r2_param = PropertyBuilder.create_argument(
        name="R2 parameter", value=4.75, learnable=True
    )
    transition_builder.add_transition(
        "new_transmission",
        "NewTransmission",
        {
            "disease_stage": "agents/citizens/disease_stage",
            "age": "agents/citizens/age",
        },
        ["disease_stage"],
        {"R2": r2_param},
    )
    transmission_substep.set_transition(transition_builder)

    # Create second substep (SEIRM Progression)
    progression_substep = SubstepBuilderWithImpl(
        name="DiseaseProgression",
        description="SEIRM disease stage progression",
        output_dir=str(substeps_dir),
    )
    progression_substep.add_active_agent("citizens")

    # Set empty observation and policy in config
    progression_substep.config["observation"] = {"citizens": None}
    progression_substep.config["policy"] = {"citizens": None}

    # Add transition for SEIRM progression
    progression_transition = TransitionBuilder()
    mortality_rate = PropertyBuilder.create_argument(
        name="Mortality rate parameter", value=0.01, learnable=True
    )
    progression_transition.add_transition(
        "seirm_progression",
        "SEIRMProgression",
        {"disease_stage": "agents/citizens/disease_stage"},
        ["disease_stage"],
        {"mortality_rate": mortality_rate},
    )
    progression_substep.set_transition(progression_transition)

    # Generate implementation files for both substeps
    print("\nGenerating substep implementation files...")
    generated_files = transmission_substep.generate_implementations()
    generated_files.extend(progression_substep.generate_implementations())
    for file_path in generated_files:
        print(f"Created: {file_path}")

    # Add both substeps to config in correct order
    config.add_substep("0", transmission_substep)
    config.add_substep("1", progression_substep)

    # Generate and write __init__.py
    init_content = registry.generate_init_code()
    init_path = model_dir / "__init__.py"
    init_path.write_text(init_content)
    print(f"\nGenerated __init__.py with registry setup")

    # Save the config
    config_path = yamls_dir / "config.yaml"
    config.save_yaml(str(config_path))
    print(f"\nGenerated config file: {config_path}")

    print(f"\nModel setup complete! Structure created in: {model_dir}")
    print("\nDirectory structure:")
    print(f"examples/models/covid/")
    print(f"├── __init__.py")
    print(f"├── substeps/")
    print(f"│   ├── __init__.py")
    print(f"│   ├── make_isolation_decision.py")
    print(f"│   ├── new_transmission.py")
    print(f"│   └── seirm_progression.py")
    print(f"└── yamls/")
    print(f"    └── config.yaml")

    print("\nYou can now:")
    print("1. Implement the TODO sections in the generated template files")
    print("2. Run the simulation using: python -m agent_torch.examples.run_covid_sim")

    return config


if __name__ == "__main__":
    setup_covid_simulation()
