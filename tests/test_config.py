from omegaconf import OmegaConf as oc

from agent_torch.config import Configurator
from fixtures.config import config, agent_count, agent_properties


def test_adding_metadata(config):
    """
    Ensure that metadata is stored correctly.
    """
    config.add_metadata("num_episodes", 7)

    assert config.get("simulation_metadata.num_episodes") == 7


def test_adding_agents_and_properties(config, agent_count, agent_properties):
    """
    Ensure that agents, as well as their properties, are added correctly.
    """
    config.add_agents("minions", agent_count)
    config.add_property("state.agents.minions", "type", **agent_properties["type"])

    assert oc.is_dict(config.get("state.agents.minions"))
    assert oc.to_object(config.get("state.agents.minions")) == {
        "number": agent_count,
        "properties": agent_properties,
    }


def test_interpolation(config):
    """
    Ensure references to other variables are replaced with their values.
    """
    config.add_metadata("num_episodes", 3)
    config.add_metadata("num_steps", "${simulation_metadata.num_episodes}")

    assert config.get("simulation_metadata.num_steps") == 3
