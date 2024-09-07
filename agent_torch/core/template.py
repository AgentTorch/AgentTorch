# core/template.py
# exports the `create_from_template` function

import importlib
import sys

from pathlib import Path
from omegaconf import OmegaConf as oc
from tqdm import trange
from agent_torch.core import Runner, Registry
from agent_torch.core.helpers import read_config, read_from_file
from agent_torch.visualize import GeoPlot


class Simulator:
    def __init__(self, config, registry, runner):
        self.config = config
        self.registry = registry
        self.runner = runner

    def execute(self):
        print(":: execution started")
        metadata = self.config.get("simulation_metadata")
        num_episodes = metadata.get("num_episodes")
        num_steps_per_episode = metadata.get("num_steps_per_episode")

        print(":: preparing simulation...")
        geoplot = GeoPlot(self.config, metadata.get("cesium_token"))

        for episode in trange(num_episodes, desc=f":: running episode", ncols=108):
            self.runner.reset()
            for _ in trange(num_steps_per_episode, desc=f":: running steps", ncols=72):
                self.runner.step(1)

            geoplot.visualize(
                name=f"solar-network-{episode}",
                state_trajectory=self.runner.state_trajectory,
                entity_position="agents/bap/position",
                entity_property="agents/bap/wallet",
            )

        print(":: execution completed")


def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def parse_entity(config, name, entities, data_dir):
    for entity, count in entities.items():
        entity_config = config["state"][name][entity]
        if config["simulation_metadata"].get(f"num_{entity}", None) is None:
            entity_config["number"] = count
        else:
            config["simulation_metadata"][f"num_{entity}"] = count

        entity_data = data_dir / entity  # <data-dir>/<entity-id>
        if not entity_data.exists() or not entity_data.is_dir():
            continue

        files = [x for x in entity_data.iterdir() if not x.is_dir()]
        for file in files:
            property_name = file.parts[-1][:-4]  # property_name.csv
            property_config = entity_config["properties"].get(property_name, None)
            if (
                property_config is None
                or property_config["initialization_function"] is None
            ):
                continue

            if (
                property_config["initialization_function"]["generator"]
                == "read_from_file"
            ):
                property_config["initialization_function"]["arguments"]["file_path"][
                    "value"
                ] = str(file)

    return config


def get_model(metadata, data, agents, objects, substeps):
    data_dir = Path(data)

    config = oc.load(metadata["config_path"])
    for k, v in substeps.items():
        if isinstance(v, str):
            substep_config = oc.load(f"{v}/state.yaml")
            config["substeps"][k] = oc.to_object(substep_config)

    config = parse_entity(config, "agents", agents, data_dir)
    config = parse_entity(config, "objects", objects, data_dir)

    config = oc.to_object(config)

    registry = Registry()
    registry.register(read_from_file, "read_from_file", "initialization")

    runner = Runner(config, registry)
    runner.init()

    return config, registry, runner


def load_from_template(model, data, agents, objects, substeps):
    module = load_module("model_template", f"{model}/__init__.py")

    metadata = module.get_model_metadata()
    config, registry, runner = get_model(metadata, data, agents, objects, substeps)

    simulator = Simulator(config, registry, runner)
    return simulator
