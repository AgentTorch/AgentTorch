# main.py
# runs the simulation

import argparse
from tqdm import trange

from agent_torch.core import Registry, Runner
from agent_torch.core.helpers import read_config, read_from_file, grid_network
from substeps import *
from helpers import *

from plot import Plot

print(":: execution started")

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="path to yaml config file")
config_file = parser.parse_args().config

config = read_config(config_file)
metadata = config.get("simulation_metadata")
num_episodes = metadata.get("num_episodes")
num_steps_per_episode = metadata.get("num_steps_per_episode")
visualize = metadata.get("visualize")

registry = Registry()
registry.register(read_from_file, "read_from_file", "initialization")
registry.register(grid_network, "grid", key="network")
registry.register(map_network, "map", key="network")

runner = Runner(config, registry)
runner.init()

print(":: preparing simulation...")

visual = Plot(metadata.get("max_x"), metadata.get("max_y"))
for episode in range(num_episodes):
    runner.reset()

    for step in trange(num_steps_per_episode, desc=f":: running simulation {episode}"):
        runner.step(1)
        visual.capture(step, runner.state)
    visual.compile(episode)

print(":: execution completed")
