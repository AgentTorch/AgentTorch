import argparse
import torch
from tqdm import trange
from simulator import get_registry, get_runner

from agent_torch.core.helpers import read_config

parser = argparse.ArgumentParser(
    description="AgentTorch: million-scale, differentiable agent-based models"
)
parser.add_argument(
    "-c",
    "--config",
    default="yamls/config.yaml",
    help="config file with simulation parameters",
)

args = parser.parse_args()
config_file = args.config

config = read_config(config_file)
registry = get_registry()
runner = get_runner(config, registry)

device = torch.device(runner.config["simulation_metadata"]["device"])
num_episodes = runner.config["simulation_metadata"]["num_episodes"]
num_steps_per_episode = runner.config["simulation_metadata"]["num_steps_per_episode"]

print(":: preparing simulation...")

runner.init()

for episode in trange(num_episodes, desc=":: running episodes"):
    runner.step(num_steps_per_episode)
    runner.reset()

print(":: finished execution")
