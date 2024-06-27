from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation

from agent_torch.models import covid
from agent_torch.populations import astoria

loader = LoadPopulation(astoria)
simulation = Executor(model=covid, pop_loader=loader)
runner = simulation.runner

runner.init()

num_steps_per_episode = runner.config['simulation_metadata']['num_steps_per_episode']
runner.step(num_steps_per_episode)