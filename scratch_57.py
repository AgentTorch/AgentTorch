from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation

from agent_torch.models import covid
from agent_torch.populations import astoria
from collections import OrderedDict
import torch

loader = LoadPopulation(astoria)
simulation = Executor(model=covid, pop_loader=loader)
runner = simulation.runner

runner.init()

def get_trajectory(parameters: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:

    assert (runner.initializer.transition_function['0']['new_transmission'].calibrate_R2.shape
            == parameters['transmission_rate'].shape)
    assert parameters['transmission_rate'].requires_grad

    runner.initializer.transition_function['0']['new_transmission'].calibrate_R2 = parameters['transmission_rate']

    num_steps_per_episode = runner.config['simulation_metadata']['num_steps_per_episode']
    runner.step(num_steps_per_episode)

    daily_deaths = runner.state_trajectory[-1][-1]['environment']['daily_deaths']
    print(daily_deaths)
    print(daily_deaths.shape)

    runner.reset()

    return OrderedDict(
        daily_deaths=daily_deaths
    )

get_trajectory(OrderedDict(transmission_rate=torch.tensor([0.5, 0.5, 0.5], requires_grad=True)[:, None]))
