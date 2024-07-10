from agent_torch.populations import sample
from agent_torch.models import covid
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation
import torch

from collections import OrderedDict
import time

loader = LoadPopulation(sample)
simulation = Executor(model=covid, pop_loader=loader)

runner = simulation.runner
runner.init()
learn_params = [(name, params) for (name, params) in runner.named_parameters()]

def get_trajectory(parameters: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    new_tensor = parameters['transmission_rate']
    old_tensor = learn_params[0][1]

    assert new_tensor.requires_grad
    assert new_tensor.shape == old_tensor.shape

    input_string = learn_params[0][0]    
    params_dict = {input_string: new_tensor}
    runner._set_parameters(params_dict)

    num_steps_per_episode = runner.config['simulation_metadata']['num_steps_per_episode']
    runner.step(num_steps_per_episode)

    daily_infected = runner.state_trajectory[-1][-1]['environment']['daily_infected']
    runner.reset()

    return OrderedDict(
        daily_infected=daily_infected
    )


for i in range(5):
    start_time = time.time()
    new_tensor = torch.tensor([3.5, 4.2, 5.6], requires_grad=True)[:, None]
    predictions = get_trajectory(OrderedDict(transmission_rate=new_tensor))
    end_time = time.time()
    print("time consumed: ", end_time - start_time)

breakpoint()

# learn_params = [(name, params) for (name, params) in runner.named_parameters()]
# new_tensor = torch.tensor([3.5, 4.2, 5.6], requires_grad=True)
# input_string = learn_params[0][0]

# params_dict = {input_string: new_tensor}
# runner._set_parameters(params_dict)

# num_steps_per_episode = runner.config['simulation_metadata']['num_steps_per_episode']
# runner.step(num_steps_per_episode)
# traj = runner.state_trajectory[-1][-1]
# preds = traj['environment']['daily_infected']