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
runner.config['simulation_metadata']['device'] = "cpu"
print(runner.config['simulation_metadata'])
runner.init()
learn_params = [(name, params) for (name, params) in runner.named_parameters()]

#
# def map_and_replace_tensor(input_string):
#     # Split the input string into its components
#     parts = input_string.split('.')
#
#     # Extract the relevant parts
#     function = parts[1]
#     index = parts[2]
#     sub_func = parts[3]
#     arg_type = parts[4]
#     var_name = parts[5]
#
#     def getter_and_setter(runner, new_value=None, mode_calibrate=True):
#         substep_type = getattr(runner.initializer, function)
#         substep_function = getattr(substep_type[str(index)], sub_func)
#
#         if mode_calibrate:
#             current_tensor = getattr(substep_function, 'calibrate_' + var_name)
#         else:
#             current_tensor = getattr(getattr(substep_function, 'learnable_args'), var_name)
#
#         if new_value is not None:
#             # assert new_value.requires_grad == current_tensor.requires_grad
#             if mode_calibrate:
#                 setvar_name = 'calibrate_' + var_name
#                 setattr(substep_function, setvar_name, new_value)
#                 current_tensor = getattr(substep_function, setvar_name)
#             else:
#                 setvar_name = var_name
#                 subfunc_param = getattr(substep_function, 'learnable_args')
#                 setattr(subfunc_param, setvar_name, new_value)
#                 current_tensor = getattr(subfunc_param, setvar_name)
#
#             return current_tensor
#         else:
#             return current_tensor
#
#     return getter_and_setter


def get_trajectory(parameters: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    # print("Computing Trajectory")
    # print("Parameters: ", parameters)
    # print("-------------------")

    new_tensor = parameters['transmission_rate']
    # old_tensor = learn_params[0][1]

    # assert new_tensor.shape == old_tensor.shape

    input_string = learn_params[0][0]
    params_dict = {input_string: new_tensor}
    runner._set_parameters(params_dict)

    # input_string = learnable_params[0][0]
    # tensorfunc = map_and_replace_tensor(input_string)
    # current_tensor = tensorfunc(runner, new_tensor, mode_calibrate=True)

    # num_steps_per_episode = runner.config['simulation_metadata']['num_steps_per_episode']
    num_steps_per_episode = 5
    runner.step(num_steps_per_episode)

    daily_infected = runner.state_trajectory[-1][-1]['environment']['daily_infected']
    runner.reset()

    if new_tensor.requires_grad:
        assert daily_infected.requires_grad, "input requires grad but output had no grad."

    return OrderedDict(
        daily_infected=daily_infected[..., None]
    )


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    for i in range(5):
        start_time = time.time()
        # new_tensor = torch.nn.Parameter(torch.tensor([3.5, 4.2, 5.6], requires_grad=True)[:, None])
        new_tensor = torch.tensor([3.5, 4.2, 5.6], requires_grad=True)[:, None]
        predictions = get_trajectory(OrderedDict(transmission_rate=new_tensor))
        predictions["daily_infected"].sum().backward()
        end_time = time.time()
        print("time consumed: ", end_time - start_time)

        # old_tensor = learn_params[0][1]
        # print("Old Parameter Grads", [param.grad for param in old_tensor])
        # print("New Parameter Grads", [param.grad for param in new_tensor])
        print("New parameter grads", new_tensor.grad)

        runner.reset()


# if __name__ == "__main__":
#     x = torch.tensor([2.], requires_grad=True)
#
#     y = x ** 2.
#
#     y.backward()
#
#     print(x.grad)

# breakpoint()

# learn_params = [(name, params) for (name, params) in runner.named_parameters()]
# new_tensor = torch.tensor([3.5, 4.2, 5.6], requires_grad=True)
# input_string = learn_params[0][0]

# params_dict = {input_string: new_tensor}
# runner._set_parameters(params_dict)

# num_steps_per_episode = runner.config['simulation_metadata']['num_steps_per_episode']
# runner.step(num_steps_per_episode)
# traj = runner.state_trajectory[-1][-1]
# preds = traj['environment']['daily_infected']