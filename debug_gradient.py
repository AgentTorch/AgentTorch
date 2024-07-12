import warnings
warnings.simplefilter("ignore")
from agent_torch.models import covid
from agent_torch.populations import sample

from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation

import torch.nn as nn
import torch

sim = Executor(covid, pop_loader=LoadPopulation(sample))
runner = sim.runner
runner.init()

mode = 1
learnable_params = [(name, param) for (name, param) in runner.named_parameters()]

class LearnableParams(nn.Module):
    def __init__(self, num_params, device='cpu'):
        super().__init__()
        self.device = device
        self.num_params = num_params
        self.learnable_params = nn.Parameter(torch.rand(num_params, device=self.device))
        self.min_values = torch.tensor(2.0,
                                       device=self.device)
        self.max_values = torch.tensor(3.5,
                                       device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        out = self.learnable_params
        ''' bound output '''
        out = self.min_values + (self.max_values -
                                 self.min_values) * self.sigmoid(out)
        return out

def map_and_replace_tensor(input_string):
    # Split the input string into its components
    parts = input_string.split('.')
    
    # Extract the relevant parts
    function = parts[1]
    index = parts[2]
    sub_func = parts[3]
    arg_type = parts[4]
    var_name = parts[5]
    
    def getter_and_setter(runner, new_value=None, mode_calibrate=True):
        substep_type = getattr(runner.initializer, function)
        substep_function = getattr(substep_type[str(index)], sub_func)

        if mode_calibrate:
            current_tensor = getattr(substep_function, 'calibrate_' + var_name)
        else:
            current_tensor = getattr(getattr(substep_function, 'learnable_args'), var_name)
        
        if new_value is not None:
            assert new_value.requires_grad == current_tensor.requires_grad
            if mode_calibrate:
                setvar_name = 'calibrate_' + var_name
                setattr(substep_function, setvar_name, new_value)
                current_tensor = getattr(substep_function, setvar_name)
            else:
                setvar_name = var_name
                subfunc_param = getattr(substep_function, 'learnable_args')
                setattr(subfunc_param, setvar_name, new_value)
                current_tensor = getattr(subfunc_param, setvar_name)

            return current_tensor
        else:
            return current_tensor

    return getter_and_setter

def execute(runner, n_steps=5):
    runner.step(n_steps)
    labels = runner.state_trajectory[-1][-1]['environment']['daily_infected']
    loss = labels.sum()
    return loss

def mode_1_test():
    print(":: test 1 - optimize internal simulation parameters!")

    loss = execute(runner)
    loss.backward()
    learn_params_grad = [(name, param, param.grad) for (name, param) in runner.named_parameters()]
    print("Gradients: ", learn_params_grad)
    print("---"*10)

def mode_2_test():
    print(":: test 2 - optimize deterministic external parameters fed into simulation")
    debug_nn_param = nn.Parameter(torch.tensor([2.7, 3.8, 4.6], requires_grad=True)[:, None])
    input_string = learnable_params[0][0]
    tensorfunc = map_and_replace_tensor(input_string)
    current_tensor = tensorfunc(runner, debug_nn_param, mode_calibrate=False)

    loss = execute(runner)
    loss.backward()
    print("Gradients: ", debug_nn_param.grad)
    print("---"*10)

def mode_3_test():
    print(":: test 3 - optimize generator function that predicts simulation parameters!")
    # sample parameters
    learn_model = LearnableParams(3)
    debug_tensor = learn_model()[:, None]
    # set parameters
    input_string = learnable_params[0][0]
    tensorfunc = map_and_replace_tensor(input_string)
    current_tensor = tensorfunc(runner, debug_tensor, mode_calibrate=True)
    # execute runner
    loss = execute(runner)
    loss.backward()
    # compute gradient
    learn_params_grad = [(param, param.grad) for (name, param) in learn_model.named_parameters()]
    print("Gradients: ", learn_params_grad)
    print("---"*10)

if mode == 1:
    mode_1_test()
elif mode == 2:
    mode_2_test()
elif mode == 3:
    mode_3_test()

# if mode ==  1:
#     print(":: test 1 - optimize internal simulation parameters!")

#     loss = execute(runner)
#     loss.backward()
#     learn_params_grad = [(name, param, param.grad) for (name, param) in runner.named_parameters()]
#     print(learn_params_grad)
#     print("---"*10)

# if mode == 2:
#     print(":: test 2 - optimize deterministic external parameters fed into simulation")
#     debug_nn_param = nn.Parameter(torch.tensor([2.7, 3.8, 4.6], requires_grad=True)[:, None])
#     input_string = learnable_params[0][0]
#     print("Input string: ", input_string)
#     tensorfunc = map_and_replace_tensor(input_string)
#     current_tensor = tensorfunc(runner, debug_nn_param, mode_calibrate=False)

#     loss = execute(runner)
#     loss.backward()
#     print(debug_nn_param.grad)
#     print("---"*10)

# if mode == 3:
#     print(":: test 3 - optimize generator function that predicts simulation parameters!")
#     learn_model = LearnableParams(3)
#     debug_tensor = learn_model()[:, None]
#     input_string = learnable_params[0][0]
#     tensorfunc = map_and_replace_tensor(input_string)
#     current_tensor = tensorfunc(runner, debug_tensor, mode_calibrate=True)
#     print("new value: ", current_tensor)
#     loss = execute(runner)
#     loss.backward()
#     learn_params_grad = [(name, param, param.grad) for (name, param) in learn_model.named_parameters()]
#     print(learn_params_grad)
#     print("---"*10)