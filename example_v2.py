from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation, DataLoader

from agent_torch.models import covid
from agent_torch.populations import mock_1m
from agent_torch.populations import astoria
from agent_torch.populations import NYC

# from custom_population import customize

import operator
from functools import reduce
import torch
import time


def set_params(runner, input_string, new_value):
    tensor_func = map_and_replace_tensor(input_string)
    current_tensor = tensor_func(runner, new_value)


def map_and_replace_tensor(input_string):
    # Split the input string into its components
    parts = input_string.split(".")

    # Extract the relevant parts
    function = parts[1]
    index = parts[2]
    sub_func = parts[3]
    arg_type = parts[4]
    var_name = parts[5]

    def getter_and_setter(runner, new_value=None):
        current = runner

        substep_type = getattr(runner.initializer, function)
        substep_function = getattr(substep_type[str(index)], sub_func)
        current_tensor = getattr(substep_function, "calibrate_" + var_name)

        if new_value is not None:
            assert new_value.requires_grad == current_tensor.requires_grad
            setvar_name = "calibrate_" + var_name
            setattr(substep_function, setvar_name, new_value)
            current_tensor = getattr(substep_function, "calibrate_" + var_name)
            return current_tensor
        else:
            return current_tensor

    return getter_and_setter


def setup(model, population):
    # Timers
    t0 = time.perf_counter()

    # Simple: Executor auto-detects CUDA and picks the right runner
    t_loader_start = time.perf_counter()
    pop_loader = LoadPopulation(population)
    dl = DataLoader(model, pop_loader)
    simulation = Executor(model=model, data_loader=dl)
    t_loader_end = time.perf_counter()
    
    runner = simulation.runner
    
    # TESTING: Override utils registry (easy to remove this line)
    #override_utils_registry(runner, use_base_utils=True)  # Use base utils
    # override_utils_registry(runner, use_base_utils=False) # Use standard utils
    
    # Time runner.init()
    t_init_start = time.perf_counter()
    runner.init()
    t_init_end = time.perf_counter()

    # Print init timing summary
    loader_exec_s = t_loader_end - t_loader_start
    runner_init_s = t_init_end - t_init_start
    total_init_s = t_init_end - t0
    print(f"\n⏱️ Init timings: loader+executor={loader_exec_s:.3f}s, runner.init()={runner_init_s:.3f}s, total={total_init_s:.3f}s")

    return runner


def simulate(runner):
    num_steps_per_episode = runner.config["simulation_metadata"][
        "num_steps_per_episode"
    ]
    
    runner.step(num_steps_per_episode)
    traj = runner.state_trajectory[-1][-1]
    preds = traj["environment"]["daily_infected"]
    loss = preds.sum()
    
    # Print performance stats if available
    if hasattr(runner, 'get_performance_stats'):
        stats = runner.get_performance_stats()
        print(f"\nPerformance Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    return loss


if __name__ == "__main__":
    script_t0 = time.perf_counter()
    
    runner = setup(covid, astoria)
    learn_params = [(name, params) for (name, params) in runner.named_parameters()]
    
    # Ensure new tensor is on the same device as the runner
    device = runner.device if hasattr(runner, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_tensor = torch.tensor([3.5, 4.2, 5.6], requires_grad=True, device=device)
    
    input_string = learn_params[0][0]
    input_string = "initializer.transition_function.0.new_transmission.learnable_args.R2"

    params_dict = {input_string: new_tensor}
    runner._set_parameters(params_dict)

    # Run simulation
    sim_t0 = time.perf_counter()
    loss = simulate(runner)
    sim_t1 = time.perf_counter()

    # Script timing summary
    script_t1 = time.perf_counter()
    print(f"\n Timings: simulation_step={sim_t1 - sim_t0:.3f}s, script_total={script_t1 - script_t0:.3f}s")

    """
    Tasks to do:
    1. Custom population size 
    2. Init Infections
    3. Set parameters
    4. Visualize values
    """
