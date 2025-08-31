from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation, DataLoader

from agent_torch.models import covid
from agent_torch.populations import astoria, NYC

import operator
from functools import reduce
import torch
import time
import pandas as pd
import os

# Module-level flag to prevent re-registering OmegaConf resolvers
RESOLVERS_DONE = False


def override_utils_registry(use_base: bool):
    """Swap COVID utils between base and optimized variants based on a flag."""
    from agent_torch.core import Registry
    module = covid
    reg: Registry = module.registry
    # Unregister existing utils keys
    for key in [
        "network_from_file",
        "read_from_file",
        "get_lam_gamma_integrals",
        "get_mean_agent_interactions",
        "get_infected_time",
        "get_next_stage_time",
        "load_population_attribute",
        "initialize_id",
    ]:
        if key in reg.initialization_helpers:
            reg.initialization_helpers.pop(key, None)
        if key in reg.network_helpers:
            reg.network_helpers.pop(key, None)
    if use_base:
        from agent_torch.models.covid.substeps import utils_base as utils_mod
    else:
        from agent_torch.models.covid.substeps import utils as utils_mod
    # Re-register from selected module
    reg.register(utils_mod.network_from_file, "network_from_file", key="network")
    reg.register(utils_mod.read_from_file, "read_from_file", key="initialization")
    reg.register(utils_mod.get_lam_gamma_integrals, "get_lam_gamma_integrals", key="initialization")
    reg.register(utils_mod.get_mean_agent_interactions, "get_mean_agent_interactions", key="initialization")
    reg.register(utils_mod.get_infected_time, "get_infected_time", key="initialization")
    reg.register(utils_mod.get_next_stage_time, "get_next_stage_time", key="initialization")
    reg.register(utils_mod.load_population_attribute, "load_population_attribute", key="initialization")
    reg.register(utils_mod.initialize_id, "initialize_id", key="initialization")


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


def setup(model, population, *, use_base_utils: bool, force_device: str):
    global RESOLVERS_DONE
    
    # Timers
    t0 = time.perf_counter()

    # Choose utils impl for this run
    override_utils_registry(use_base_utils)

    # Force device override in config (cpu|cuda|auto)
    assert force_device in ("cpu", "cuda", "auto")
    # Load config via DataLoader and mutate before runner init
    t_loader_start = time.perf_counter()
    pop_loader = LoadPopulation(population)
    dl = DataLoader(model, pop_loader)
    
    # Prevent re-registering OmegaConf resolvers on subsequent runs
    dl.register_resolvers = not RESOLVERS_DONE
    
    # Override device BEFORE creating Executor so transition modules get correct device
    dl.config["simulation_metadata"]["device"] = force_device
    
    simulation = Executor(model=model, data_loader=dl)
    t_loader_end = time.perf_counter()
    
    runner = simulation.runner
    # enforce runner device path selection to avoid mismatched devices
    if force_device in ("cpu", "cuda"):
        runner.use_gpu = (force_device == "cuda")
        runner.device = torch.device(force_device)

    # Time runner.init()
    t_init_start = time.perf_counter()
    runner.init()
    t_init_end = time.perf_counter()

    # Mark resolvers as registered after first run
    RESOLVERS_DONE = True
    
    # Init timing summary
    loader_exec_s = t_loader_end - t_loader_start
    runner_init_s = t_init_end - t_init_start
    total_init_s = t_init_end - t0
    print(f"\n Init timings: loader+executor={loader_exec_s:.3f}s, runner.init()={runner_init_s:.3f}s, total={total_init_s:.3f}s")

    return runner, loader_exec_s, runner_init_s, total_init_s


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
    results = []
    # Bench 1: CUDA + optimized utils (Astoria)
    r1, l1, i1, it1 = setup(covid, astoria, use_base_utils=False, force_device="cuda")
    s1_start = time.perf_counter(); loss1 = simulate(r1); s1_end = time.perf_counter()
    results.append({"pop":"astoria","device":"cuda","utils":"optimized","init_s":it1,"step_s":s1_end-s1_start,"total_s":it1 + (s1_end-s1_start),"loss":float(loss1)})

    # Bench 2: CPU + base utils (Astoria)
    r2, l2, i2, it2 = setup(covid, astoria, use_base_utils=True, force_device="cpu")
    s2_start = time.perf_counter(); loss2 = simulate(r2); s2_end = time.perf_counter()
    results.append({"pop":"astoria","device":"cpu","utils":"base","init_s":it2,"step_s":s2_end-s2_start,"total_s":it2 + (s2_end-s2_start),"loss":float(loss2)})

    # Bench 3: CUDA + optimized utils (NYC)
    r3, l3, i3, it3 = setup(covid, NYC, use_base_utils=False, force_device="cuda")
    s3_start = time.perf_counter(); loss3 = simulate(r3); s3_end = time.perf_counter()
    results.append({"pop":"NYC","device":"cuda","utils":"optimized","init_s":it3,"step_s":s3_end-s3_start,"total_s":it3 + (s3_end-s3_start),"loss":float(loss3)})
    # Build dataframe for quick print/CSV; plotting can be done externally if needed
    df = pd.DataFrame(results)
    print("\nBenchmark results:\n", df)

    # Plotting
    os.makedirs("plots", exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ast = df[df["pop"] == "astoria"]
    if len(ast) == 2:
        labels = ["cuda+opt", "cpu+base"]
        x = range(2)
        # Init times
        plt.figure(figsize=(6,4))
        plt.bar(x, ast.init_s.values, color=["tab:blue","tab:orange"]) 
        plt.xticks(x, labels)
        plt.ylabel("seconds")
        plt.title("Astoria: init times")
        plt.tight_layout(); plt.savefig("plots/astoria_init_times.png")
        # Step times
        plt.figure(figsize=(6,4))
        plt.bar(x, ast.step_s.values, color=["tab:blue","tab:orange"]) 
        plt.xticks(x, labels)
        plt.ylabel("seconds")
        plt.title("Astoria: step times")
        plt.tight_layout(); plt.savefig("plots/astoria_step_times.png")
        # Total times
        plt.figure(figsize=(6,4))
        plt.bar(x, ast.total_s.values, color=["tab:blue","tab:orange"]) 
        plt.xticks(x, labels)
        plt.ylabel("seconds")
        plt.title("Astoria: total times")
        plt.tight_layout(); plt.savefig("plots/astoria_total_times.png")

    nyc = df[(df["pop"] == "NYC") & (df["device"] == "cuda")]
    if len(nyc) == 1:
        labels = ["init","step","total"]
        vals = [float(nyc.init_s.values[0]), float(nyc.step_s.values[0]), float(nyc.total_s.values[0])]
        plt.figure(figsize=(6,4))
        plt.bar(range(3), vals, color=["tab:blue","tab:green","tab:purple"]) 
        plt.xticks(range(3), labels)
        plt.ylabel("seconds")
        plt.title("NYC (cuda+opt): timings")
        plt.tight_layout(); plt.savefig("plots/nyc_cuda_times.png")

    """
    Tasks to do:
    1. Custom population size 
    2. Init Infections
    3. Set parameters
    4. Visualize values
    """
