def debug():
    import os
    import sys

    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    package_root_directory = os.path.dirname(
        os.path.dirname(os.path.dirname(current_directory))
    )
    sys.path.insert(0, package_root_directory)
    sys.path.append(current_directory)


debug()

if __name__ == "__main__":
    from agent_torch.core.executor import Executor
    from agent_torch.core.dataloader import LinkPopulation
    from agent_torch.populations import astoria
    from agent_torch.models import covid

    nyc_population = LinkPopulation(astoria)  # all data for config.simulation_metadata
    simulation = Executor(
        model=covid, pop_loader=nyc_population
    )  # _update_config(), _get_runner()

    simulation.init()
    # simulation.step(num_steps)
    # simulation.optimize(num_episodes, num_steps_per_episode)

    # # support:
    # simulation.grads() - {[p.grad for p in runner.parameters()]}
