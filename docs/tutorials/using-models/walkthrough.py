if __name__ == "__main__":
    from agent_torch.core.executor import Executor
    from agent_torch.core.dataloader import LinkPopulation
    from agent_torch.populations import NYC
    from agent_torch.models import covid

    nyc_population = LinkPopulation(NYC)  # all data for config.simulation_metadata
    simulation = Executor(
        model=covid, pop_loader=nyc_population
    ) 

    simulation.init()
