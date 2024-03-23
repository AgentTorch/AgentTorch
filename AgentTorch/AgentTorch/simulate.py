from AgentTorch.dataloader import DataLoader
import importlib

class Executor():
    def __init__(self,model,region, population_size) -> None:    
        data_loader = DataLoader(model, region, population_size)
        self.config = data_loader.get_config()
        self.runner = self._get_runner(model)
        self.runner.init()
        
    def execute(self):
        self.runner.execute()
    
    def _get_runner(self, model):
        module_name = f'{model}.simulator'
        module = importlib.import_module(module_name)
        simulaton_runner = module.SimulationRunner
        registry = self._get_registry(model)
        runner = simulaton_runner(self.config, registry)
        return runner
    
    def _get_registry(self, model):
        module_name = f'{model}.simulator'
        module = importlib.import_module(module_name)
        simulation_registry = module.simulation_registry
        registry = simulation_registry()
        return registry