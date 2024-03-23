from AgentTorch.dataloader import DataLoader
import importlib

class AgentSim():
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
        opdyn_runner = module.OpDynRunner
        registry = self._get_registry(model)
        runner = opdyn_runner(self.config, registry)
        return runner
    
    def _get_registry(self, model):
        module_name = f'{model}.simulator'
        module = importlib.import_module(module_name)
        opdyn_registry = module.opdyn_registry
        registry = opdyn_registry()
        return registry