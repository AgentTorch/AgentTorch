import importlib
import pdb
import sys

from agent_torch.dataloader import DataLoader

class BaseExecutor():
    def __init__(self, model):
        self.model = model
    
    def _get_runner(self, config):
        module_name = f'{self.model.__name__}.simulator'
        module = importlib.import_module(module_name)
        simulaton_runner = module.SimulationRunner
        registry = self._get_registry()
        runner = simulaton_runner(config, registry)
        return runner
    
    def _get_registry(self):
        module_name = f'{self.model.__name__}.simulator'
        module = importlib.import_module(module_name)
        simulation_registry = module.simulation_registry
        sys.path.insert(0, self.model.__path__[0])
        registry = simulation_registry()
        return registry
    
class Executor(BaseExecutor):
    def __init__(self,model,data_loader=None,pop_loader=None) -> None: 
        super().__init__(model) 
        if pop_loader:
            self.pop_loader = pop_loader
            self.data_loader = DataLoader(model,self.pop_loader)
        else:
            self.data_loader = data_loader
        
        self.config = self.data_loader.get_config()
        
        # self.runner = self._get_runner(self.config)
        # self.runner.init()
        # self.calibrator = Calibrator(self.runner)
    def init(self):
        pass
        # self.runner.init()
    def execute(self):
        # self.calibrator.run()
        pass
    
    
