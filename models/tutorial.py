import sys
sys.path.append('/Users/shashankkumar/Documents/GitHub/MacroEcon/AgentTorch')
sys.path.append('/Users/shashankkumar/Documents/GitHub/MacroEcon/AgentTorch/AgentTorch')
sys.path.append('/Users/shashankkumar/Documents/GitHub/MacroEcon/data/step2/process')
sys.path.append('/Users/shashankkumar/Documents/GitHub/MacroEcon/models/macro_economics')
from AgentTorch.helpers import read_config
from AgentTorch.dataloader import DataLoaderBase
import importlib
from AgentTorch.helpers import read_config
from preprocess_data import preprocess_data
from simulator import OpDynRunner, opdyn_registry

class DataLoader(DataLoaderBase):
    def __init__(self, model, region):
        super().__init__('simulator_data',model)
        self.config_path = self._get_config_path(model)
        self.config_path = self._set_input_data_dir(self.config_path,region)
        self.config = self._read_config(self.config_path)
    
    def _read_config(self,config_path):
        config = read_config(str(config_path))
        return config
    
    def _set_input_data_dir(self, config_path,region):
        return self._set_config_attribute(config_path,'population_dir', region)
    
    def get_config(self):
        return self.config
    

class AgentSim():
    def __init__(self,model,region) -> None:    
        data_loader = DataLoader(model, region)
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
    
USER_INP_MODEL = 'macro_economics'
USER_INP_DATA = 'NYC'
agent_sim = AgentSim(USER_INP_MODEL, USER_INP_DATA)
agent_sim.execute()