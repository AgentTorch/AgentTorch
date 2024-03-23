from abc import ABC, abstractmethod
import os
import yaml
from AgentTorch.helpers.general import read_config

class DataLoaderBase(ABC):
    
    @abstractmethod
    def __init__(self, data_dir,model):
        self.data_dir = data_dir
        self.opdyn_registry = None
        self.model = model
    
    @abstractmethod
    def get_config(self):
        pass
    
    @abstractmethod
    def _set_input_data_dir(self):
        pass
    
    def _get_config_path(self, model):
        model_path = self._get_folder_path(model)
        return os.path.join(model_path, 'config_refactor.yaml')
    
    def _get_folder_path(self,folder_name):
        for root, dirs, files in os.walk(os.path.expanduser('~')):
            if folder_name in dirs:
                return os.path.join(root, folder_name)
        return None

    def _get_input_data_path(self, data):
        input_data_dir = self._get_folder_path(self.data_dir)
        return os.path.join(input_data_dir, data)
    
    def _set_input_data_dir(self,config_path, attribute, region):
        input_data_dir = self._get_input_data_path(region)
        return self._set_config_attribute(config_path, attribute, input_data_dir)
    
    def _set_config_attribute(self, config_path,attribute, value):
        config_path = self._get_config_path(self.model)
        config = read_config(config_path)
        # with open(config_path, 'r') as f:
        #     config = yaml.load(f, Loader=yaml.FullLoader)
        
        config['simulation_metadata'][attribute] = value
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        return config_path
    

class DataLoader(DataLoaderBase):
    def __init__(self, model, region, population_size):
        super().__init__('simulator_data',model)
        self.config_path = self._get_config_path(model)
        self.config_path = self.set_input_data_dir(self.config_path,region)
        self.config = self._read_config(self.config_path)
    
    def _read_config(self,config_path):
        config = read_config(str(config_path))
        return config
    
    def set_input_data_dir(self, config_path,region):
        return self._set_config_attribute(config_path,'population_dir', region)
    
    def set_population_size(self, config_path, num_agents):
        return self._set_config_attribute(config_path,'num_agents', num_agents)
    
    def get_config(self):
        return self.config