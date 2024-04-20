from abc import ABC, abstractmethod
import os
import yaml
import pdb
from AgentTorch.helpers import read_config

class DataLoaderBase(ABC):
    @abstractmethod
    def __init__(self, data_dir, model):
        self.data_dir = data_dir
        self.model = model
    
    @abstractmethod
    def get_config(self):
        pass
    
    @abstractmethod
    def _set_input_data_dir(self):
        pass
    
    def _get_config_path(self, model):
        model_path = self._get_folder_path(model)
        return os.path.join(model_path, 'config.yaml')
    
    def _get_folder_path(self, folder):
        folder_path = folder.__path__[0]
        return folder_path

    def _get_input_data_path(self, data):
        input_data_dir = self._get_folder_path(self.data_dir)
        return os.path.join(input_data_dir, data)
    
    def _set_input_data_dir(self, config, attribute, region):
        input_data_dir = self._get_input_data_path(region)
        return self._set_config_attribute(config, attribute, input_data_dir)
    
    def _set_config_attribute(self, config, attribute, value):
        config['simulation_metadata'][attribute] = value
        return config
            

class DataLoader(DataLoaderBase):
    def __init__(self, model, region, population_size):
        super().__init__('populations',model)
        
        self.config_path = self._get_config_path(model)
        self.config = self._read_config(self.config_path)
        
        self.config = self.set_input_data_dir(self.config, region)
        self.config = self.set_population_size(self.config, population_size)
        
        self._write_config(self.config_path)
            
    def _read_config(self, config_path):
        with open(config_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    
    def _write_config(self, config_path):
        with open(config_path, 'w') as file:
            yaml.dump(self.config, file)
        
        print("Config saved at: ", config_path)

    def set_input_data_dir(self, config_path, region):
        return self._set_config_attribute(config_path,'population_dir', region.__path__[0])
    
    def set_population_size(self, config_path, population_size):
        return self._set_config_attribute(config_path, 'num_agents', population_size)
    
    def get_config(self):
        omega_config = read_config(self.config_path)
        return omega_config