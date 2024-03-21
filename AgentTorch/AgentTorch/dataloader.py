from abc import ABC, abstractmethod
import importlib
import os

import yaml

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

    # @abstractmethod
    # def get_registry(self):
    #     pass
    
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
    
    def _set_config_attribute(self, config_path,attribute, value):
        config_path = self._get_config_path(self.model)
        input_data_dir = self._get_input_data_path(value)
        # input_data_path = os.path.join(input_data_dir, value)
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        config['simulation_metadata'][attribute] = input_data_dir
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        return config_path
    

    