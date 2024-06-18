from abc import ABC, abstractmethod
import os
import yaml
from agent_torch.helpers import read_config


class DataLoaderBase(ABC):
    @abstractmethod
    def __init__(self, data_dir, model):
        self.data_dir = data_dir
        self.model = model

    @abstractmethod
    def get_config(self):
        pass

    @abstractmethod
    def set_input_data_dir(self):
        pass

    def _get_config_path(self, model):
        model_path = self._get_folder_path(model)
        return os.path.join(model_path, "yamls", "config.yaml")

    def _get_folder_path(self, folder):
        folder_path = folder.__path__[0]
        return folder_path

    def _get_input_data_path(self, data):
        input_data_dir = self._get_folder_path(self.data_dir)
        return os.path.join(input_data_dir, data)

    def set_config_attribute(self, attribute, value):
        self.config["simulation_metadata"][attribute] = value
        self._write_config()  # Save the config file after setting the attribute

    def _write_config(self):
        with open(
            self.config_path, "w"
        ) as file:  # Use the model attribute to get the config path
            yaml.dump(self.config, file)

        print("Config saved at: ", self.config_path)


class DataLoader(DataLoaderBase):
    def __init__(self, model, region, population_size):
        super().__init__("populations", model)

        self.config_path = self._get_config_path(model)
        self.config = self._read_config()
        self.population_size = population_size
        self.set_input_data_dir(region)
        self.set_population_size(population_size)

        self._write_config()

    def _read_config(self):
        with open(self.config_path, "r") as file:
            data = yaml.safe_load(file)
        return data

    def set_input_data_dir(self, region):
        return self.set_config_attribute("population_dir", region.__path__[0])

    def set_population_size(self, population_size):
        self.population_size = population_size  # update current population size
        return self.set_config_attribute("num_agents", population_size)

    def get_config(self):
        omega_config = read_config(self.config_path)
        return omega_config
