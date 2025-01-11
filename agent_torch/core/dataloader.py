from abc import ABC, abstractmethod
import glob
import os
import pandas as pd
import torch
import yaml
import pdb
import dask.dataframe as dd
from agent_torch.core.helpers import read_config


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


class DataLoader(DataLoaderBase):
    def __init__(self, model, population):
        super().__init__("populations", model)

        self.config_path = self._get_config_path(model)
        self.config = self._read_config()
        self.population_size = population.population_size
        self.set_input_data_dir(population.population_folder_path)
        self.set_population_size(population.population_size)
        self.register_resolvers = True
        self._write_config()

    def _read_config(self):
        with open(self.config_path, "r") as file:
            data = yaml.safe_load(file)
        return data

    def set_input_data_dir(self, population_dir):
        return self.set_config_attribute("population_dir", population_dir)

    def set_population_size(self, population_size):
        self.population_size = population_size  # update current population size
        return self.set_config_attribute("num_agents", population_size)

    def get_config(self):
        omega_config = read_config(self.config_path, self.register_resolvers)
        self.register_resolvers = False
        return omega_config


class LoadPopulation:
    def __init__(self, region):
        self.population_folder_path = region.__path__[0]
        self.population_size = 0
        self.load_population()

    def convert_to_parquet(self, pickle_file):
        parquet_file = pickle_file.replace(".pickle", ".parquet")
        if not os.path.exists(parquet_file):
            data = pd.read_pickle(pickle_file)

            if isinstance(data, pd.Series):
                data = data.to_frame(name="value")  

            data.to_parquet(parquet_file, index=False)

    def load_population(self):
        pickle_files = glob.glob(
            f"{self.population_folder_path}/*.pickle", recursive=False
        )

        for pickle_file in pickle_files:
            self.convert_to_parquet(pickle_file)

        parquet_files = glob.glob(
            f"{self.population_folder_path}/*.parquet", recursive=False
        )

        for file in parquet_files:
            with open(file, "rb") as f:
                key = os.path.splitext(os.path.basename(file))[0]
                df = dd.read_parquet(file)
                df = df.compute()
                setattr(self, key, torch.from_numpy(df.values).float())
        self.population_size = len(df)


class LinkPopulation(DataLoader):
    def __init__(self, region):
        self.population_folder_path = region.__path__[0]
        self.population_size = 0
        self.load_population()

    def convert_to_parquet(self, pickle_file):
        parquet_file = pickle_file.replace(".pickle", ".parquet")
        if not os.path.exists(parquet_file):
            data = pd.read_pickle(pickle_file)

            if isinstance(data, pd.Series):
                data = data.to_frame(name="value") 

            data.to_parquet(parquet_file, index=False)

    def load_population(self):
        pickle_files = glob.glob(
            f"{self.population_folder_path}/*.pickle", recursive=False
        )

        for pickle_file in pickle_files:
            self.convert_pickle_to_parquet(pickle_file)

        parquet_files = glob.glob(
            f"{self.population_folder_path}/*.parquet", recursive=False
        )

        for file in parquet_files:
            with open(file, "rb") as f:
                key = os.path.splitext(os.path.basename(file))[0]
                df = dd.read_parquet(file)
                setattr(self, key, torch.from_numpy(df.values).float())
        self.population_size = len(df)
