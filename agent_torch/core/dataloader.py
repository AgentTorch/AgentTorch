from abc import ABC, abstractmethod
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import pandas.api.types as pdt
import torch
import yaml
from agent_torch.core.helpers import read_config


class DataLoaderBase(ABC):
    """base api for model dataloaders.

    handles paths and config write helpers for concrete loaders.
    """
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
        """resolve path to model config.yaml."""
        model_path = self._get_folder_path(model)
        return os.path.join(model_path, "yamls", "config.yaml")

    def _get_folder_path(self, folder):
        """return filesystem path for a module/package."""
        folder_path = folder.__path__[0]
        return folder_path

    def _get_input_data_path(self, data):
        """resolve a path inside the data directory for this loader."""
        input_data_dir = self._get_folder_path(self.data_dir)
        return os.path.join(input_data_dir, data)

    def set_config_attribute(self, attribute, value):
        """set a simulation_metadata attribute and persist config.

        batching of writes is supported by toggling _suppress_config_writes.
        """
        # Convert Path-like objects to strings to prevent YAML serialization issues
        if hasattr(value, "__fspath__"):
            value = str(value)
        self.config["simulation_metadata"][attribute] = value
        # Defer writes during batch setup
        if not getattr(self, "_suppress_config_writes", False):
            self._write_config()

    def _write_config(self):
        with open(
            self.config_path, "w", encoding="utf-8"
        ) as file:  # Use the model attribute to get the config path
            yaml.dump(self.config, file)


class DataLoader(DataLoaderBase):
    """model dataloader that writes runtime config and exposes omega config."""
    def __init__(self, model, population):
        super().__init__("populations", model)

        self.config_path = self._get_config_path(model)
        self.config = self._read_config()
        self.population_size = population.population_size
        # Batch config updates to avoid redundant writes
        self._suppress_config_writes = True
        self.set_input_data_dir(population.population_folder_path)
        self.set_population_size(population.population_size)
        self.register_resolvers = True
        self._write_config()

    def _read_config(self):
        """read config.yaml into a python dict."""
        with open(self.config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data

    def set_input_data_dir(self, population_dir):
        return self.set_config_attribute("population_dir", population_dir)

    def set_population_size(self, population_size):
        self.population_size = population_size  # update current population size
        return self.set_config_attribute("num_agents", population_size)

    def get_config(self):
        """load and return the resolved (omega) config once per process."""
        omega_config = read_config(self.config_path, self.register_resolvers)
        self.register_resolvers = False
        return omega_config


class LoadPopulation:
    """eager loader for population pickle files.

    loads numeric series/dataframes as zero-copy torch tensors when possible,
    keeps mixed/non-numeric data as pandas for higher-level mapping.
    """
    def __init__(self, region):
        self.population_folder_path = region.__path__[0]
        self.population_size = 0
        self.load_population()

    def convert_to_parquet(self, pickle_file):
        parquet_file = pickle_file.replace(".pickle", ".parquet").replace(
            ".pkl", ".parquet"
        )
        if not os.path.exists(parquet_file):
            data = pd.read_pickle(pickle_file)

            if isinstance(data, pd.Series):
                data = data.to_frame(name="value")

            data.to_parquet(parquet_file, index=False)

    def load_population(self):
        """load population data with improved performance and robustness.
        - Parallel file reads (threaded)
        - Robust numeric detection
        - Zero-copy NumPy→Torch when possible
        - Authoritative population_size using age.pickle when available
        """
        # Discover files
        pickle_files = glob.glob(
            f"{self.population_folder_path}/*.pickle", recursive=False
        ) + glob.glob(f"{self.population_folder_path}/*.pkl", recursive=False)

        # Establish population size from the first file
        base_size = None
        if pickle_files:
            try:
                _first_obj = pd.read_pickle(pickle_files[0])
                if isinstance(_first_obj, (pd.Series, pd.DataFrame)):
                    base_size = len(_first_obj)
            except Exception:
                base_size = None

        results = {}
        sizes = {}
        last_df = None

        def _load_one(file_path):
            key = os.path.splitext(os.path.basename(file_path))[0]
            try:
                obj = pd.read_pickle(file_path)
            except Exception as e:
                return key, ("error", e), 0, None

            # Series or single-column DF → 1D
            if isinstance(obj, pd.Series) or (isinstance(obj, pd.DataFrame) and obj.shape[1] == 1):
                series = obj if isinstance(obj, pd.Series) else obj.iloc[:, 0]
                n = len(series)
                arr = series.to_numpy()
                if pdt.is_numeric_dtype(series.dtype):
                    # zero-copy where possible to float32
                    arr32 = series.to_numpy(dtype="float32", copy=False)
                    tensor = torch.from_numpy(arr32)
                    return key, ("tensor", tensor), n, None
                else:
                    # keep raw series
                    return key, ("series", series), n, None

            # Multi-column DataFrame
            if isinstance(obj, pd.DataFrame):
                n = len(obj)
                # all numeric?
                if all(pdt.is_numeric_dtype(obj[c]) for c in obj.columns):
                    arr32 = obj.to_numpy(dtype="float32", copy=False)
                    tensor = torch.from_numpy(arr32)
                    return key, ("tensor", tensor), n, obj
                else:
                    return key, ("dataframe", obj), n, obj


        # Parallel read
        with ThreadPoolExecutor(max_workers=min(6, max(1, os.cpu_count() or 4))) as ex:
            fut_map = {ex.submit(_load_one, f): f for f in pickle_files}
            for fut in as_completed(fut_map):
                key, payload, n, df_candidate = fut.result()
                kind, value = payload
                if kind == "error":
                    print(f"   {key:20} -> error loading: {value}")
                    continue
                results[key] = (kind, value)
                sizes[key] = n
                if df_candidate is not None:
                    last_df = df_candidate

        # Assign to self
        for key, (kind, value) in results.items():
            if kind == "tensor":
                setattr(self, key, value)
            elif kind in ("series", "dataframe", "raw"):
                # Keep as pandas/raw for higher-level handling
                if kind != "tensor":
                    msg = "Series" if kind == "series" else ("DataFrame" if kind == "dataframe" else type(value).__name__)
                    print(f"   {key:20} -> stored as raw {msg}{' (mixed/non-numeric dtypes)' if kind=='dataframe' else ''}")
                setattr(self, key, value)

        # Set population size to the first file's length; fallback to any loaded df
        if base_size is not None:
            self.population_size = base_size
        elif last_df is not None:
            self.population_size = len(last_df)
        else:
            self.population_size = 0

        print(
            f"LoadPopulation folder = {self.population_folder_path}, population size = {self.population_size}"
        )


class LinkPopulation(DataLoader):
    def __init__(self, region):
        self.population_folder_path = region.__path__[0]
        self.population_size = 0
        self.load_population()

    def convert_to_parquet(self, pickle_file):
        parquet_file = pickle_file.replace(".pickle", ".parquet").replace(
            ".pkl", ".parquet"
        )
        if not os.path.exists(parquet_file):
            data = pd.read_pickle(pickle_file)

            if isinstance(data, pd.Series):
                data = data.to_frame(name="value")

            data.to_parquet(parquet_file, index=False)

    def load_population(self):
        pickle_files = glob.glob(
            f"{self.population_folder_path}/*.pickle", recursive=False
        ) + glob.glob(f"{self.population_folder_path}/*.pkl", recursive=False)

        # convert existing pickle files to parquet once
        for pickle_file in pickle_files:
            self.convert_to_parquet(pickle_file)

        parquet_files = glob.glob(
            f"{self.population_folder_path}/*.parquet", recursive=False
        )

        last_len = 0
        for file in parquet_files:
            key = os.path.splitext(os.path.basename(file))[0]
            df = pd.read_parquet(file)
            arr = df.to_numpy()
            setattr(self, key, torch.from_numpy(arr).float())
            last_len = len(df)
        self.population_size = last_len
