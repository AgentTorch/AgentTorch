import pandas as pd
import os
import json

from agent_torch import populations
from agent_torch.data.census.generate.base_pop import base_pop_wrapper
from agent_torch.data.census.generate.household import household_wrapper
from agent_torch.data.census.generate.mobility_network import mobility_network_wrapper


class CensusDataLoader:
    def __init__(self, use_parallel=False, n_cpu=8):
        self.use_parallel = use_parallel
        self.n_cpu = n_cpu
        self.population_dir = populations.__path__[0]
        self.population_df = None

    def generate_basepop(
        self,
        input_data,
        region,
        save_path=None,
        area_selector=None,
        export=True,
        num_individuals=None,
    ):
        """
        Generate base population data for a given region.

        Args:
            input_data (str): Path to the input data file.
            region (str): Name of the region.
            save_path (str, optional): Path to save the population dataframe as a pickle file. Defaults to None.
            area_selector (callable, optional): Function to filter the input data based on area. Defaults to None.
            export (bool, optional): Whether to export the generated data. Defaults to True.

        Returns:
            None

        Raises:
            None

        """
        self.population_df, self.address_df = base_pop_wrapper(
            input_data,
            area_selector=area_selector,
            use_parallel=self.use_parallel,
            n_cpu=self.n_cpu,
        )

        if save_path is not None:
            self.population_df.to_pickle(save_path)

        if num_individuals is not None:
            self.population_df = self.population_df.head(num_individuals)

        if export:
            self.export(region, num_individuals=num_individuals)

    def generate_household(
        self,
        household_data,
        household_mapping,
        region,
        geo_address_data=None,
        save_path=None,
        geo_address_save_path=None,
        export=True,
    ):
        """
        Generate household population based on the provided household data and mapping.

        Args:
            household_data (list): List of household data.
            household_mapping (dict): Mapping of household data.
            region (str): Region for which the household population is generated.
            geo_address_data (optional): Geo address data.
            save_path (optional): Path to save the population dataframe.
            geo_address_save_path (optional): Path to save the address dataframe.
            export (bool): Flag to indicate whether to export the generated population.

        Returns:
            None

        Raises:
            None

        """
        adult_list = household_mapping["adult_list"]
        children_list = household_mapping["children_list"]

        if self.population_df is None:
            print("Generate base population first!!!")
            return

        self.population_df, self.address_df = household_wrapper(
            household_data,
            self.population_df,
            base_address=self.address_df,
            adult_list=adult_list,
            children_list=children_list,
            geo_address_data=geo_address_data,
            use_parallel=self.use_parallel,
            n_cpu=self.n_cpu,
        )
        if save_path is not None:
            self.population_df.to_pickle(save_path)

        if geo_address_data is not None:
            self.address_df.to_pickle(geo_address_save_path)

        if export:
            self.export(region)

    def generate_mobility_networks(
        self, num_steps, mobility_mapping, region, save_path=None
    ):
        """
        Generates mobility networks based on the given parameters.

        Args:
            num_steps (int): The number of steps to generate the mobility networks for.
            mobility_mapping (dict): A dictionary containing the interaction map and age map.
                The interaction map should be a dictionary mapping age groups to interaction probabilities.
                The age map should be a dictionary mapping age groups to age categories.
            save_path (str, optional): The path to save the generated mobility networks. If not provided, the networks will not be saved.

        Returns:
            None

        Raises:
            None

        Notes:
            - This function requires the base population to be generated first.
            - The generated mobility networks will be saved in the specified save_path.

        """
        save_root = os.path.join(self.population_dir, region)
        save_dir = os.path.join(save_root, "mobility_networks")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.population_df is None:
            print("Generate base population first!!!")
            return

        interaction_by_age_dict = mobility_mapping["interaction_map"]
        age_by_category_dict = mobility_mapping["age_map"]

        age_df = self.population_df["age"]
        self.mobility_network_paths = mobility_network_wrapper(
            age_df,
            num_steps,
            interaction_by_age_dict,
            age_by_category_dict,
            save_path=save_dir,
        )

    def export(self, region, population_data_path=None, num_individuals=None):
        """
        Export demographic data for a specific region.

        Args:
            region (str): The name of the region to export data for.
            population_data_path (str, optional): The path to the population data file. If not provided, the default population data will be used. Should be a pickle file.
            top_k (int, optional): The number of top records to export. If not provided, all records will be exported.

        Returns:
            None
        """
        save_dir = os.path.join(self.population_dir, region)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # creating the demographic files
        if population_data_path is not None:
            df = pd.read_pickle(population_data_path)
        else:
            df = self.population_df.copy()

        if num_individuals is not None:
            df = df.head(num_individuals)

        attributes = df.keys()
        mapping_collection = {}
        for attribute in attributes:
            if attribute == "index":
                continue
            df[attribute], mapping = pd.factorize(df[attribute])
            output_att_path = os.path.join(save_dir, attribute)
            df[attribute].to_pickle(f"{output_att_path}.pickle")
            mapping_collection[attribute] = mapping.tolist()
        output_mapping_path = os.path.join(save_dir, "mapping.json")

        with open(output_mapping_path, "w") as f:
            json.dump(mapping_collection, f)
