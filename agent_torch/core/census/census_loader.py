import pandas as pd
import os
import json

from agent_torch.core.census.generate.base_pop import base_pop_wrapper
from agent_torch.core.census.generate.household import household_wrapper
from agent_torch.core.census.generate.mobility_network import mobility_network_wrapper
import populations


class CensusDataLoader:
    def __init__(self, use_parallel=False, n_cpu=8):
        self.use_parallel = use_parallel
        self.n_cpu = n_cpu
        self.population_dir = populations.__path__[0]

    def generate_basepop(
        self, input_data, region, save_path=None, area_selector=None, export=True
    ):
        self.population_df, self.address_df = base_pop_wrapper(
            input_data,
            area_selector=area_selector,
            use_parallel=self.use_parallel,
            n_cpu=self.n_cpu,
        )

        if save_path is not None:
            self.population_df.to_pickle(save_path)

        if export:
            save_dir = os.path.join(self.population_dir, region)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.export(save_dir)

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
            save_dir = os.path.join(self.population_dir, region)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.export(save_dir)

    def generate_mobility_networks(self, num_steps, mobility_mapping, save_path=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

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
            save_path=save_path,
        )

    def export(self, save_dir, population_data_path=None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # creating the demographic files
        if population_data_path is not None:
            df = pd.read_pickle(population_data_path)
        else:
            df = self.population_df.copy()
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
