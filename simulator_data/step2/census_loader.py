import numpy as np
import pandas as pd
import sys
import os

from process.base_pop import base_pop_wrapper
from process.household import household_wrapper
from process.mobility_network import mobility_network_wrapper

import pdb

class CensusDataLoader:
    def __init__(self, use_parallel=False, n_cpu=8, area_list=None, geo_mapping=None):
        self.use_parallel = use_parallel
        self.n_cpu = n_cpu
        self.area_list = area_list
        self.geo_mapping = geo_mapping
        
        self.population_df = None
        self.address_df = None
    
    def generate_basepop(self, input_data, save_path=None):                
        self.population_df, self.address_df = base_pop_wrapper(input_data, area_selector=self.area_list, use_parallel=self.use_parallel, n_cpu=self.n_cpu)
        
        if save_path is not None:
            self.population_df.to_pickle(save_path)
        
    def generate_household(self, household_data, household_mapping, geo_address_data=None, save_path=None):
        adult_list = household_mapping['adult_list']
        children_list = household_mapping['children_list']
        
        if self.population_df is None:
            print("Generate base population first!!!")
            return
            
        self.population_df, self.address_df = household_wrapper(household_data, self.population_df, 
                                                         base_address = self.address_df, adult_list=adult_list,
                                                         children_list=children_list,
                                                         geo_address_data=self.geo_mapping,
                                                         use_parallel=self.use_parallel,n_cpu=self.n_cpu)
        if save_path is not None:
            self.population_df.to_pickle(save_path)
            
        
    def generate_mobility_networks(self, num_steps, mobility_mapping, save_path=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        if self.population_df is None:
            print("Generate base population first!!!")
            return
        
        interaction_by_age_dict = mobility_mapping['interaction_map']
        age_by_category_dict = mobility_mapping['age_map']
        
        age_df = self.population_df['age']
        self.mobility_network_paths = mobility_network_wrapper(age_df, num_steps, interaction_by_age_dict, age_by_category_dict, save_path=save_path)
        
    def export(self, save_dir):
        df = self.population_df
        attributes = df.keys()
        mapping_collection = {}
        for attribute in attributes:
            df[attribute],mapping = pd.factorize(df[attribute])
            output_att_path = os.path.join(save_dir, attribute)
            df[attribute].to_pickle(f'{output_att_path}.pickle')
            mapping_collection[attribute] = mapping.tolist()
        output_mapping_path = os.path.join(save_dir, 'population_mapping.json')

        with open(output_mapping_path, 'w') as f:
            json.dump(mapping_collection, f)