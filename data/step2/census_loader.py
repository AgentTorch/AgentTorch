DATA_LOADER_PATH = ''

import numpy as np
import pandas as pd
import sys
sys.path.append(DATA_LOADER_PATH)

from process.base_pop import base_pop_wrapper
from process.household import household_wrapper

class CensusDataLoader:
    def __init__(self, use_parallel=False, n_cpu=8):
        self.use_parallel = use_parallel
        self.n_cpu = n_cpu
        
        self.population_df = None
        self.address_df = None
    
    def generate_basepop(self, pop_data, pop_mapping, area_selector=None, save_path=None):
        self.population_df, self.address_df = base_pop_wrapper(input_data=pop_data, input_mapping=pop_mapping,
                                                         use_parallel=self.use_parallel,n_cpu=self.n_cpu,
                                                         area_selector=area_selector)
        
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
                                                         geo_address_data=geo_address_data,
                                                         use_parallel=self.use_parallel,n_cpu=self.n_cpu)
    
        if save_path is not None:
            base_population.to_pickle(save_path)