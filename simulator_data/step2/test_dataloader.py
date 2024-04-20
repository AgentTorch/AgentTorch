import os
import numpy as np
import pandas as pd
import pdb

from census_loader import CensusDataLoader

STEP2_DATA_PATH = '/u/ayushc/projects/GradABM/MacroEcon/simulator_data/NYC/output_pop_data/NYC_POP.pkl'
STEP2_SAVE_ROOT = '/u/ayushc/projects/GradABM/MacroEcon/tmp/'

pop_data_df = pd.read_pickle(STEP2_DATA_PATH)

area_selector = pop_data_df['age_gender']['area'][:1]
geo_mapping = None

loader = CensusDataLoader(n_cpu=8, use_parallel=False, area_list=area_selector, geo_mapping=geo_mapping)

save_population_data_path = os.path.join(STEP2_SAVE_ROOT, 'population_data.pkl')

loader.generate_basepop(input_data=pop_data_df, save_path=save_population_data_path)

pdb.set_trace()