def debug():
    import os
    import sys

    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    package_root_directory = os.path.dirname(
        os.path.dirname(os.path.dirname(current_directory))
    )
    sys.path.insert(0, package_root_directory)
    sys.path.append(current_directory)
debug()

import numpy as np
import pandas as pd
from agent_torch.core.census.census_loader import CensusDataLoader

POPULATION_DATA_PATH = None
HOUSEHOLD_DATA_PATH = None
HOUSEHOLD_MAPPING = {
    "adult_list": ["20t29", "30t39", "40t49", "50t64", "65A"],
    "children_list": ["U19"],
}
HOUSEHOLD_DATA = np.load(HOUSEHOLD_DATA_PATH, allow_pickle=True)
pop_data_df = pd.read_pickle(POPULATION_DATA_PATH)

area_selector = None
geo_mapping = None

census_data_loader = CensusDataLoader(n_cpu=8, use_parallel=True)

census_data_loader.generate_basepop(
    input_data=pop_data_df, region="astoria", area_selector=area_selector
)
census_data_loader.generate_household(
    household_data=HOUSEHOLD_DATA, household_mapping=HOUSEHOLD_MAPPING, region="astoria"
)
