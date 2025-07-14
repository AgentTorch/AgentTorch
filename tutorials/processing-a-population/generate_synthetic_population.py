import numpy as np
import pandas as pd
from agent_torch.data.census.census_loader import CensusDataLoader


# Path to the population data file. Should be updated with the actual file path.
# Path to the household data file. Should be updated with the actual file path.

# Mapping of age groups for adults and children in the household data.
AGE_GROUP_MAPPING = {
    "adult_list": ["20t29", "30t39", "40t49", "50t64", "65A"],  # Age ranges for adults.
    "children_list": ["U19"],  # Age range for children.
}
HOUSEHOLD_DATA_PATH = (
    "docs/tutorials/processing-a-population/sample_data/NYC/household.pkl"
)
POPULATION_DATA_PATH = (
    "docs/tutorials/processing-a-population/sample_data/NYC/population.pkl"
)

# Load household data from the specified path. Ensure the path is correctly set before loading.
HOUSEHOLD_DATA = pd.read_pickle(HOUSEHOLD_DATA_PATH)
# Load population data from the specified path. Ensure the path is correctly set before loading.
BASE_POPULATION_DATA = pd.read_pickle(POPULATION_DATA_PATH)

# Placeholder for area selection criteria, if any. Update or use as needed.
area_selector = None
# Placeholder for geographic mapping data, if any. Update or use as needed.
geo_mapping = None

# Initialize the census data loader with specified number of CPUs and parallel processing option.
census_data_loader = CensusDataLoader(n_cpu=8, use_parallel=True)

# Generate base population data for a specified region, area, and population size.
# The data will be exported to folder named "region" under "populations" folder.
# Parameter 'num_individuals' can be used to specify the number of individuals to generate.
census_data_loader.generate_basepop(
    input_data=BASE_POPULATION_DATA,  # The population data frame.
    region="astoria",  # The target region for generating base population.
    area_selector=area_selector,  # Area selection criteria, if applicable.
)

# Generate household data for the specified region using the loaded household data and mapping.
# The data will be exported to folder named "region" under "populations" folder.
census_data_loader.generate_household(
    household_data=HOUSEHOLD_DATA,  # The loaded household data.
    household_mapping=AGE_GROUP_MAPPING,  # Mapping of age groups for household composition.
    region="astoria",  # The target region for generating households.
)
