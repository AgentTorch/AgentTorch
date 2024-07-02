import pandas as pd
from agent_torch.data.census.census_loader import CensusDataLoader

# Path to the population data file. Should be updated with the actual file path.
# Should be a pickle file with the population data, in dataframe format.
POPULATION_DATA_PATH = None

# Initialize the census data loader with specified number of CPUs and parallel processing option.
census_data_loader = CensusDataLoader(n_cpu=8, use_parallel=True)

# Export the population data for the specified region. 
# The data will be exported to folder named "region" under "populations" folder.
census_data_loader.export(population_data_path=POPULATION_DATA_PATH,region="astoria")

# Sample top k rows of the population data, for experimentation.
# The data will be exported to folder named "region" under "populations" folder.
census_data_loader.export(population_data_path=POPULATION_DATA_PATH, region = 'astoria',num_individuals=100)