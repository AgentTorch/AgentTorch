This folder outlines hierarchy of data loaders


# Use-case 1: Execute existing simulation with existing population

from AgentTorch.populations import NZ
from AgentTorch.models import covid19_model
from AgentTorch.simulate import Executor

Executor(covid19_model, NZ)

# Use Case 2: Import your population and run existing simulation
from AgentTorch.loaders import CensusDataLoader
from AgentTorch.simulate import NewExecution

loader = CensusDataLoader()

Step 1: User Inputs Census Data (following AgentTorch templates)
	- population_summary.pkl 
	- household_summary.pkl
	- geo_mapping.pkl
	- mobility_mapping.pkl

Step 2: CensusDataLoader generates synthetic population

- base_pop
    - User input: population_summary.pkl (data frame of census statistics -> see an example here)
    - loader.generate_basepop() -> population.pkl
- households
	- User input: households_summary.pkl and geo addresses (data frame of census statistics -> see an example)
	- loader.generate_household() -> household.pkl
- mobility
	- User input: mobility_mapping.json
	- loader.generate_mobility_networks() -> mobility_networks/i.pkl

Step 3: Executor preprocesses synthetic population for simulation execution
	- Turn each column of data frame into pickle file and store a mapping
	- Input: loader.preprocess_synthetic_data()
	- Output: a folder that is directly consumed by model config