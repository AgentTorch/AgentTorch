from agent_torch.data.census.census_loader import CensusDataLoader
import torch.nn as nn
import os
import pandas as pd
import json
import torch
import numpy as np

# import ray
# ray.init(address='auto')

AGE_GROUP_MAPPING = {
    "adult_list": ["20t29", "30t39", "40t49", "50t64", "65A"],  # Age ranges for adults
    "children_list": ["U19"],  # Age range for children
}

MOBILITY_MAPPING = json.load(open('mobility_mapping.json'))

def _initialize_infections(num_agents, save_dir=None, initial_infection_ratio=0.04):
    # figure out initial infection ratio    
    correct_num_cases = False
    num_tries = 0

    num_cases = int(initial_infection_ratio*num_agents)

    if save_dir is None:
        save_path = './disease_stages_{}.csv'.format(num_agents)
    else:
        save_path = os.path.join(save_dir, 'disease_stages.csv')
        
    while not correct_num_cases:
        prob_infected = initial_infection_ratio * torch.ones(
            (num_agents, 1)
        )
        p = torch.hstack((prob_infected, 1 - prob_infected))
        cat_logits = torch.log(p + 1e-9)
        agent_stages = nn.functional.gumbel_softmax(
            logits=cat_logits, tau=1, hard=True, dim=1
        )[:, 0]
        tensor_np = agent_stages.numpy().astype(int)
        tensor_np = np.array(tensor_np, dtype=np.uint8)
        np.savetxt("temp_infections", tensor_np, delimiter='\n')

        # check if the generated file has the correct number of cases
        arr = np.loadtxt(f"temp_infections")
        if arr.sum() == num_cases:
            correct_num_cases = True
            # write the correct file with the line that says stages
            with open(
                save_path,
                "w",
            ) as f:
                f.write("stages\n")
                np.savetxt(f, tensor_np)

        num_tries += 1
        if num_tries >= 1000:
            raise Exception("failed to create disease stages file")

def customize(sample_dir, num_agents=None, region="sample", area_selector=None, use_household=False):
    population_data_path = os.path.join(sample_dir, 'population.pkl')
    household_data_path = os.path.join(sample_dir, 'household.pkl')

    # Load household data
    HOUSEHOLD_DATA = pd.read_pickle(household_data_path)

    # Load population data
    BASE_POPULATION_DATA = pd.read_pickle(population_data_path)

    census_data_loader = CensusDataLoader(n_cpu=8, use_parallel=False)
    print("Will save at: ", census_data_loader.population_dir)

    if num_agents is not None:
        census_data_loader.generate_basepop(
            input_data=BASE_POPULATION_DATA,  # The population data frame
            region=region,  # The target region for generating base population
            area_selector=area_selector,  # Area selection criteria, if applicable
            num_individuals = num_agents # Saves data for first 100 individuals, from the generated population
        )
    else:
        census_data_loader.generate_basepop(
            input_data=BASE_POPULATION_DATA,  # The population data frame
            region=region,  # The target region for generating base population
            area_selector=area_selector  # Area selection criteria, if applicable
        )

    if use_household:
        census_data_loader.generate_household(
            household_data=HOUSEHOLD_DATA,  # The loaded household data
            household_mapping=AGE_GROUP_MAPPING,  # Mapping of age groups for household composition
            region=region  # The target region for generating households
        )

    census_data_loader.generate_mobility_networks(
        num_steps=2, 
        mobility_mapping=MOBILITY_MAPPING, 
        region=region
    )

    census_data_loader.export(region)

    return census_data_loader.population_dir


if __name__ == '__main__':
    sample_dir = os.path.join(os.getcwd(), 'docs/tutorials/processing-a-population/sample_data/NYC')
    num_agents = 1000
    region = 'sample'
    area_selector = ['BK0101']
    print("Customizing population")
    pop_save_dir = customize(sample_dir, num_agents=num_agents, region=region, area_selector=area_selector)

    initial_infection_ratio = 0.04
    print("Initializing infections")
    save_dir = os.path.join(pop_save_dir, region)
    _initialize_infections(num_agents, save_dir=save_dir, initial_infection_ratio=initial_infection_ratio)