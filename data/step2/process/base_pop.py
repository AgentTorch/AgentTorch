'''
Source: https://github.com/sneakatyou/Syspop/tree/NYC/syspop/process
'''

from datetime import datetime
from genericpath import exists
from logging import getLogger
from os import makedirs
from pickle import dump as pickle_dump

import ray
from numpy.random import choice
from pandas import DataFrame
import numpy as np
import pandas as pd
import random
logger = getLogger()

def get_probability(attribute : list):
    return [i/sum(attribute) for i in attribute]

def get_index(value, age_ranges):
    try:
        return age_ranges.index(value)
    except ValueError:
        return "Value not found in the list"

@ray.remote
def create_base_pop_remote(area_data,input_mapping, output_area,age):
    return create_base_pop(area_data,input_mapping, output_area,age)


def create_base_pop(area_data,input_mapping, output_area,current_age):
    population = []
    # number_of_individuals = area_data[number_of_individuals]
    
    age_index = input_mapping['age'].index(current_age)
    number_of_individuals = area_data['age_gender'][age_index] + area_data['age_gender'][age_index+1]
    if number_of_individuals == 0:
        return []
    
    # gender_prob = area_data['age_gender_prob'][age]
    # Randomly assign gender and ethnicity to each individual

    male_prob = area_data['age_gender'][age_index]
    female_prob = area_data['age_gender'][age_index+1] #TODO: Change it to be generalised
    combined_gender_prob = [male_prob,female_prob]
    gender_prob = get_probability(combined_gender_prob)
    
    genders = choice(input_mapping['gender'], size=number_of_individuals, p=gender_prob)

    ethnicity_prob = get_probability(area_data['race'])
    ethnicities = choice(
        input_mapping['race'],
        size=number_of_individuals,
        p=ethnicity_prob,
    )
    
    education_prob = get_probability(area_data['education'])
    education = choice(
        input_mapping['education'],
        size=number_of_individuals,
        p=education_prob,
    )
    
    employment_insurance_prob = get_probability(area_data['employment_insurance'])
    employment_insurance = choice(
        input_mapping['employment_insurance'],
        size=number_of_individuals,
        p=employment_insurance_prob,
    )
    attributes_list = [genders, ethnicities,education,employment_insurance]
    for gender, ethnicity,education,employment_insurance in zip(genders, ethnicities,education,employment_insurance):
        individual = {
            "area": output_area,
            "age": current_age,
            "gender": gender,
            "ethnicity": ethnicity,
            "education": education,
            "employment_insurance": employment_insurance
        }
        population.append(individual)

    return population


def base_pop_wrapper(
    input_data: dict,
    input_mapping: dict,
    use_parallel: bool = False,
    n_cpu: int = 8,
) -> DataFrame:
    """Create base population

    Args:
        gender_data (DataFrame): Gender data for each age
        ethnicity_data (DataFrame): Ethnicity data for each age
        output_area_filter (list or None): With area ID to be used
        use_parallel (bool, optional): If apply ray parallel processing. Defaults to False.

    Returns:
        DataFrame: Produced base population
    """
    start_time = datetime.utcnow()

    if use_parallel:
        ray.init(num_cpus=n_cpu, include_dashboard=False)

    results = []

    output_areas = list(input_data.keys())
    total_output_area = len(output_areas)
    for i, output_area in enumerate(output_areas):
        logger.info(f"Processing: {i}/{total_output_area}")
        for age in input_mapping['age']:
            age_index = input_mapping['age'].index(age)
            if use_parallel:
                result = create_base_pop_remote.remote(
                    area_data=input_data[output_area], output_area=output_area,age=age,input_mapping=input_mapping
                )
            else:
                result = create_base_pop(
                    area_data=input_data[output_area], output_area=output_area,age=age,input_mapping=input_mapping
                )
            results.append(result)

    if use_parallel:
        results = ray.get(results)
        ray.shutdown()

    population = [item for sublist in results for item in sublist]

    end_time = datetime.utcnow()
    total_mins = (end_time - start_time).total_seconds() / 60.0

    # create an empty address dataset
    base_address = DataFrame(columns=["type", "name", "latitude", "longitude"])

    logger.info(f"Processing time (base population): {total_mins}")

    # Convert the population to a DataFrame
    return DataFrame(population), base_address

if __name__ == "__main__":
    
    output_dir = "/tmp/syspop_test/NYC/1"
    if not exists(output_dir):
        makedirs(output_dir)
    file = np.load("/Users/shashankkumar/Documents/GitHub/MacroEcon/all_nta_agents.npy", allow_pickle=True)
    file_dict = file.item()
    
    base_population,base_address = base_pop_wrapper(input_data=file_dict['valid_ntas'],input_mapping=file_dict['mapping'],use_parallel=True,n_cpu=8)
    base_population.to_pickle(output_dir + "/base_population.pkl")
    
    
