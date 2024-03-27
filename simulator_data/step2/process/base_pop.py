'''
Source: https://github.com/sneakatyou/Syspop/tree/NYC/syspop/process
'''

from datetime import datetime
from genericpath import exists
from os import makedirs
from pickle import dump as pickle_dump

import ray
from numpy.random import choice
from pandas import DataFrame
import numpy as np
import pandas as pd
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def get_probability(attribute : list):
    return [i/sum(attribute) for i in attribute]

def get_index(value, age_ranges):
    try:
        return age_ranges.index(value)
    except ValueError:
        return "Value not found in the list"

@ray.remote
def create_base_pop_remote(df_age_gender,df_ethnicity,age,area):
    return create_base_pop(df_age_gender,df_ethnicity,age,area)


def create_base_pop(df_age_gender,df_ethnicity,age,area):
    population = []
    # number_of_individuals = area_data[number_of_individuals]
    age_gender_data = df_age_gender[df_age_gender['area']==area]
    age_gender_data = age_gender_data[age_gender_data['age']==age]
    ethnicity_data = df_ethnicity[df_ethnicity['area']==area]
    
    if 'population_count' in age_gender_data.columns:
        number_of_individuals = int(age_gender_data['population_count'].values[0])
    else:
        number_of_individuals = int(age_gender_data.sum(axis=0)[['count']].sum())
    if number_of_individuals == 0:
        return []
    
    # gender_prob = area_data['age_gender_prob'][age]
    # Randomly assign gender and ethnicity to each individual
    age_gender_data['probability'] = age_gender_data.apply(lambda row: row['count'] / number_of_individuals, axis=1)
    gender_choices = age_gender_data['gender']
    gender_probablities = age_gender_data['probability']
    gender_probablities = gender_probablities/sum(gender_probablities)
    genders = choice(gender_choices, size=number_of_individuals, p=gender_probablities)

    total_population = ethnicity_data.groupby('area')['count'].sum()
    ethnicity_data['probability'] = ethnicity_data.apply(lambda row: row['count'] / total_population[row['area']], axis=1)
    ethnicity_choices = ethnicity_data['ethnicity']
    ethnicity_probabilities = ethnicity_data['probability']
    ethnicity_probabilities = ethnicity_probabilities/sum(ethnicity_probabilities)
    ethnicities = choice(
        ethnicity_choices,
        size=number_of_individuals,
        p=ethnicity_probabilities,
    )
    
    # education_prob = get_probability(area_data['education'])
    # education = choice(
    #     input_mapping['education'],
    #     size=number_of_individuals,
    #     p=education_prob,
    # )
    
    # employment_insurance_prob = get_probability(area_data['employment_insurance'])
    # employment_insurance = choice(
    #     input_mapping['employment_insurance'],
    #     size=number_of_individuals,
    #     p=employment_insurance_prob,
    # )

    for gender, ethnicity in zip(genders, ethnicities):
        individual = {
            "area": area,
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity
        }
        population.append(individual)
    # for gender, ethnicity,education,employment_insurance in zip(genders, ethnicities,education,employment_insurance):
    #     individual = {
    #         "area": area,
    #         "age": age,
    #         "gender": gender,
    #         "ethnicity": ethnicity,
    #         "education": education,
    #         "employment_insurance": employment_insurance
    #     }
    #     population.append(individual)

    return population


def base_pop_wrapper(
    input_data,
    area_selector=None,
    use_parallel=False,
    n_cpu=8,
) -> DataFrame:
    
    df_age_gender = input_data['age_gender']
    df_ethnicity = input_data['ethnicity']
    # df_age_gender_melt = df_age_gender.melt(
    #     id_vars=["area", "gender"], var_name="age", value_name="count"
    # )

    
    start_time = datetime.utcnow()

    if use_parallel:
        ray.init(num_cpus=n_cpu, include_dashboard=False)

    results = []
    if area_selector is None:
        output_areas = df_age_gender['area'].unique()
    else:
        output_areas = area_selector
    total_output_area = len(output_areas)
    age_list = df_age_gender['age'].unique()
    for i, output_area in enumerate(output_areas):
        
        logger.info(f"Processing: {i}/{total_output_area}")
        total_individuals = 0
        for age in age_list:
            if use_parallel:
                result = create_base_pop_remote.remote(
                    df_age_gender=df_age_gender,df_ethnicity=df_ethnicity,age=age,area=output_area
                )
            else:
                result = create_base_pop(
                    df_age_gender=df_age_gender,df_ethnicity=df_ethnicity,age=age,area=output_area
                )
            # total_individuals+=num_individuals
            results.append(result)
        # assert total_individuals == input_data[output_area]['num_agents']

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
    # area_selector = ['BK0101']
    # area_selector = [100100]
    area_selector = [10101]
    output_dir = "/Users/shashankkumar/Documents/GitHub/MacroEcon/simulator_data/census_populations/SF/synthetic_populations"
    if not exists(output_dir):
        makedirs(output_dir)
    
    pop_path = "/Users/shashankkumar/Documents/GitHub/MacroEcon/simulator_data/census_populations/SF/user_input/SF_POP.pkl"
    # df = pd.read_pickle("/Users/shashankkumar/Documents/GitHub/Syspop/syspop/att_dict.pkl")
    df = pd.read_pickle(pop_path)

    base_population,base_address = base_pop_wrapper(input_data=df)
    base_population.to_pickle(output_dir + "/base_population.pkl")
    
    
