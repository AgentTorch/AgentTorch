import numpy as np
import pandas as pd
import json
import pdb

from utils import *

race_mapping = {'hispanic': ['Hsp1'], 'white': ['WtNH'], 'black': ['BlNH'], 
                'native': ['NHPINH', 'AIANNH'], 'other': ['OthNH', 'Rc2plNH'], 'asian': ['AsnNH']}

age_mapping = {'U19': ['PopU5', 'Pop5t9', 'Pop10t14', 'Pop15t19'], '20t29': ['Pop20t24', 'Pop25t29'],  
                '30t39': ['Pop30t34', 'Pop35t39'], '40t49': ['Pop40t44', 'Pop45t49'], 
                '50t64': ['Pop50t54', 'Pop55t59', 'Pop60t64'], '65A': ['Pop65t69', 'Pop70t74', 'Pop75t79','Pop80t84', 'Pop85pl']}

education_mapping = {'high_school': ['EA_LT9G', 'EA_9t12ND', 'EA_HScGrd'],
                    'college_degree': ['EA_SClgND', 'EA_AscD', 'EA_BchD'],
                    'graduate_degree': ['EA_GrdPfD'],
                    'studying': ['Pop3plEn']}

employment_insurance_mapping = {'employed_insured': 'EmHIns', 'employed_uninsured': 'EmNHIns',
                                'unemployed_insured': 'UEmHIns', 'unemployed_uninsured': 'UEmNHIns',
                                'nolabor_insured': 'NLFHIns', 'nolabor_uninsured': 'NLFNHIns'}


def get_gender_split(total_estimate, total_percentage, male_ratio):
    total_estimate = total_estimate.item()
    total_percentage = total_percentage.item()

    male_estimate = int(max(0, male_ratio*total_estimate))
    female_estimate = total_estimate - male_estimate

    male_percentage = int(max(0, male_ratio*total_percentage))
    female_percentage = (total_percentage - male_percentage)

    male_probability, female_probability = male_percentage / 100, female_percentage / 100

    return_dict = {'male': [male_estimate, male_probability], 'female': [female_estimate, female_probability]}

    return return_dict

# Property 1: Race
def get_nta_race(df, nta_id, race_mapping):
    nta_df = df[df['GeoID'] == nta_id]

    nta_race = {}
    for key in race_mapping:
        nta_race[key] = {}

        estimate, percentage = merge_nta_stats(nta_df, race_mapping[key])

        nta_race[key]['estimate'] = max(0, estimate.item())
        nta_race[key]['probability'] = max(0, percentage.item())/100

    return nta_race


# Property 2,3: Age and Gender

def get_nta_age_gender(df, nta_id, age_mapping, male_ratio=0.508):
    '''estimate, percentage'''
    nta_df = df[df['GeoID'] == nta_id]

    nta_age_gender = {}
    for key in age_mapping:
        attr_vals = age_mapping[key]

        total_estimate, total_percentage = merge_nta_stats(nta_df, attr_vals)
        if key == 'U19':
            male_ratio = compute_stats(nta_df, 'PopU18M')[-1] / 100.0 # percentage of male < 19
        if key == '65A':
            male_ratio = compute_stats(nta_df, 'Pop65plM')[-1] / 100.0 # percentage of male > 65
        
        gender_split_stats = get_gender_split(total_estimate, total_percentage, male_ratio=male_ratio)

        male_stats = gender_split_stats['male']
        female_stats = gender_split_stats['female']

        key_male = key + '_male'
        key_female = key + '_female'

        nta_age_gender[key_male] = {'estimate': male_stats[0], 'probability': male_stats[1]}
        nta_age_gender[key_female] = {'estimate': female_stats[0], 'probability': female_stats[1]}

        # nta_age_gender[key] = age_gender_stats

    return nta_age_gender


# Property 4: Education Status

def get_nta_education(df, nta_id, education_mapping):
    '''Education status can be completed (>25) or studying (for <=25)'''
    nta_df = df[df['GeoID'] == nta_id]
    nta_education = {} # for agents 25 and above

    studying = compute_stats(nta_df, 'Pop3plEn')[0]
    nolonger_studying = compute_stats(nta_df, 'EA_P25pl')[0]
    total_educated_studying = studying + nolonger_studying
    total_educated_studying = total_educated_studying.item()

    for key in education_mapping: # education_mapping is for >25 yr old agents
        nta_education[key] = {}
        estimate, _ = merge_nta_stats(nta_df, education_mapping[key])
        estimate = estimate.item()

        nta_education[key]['estimate'] = max(0, estimate)
        nta_education[key]['probability'] = nta_education[key]['estimate'] / total_educated_studying
        
    return nta_education


# Property 5,6: Employment and Insurance status

def get_nta_employ_insure(df, nta_id, employ_insure_mapping):
    nta_df = df[df['GeoID']==nta_id]

    total_eligible_agents = compute_stats(nta_df, 'CNI1864_2')[0]
    total_eligible_agents = total_eligible_agents.item()

    nta_employ_insure = {}
    for category in employ_insure_mapping:
        estimate, _ = compute_stats(nta_df, employ_insure_mapping[category])
        estimate = estimate.item()

        nta_employ_insure[category] = {}
        nta_employ_insure[category]['estimate'] = estimate
        nta_employ_insure[category]['probability'] = estimate / total_eligible_agents

    return nta_employ_insure

