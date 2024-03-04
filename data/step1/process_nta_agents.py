import numpy as np
import pdb
import pandas as pd
from agents import *

# education from social file
social_df = pd.read_excel(social_file)
# age, gender, race from demographic file
demo_df = pd.read_excel(demographic_file)
# employment, insurance from economic file
econ_df = pd.read_excel(economic_file)

def process_nta_agents(NTA_ID, data_mode='estimate'):
    '''
    Args:
        NTA_ID - indicates the neighborhood tabulation area
        data_mode - estimate or probability. Returns raw quantities or probabilities
    Return:
        nta_dict
    '''
    global demo_df, econ_df, social_df, age_mapping, race_mapping, education_mapping, employment_insurance_mapping
    
    nta_race = get_nta_race(demo_df, NTA_ID, race_mapping)
    nta_age_gender = get_nta_age_gender(demo_df, NTA_ID, age_mapping)
    nta_employment_insurance = get_nta_employ_insure(econ_df, NTA_ID, employment_insurance_mapping)
    nta_education_level = get_nta_education(social_df, NTA_ID, education_mapping)
        
    num_agents_age_data = sum([nta_age_gender[key_ix]['estimate'] for key_ix in nta_age_gender])
    num_agents_race_data = sum([nta_race[key_ix]['estimate'] for key_ix in nta_race])
    
    assert int(num_agents_age_data) > 0 # actual agents live in the NTA
    assert int(num_agents_age_data) == int(num_agents_race_data)
    
    # return dictionary
    nta_dict = dict()
    nta_dict['nta_id'] = NTA_ID
    nta_dict['num_agents'] = num_agents_age_data
    
    race_stats = [nta_race[key_ix][data_mode] for key_ix in nta_race]
    education_stats = [nta_education_level[key_ix][data_mode] for key_ix in nta_education_level]
    employment_insurance_stats = [nta_employment_insurance[key_ix][data_mode] for key_ix in nta_employment_insurance]
    age_gender_stats = [nta_age_gender[key_ix][data_mode] for key_ix in nta_age_gender]

    nta_dict['race'] = race_stats
    nta_dict['education'] = education_stats
    nta_dict['employment_insurance'] = employment_insurance_stats
    nta_dict['age_gender'] = age_gender_stats
    
    return nta_dict

def get_feature_mapping():
    global age_mapping, race_mapping, education_mapping, employment_insurance_mapping
    
    feature_maps = dict()
    
    feature_maps['race'] = list(race_mapping.keys())
    feature_maps['education'] = list(education_mapping.keys())
    feature_maps['employment_insurance'] = list(employment_insurance_mapping.keys())
    
    feature_maps['age_gender'] = []
    for key in age_mapping.keys():
        feature_maps['age_gender'].append(key + '_M')
        feature_maps['age_gender'].append(key + '_F')
    
    return feature_maps
        
all_nta_ids = demo_df['GeoID']
print("Total NTAs: ", all_nta_ids.shape[0])

# Step 1 - generate maping dict
mapping_dict = get_feature_mapping()

# Step 2 - generate data dict
save_data = dict()

all_nta_dict = dict()
nonagent_ntas = []
DATA_MODE = 'estimate'

for nta_indx in range(all_nta_ids.shape[0]):
    NTA_ID = all_nta_ids[nta_indx]
    print("NTA_ID: ", NTA_ID)
    
    try:
        nta_dict = process_nta_agents(NTA_ID, data_mode=DATA_MODE)
        all_nta_dict[NTA_ID] = nta_dict
    except:
        print("Assertion failed for nta: ", NTA_ID)
        nonagent_ntas.append(NTA_ID)
        continue

print("Empty NTAs : ", len(nonagent_ntas))

# Save data_dict and mapping_dict
save_data['empty_ntas'] = nonagent_ntas
save_data['mapping'] = mapping_dict
save_data['valid_ntas'] = all_nta_dict

np.save("all_nta_agents.npy", save_data)
