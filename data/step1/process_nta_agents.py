import pickle
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
    df_ethnicity = pd.DataFrame(nta_race, index=[0])
    df_ethnicity['area'] = NTA_ID
    
    nta_age_gender = get_nta_age_gender(demo_df, NTA_ID, age_mapping)
    df_age_gender = pd.DataFrame.from_dict(nta_age_gender, orient='index')
    df_age_gender.reset_index(inplace=True)
    df_age_gender.rename(columns={'index':'age'}, inplace=True)
    df_age_gender['area'] = NTA_ID
    df_age_gender = df_age_gender.melt(id_vars=['age', 'area'], var_name='gender')
    df_age_gender = df_age_gender.pivot(index=['area', 'gender'], columns='age', values='value')
    df_age_gender.reset_index(inplace=True)

    nta_employment_insurance = get_nta_employ_insure(econ_df, NTA_ID, employment_insurance_mapping)
    df_employment_insurance = pd.DataFrame(nta_employment_insurance, index=[0])
    df_employment_insurance['area'] = NTA_ID
    
    nta_education_level = get_nta_education(social_df, NTA_ID, education_mapping)
    df_education = pd.DataFrame(nta_education_level, index=[0])
    df_education['area'] = NTA_ID    


    return df_age_gender,df_ethnicity,df_education,df_employment_insurance

def get_feature_mapping():
    global age_mapping, race_mapping, education_mapping, employment_insurance_mapping
    
    feature_maps = dict()
    
    feature_maps['race'] = list(race_mapping.keys())
    feature_maps['education'] = list(education_mapping.keys())
    feature_maps['employment_insurance'] = list(employment_insurance_mapping.keys())
    feature_maps['age'] = list(age_mapping.keys())
    feature_maps['gender'] = ['Male','Female']
    
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
DATA_MODE = 'estimate' # probability

df_age_gender = pd.DataFrame()
df_ethnicity = pd.DataFrame()
df_education = pd.DataFrame()
df_employment_insurance = pd.DataFrame()
for nta_indx in range(all_nta_ids.shape[0]):
    NTA_ID = all_nta_ids[nta_indx]
    print("NTA_ID: ", NTA_ID)
    
    try:
        nta_dict = process_nta_agents(NTA_ID,data_mode=DATA_MODE)
        df_age_gender = pd.concat([df_age_gender,nta_dict[0]],ignore_index=True)
        df_ethnicity = pd.concat([df_ethnicity,nta_dict[1]],ignore_index=True)
        df_education = pd.concat([df_education,nta_dict[2]],ignore_index=True)
        df_employment_insurance = pd.concat([df_employment_insurance,nta_dict[3]],ignore_index=True)
        # all_nta_dict[NTA_ID] = nta_dict
    except:
        print("Assertion failed for nta: ", NTA_ID)
        nonagent_ntas.append(NTA_ID)
        continue

print("Empty NTAs : ", len(nonagent_ntas))
output_dict = {
    'age_gender' : df_age_gender,
    'ethnicity' : df_ethnicity,
    'education' : df_education,
    'employment_insurance' : df_employment_insurance
}
with open('output_dict.pkl', 'wb') as f:
    pickle.dump(output_dict, f)

# Save data_dict and mapping_dict
save_data['empty_ntas'] = nonagent_ntas
save_data['mapping'] = mapping_dict
save_data['valid_ntas'] = all_nta_dict

np.save("all_nta_agents.npy", save_data)
