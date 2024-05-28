import numpy as np
import pandas as pd

from households import *

demo_df = pd.read_excel(demographic_file)
# occupants from social file
social_df = pd.read_excel(social_file)
# income from economic file
econ_df = pd.read_excel(economic_file)
# ownership, mortgage/rent expense from household file
house_df = pd.read_excel(housing_file)

def process_nta_household(NTA_ID):
    global econ_df, social_df, income_mapping, ownership_mapping

    nta_household_income = get_nta_household_income(econ_df, NTA_ID, income_mapping)
    nta_household_ownership = get_nta_household_ownership(house_df, NTA_ID, ownership_mapping)

    num_households_income = sum([nta_household_income[key_ix]['estimate'] for key_ix in nta_household_income])
    num_households_onwership = sum([nta_household_ownership[key_ix]['estimate'] for key_ix in nta_household_ownership])

    assert num_households_income > 0
    assert num_households_income == num_households_onwership

    income_prob =  [nta_household_income[key_ix]['probability'] for key_ix in nta_household_income]
    ownership_prob = [nta_household_ownership[key_ix]['probability'] for key_ix in nta_household_ownership]

    nta_dict = {}

    nta_dict['nta_id'] = NTA_ID
    nta_dict['num_households'] = num_households_income

    nta_dict['income_prob'] = income_prob
    nta_dict['ownership_prob'] = ownership_prob

    return nta_dict

all_nta_ids = demo_df['GeoID']
print("Total NTAs: ", all_nta_ids.shape[0])

all_nta_dict = {}
unhoused_ntas = []

for indx in range(all_nta_ids.shape[0]):
    NTA_ID = all_nta_ids[indx]

    if indx % 20 == 0:
        print("Done: ", indx)
    try:
        nta_dict = process_nta_household(NTA_ID)
        all_nta_dict[NTA_ID] = nta_dict
    except AssertionError:
        print("Assertion failed for nta: ", NTA_ID)
        unhoused_ntas.append(NTA_ID)
        continue

all_nta_dict['unhoused'] = unhoused_ntas 

print("Saving NTA dict")
np.save("all_nta_households.npy", all_nta_dict)