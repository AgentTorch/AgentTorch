import numpy as np
from agents import *

# education from social file
social_df = pd.read_excel(social_file)
# age, gender, race from demographic file
demo_df = pd.read_excel(demographic_file)
# employment, insurance from economic file
econ_df = pd.read_excel(economic_file)


def process_nta_agent(NTA_ID):
    global demo_df, econ_df, social_df, age_mapping, race_mapping, education_mapping, employment_insurance_mapping

    nta_race = get_nta_race(demo_df, NTA_ID, race_mapping)
    nta_age_gender = get_nta_age_gender(demo_df, NTA_ID, age_mapping)
    nta_employ_insure = get_nta_employ_insure(
        econ_df, NTA_ID, employment_insurance_mapping
    )
    nta_education = get_nta_education(social_df, NTA_ID, education_mapping)

    num_agents_age = sum([nta_race[key_ix]["estimate"] for key_ix in nta_race])
    num_agents_race = sum(
        [nta_age_gender[key_ix]["estimate"] for key_ix in nta_age_gender]
    )

    assert int(num_agents_age) > 0
    assert int(num_agents_age) == int(num_agents_race)

    race_prob = [nta_race[key_ix]["probability"] for key_ix in nta_race]
    age_gender_prob = [
        nta_age_gender[key_ix]["probability"] for key_ix in nta_age_gender
    ]
    education_prob = [
        nta_education[key_ix]["probability"][0] for key_ix in nta_education
    ]
    insurance_employ_prob = [
        nta_employ_insure[key_ix]["probability"][0] for key_ix in nta_employ_insure
    ]

    nta_dict = {}

    nta_dict["nta_id"] = NTA_ID
    nta_dict["num_agents"] = num_agents_age

    nta_dict["race_prob"] = race_prob
    nta_dict["age_gender_prob"] = age_gender_prob
    nta_dict["education_prob"] = education_prob
    nta_dict["insurance_employ_prob"] = insurance_employ_prob

    return nta_dict


all_nta_ids = demo_df["GeoID"]
print("Total NTAs: ", all_nta_ids.shape[0])

all_nta_dict = {}
nonagent_ntas = []

for indx in range(all_nta_ids.shape[0]):
    NTA_ID = all_nta_ids[indx]

    if indx % 20 == 0:
        print("Done: ", indx)

    try:
        nta_dict = process_nta_agent(NTA_ID)
        all_nta_dict[NTA_ID] = nta_dict
    except:
        print("Assertion failed for nta: ", NTA_ID)
        nonagent_ntas.append(NTA_ID)
        continue

all_nta_dict["nonagent"] = nonagent_ntas

print("Saving NTA dict")
np.save("all_nta_agents.npy", all_nta_dict)
