import numpy as np
import pandas as pd


soc_data_path = '/Users/shashankkumar/Documents/AgentTorch_Official/AgentTorch/AgentTorch/helpers/census_data/nyc/nta/soc_2021acs5yr_nta.xlsx'
dfh = pd.read_excel(soc_data_path)
attributes = ['GeoID','HHPopE','Rshp_ChE','HH1E','RshpHHldrE','Rshp_SpE',"Rshp_ChE",'Rshp_OthRE','Rshp_NRE','Rshp_NRUPE','Rshp_SSUPE','AvgHHSzE']

df = dfh[attributes]
df.rename(columns={'GeoID':'area', 'HHPopE':'people_num', 'Rshp_ChE':'children_num', 'HH1E':'household_num'}, inplace=True)

mapping = {"HHPopE": "Population_in_households",
"RshpHHldrE": "Householder",
"Rshp_SpE": "Spouse",
"Rshp_ChE": "children_num",
"Rshp_OthRE": "Other_relatives",
"Rshp_NRE": "Nonrelatives",
"Rshp_NRUPE": "Unmarried_partner",
"Rshp_SSUPE": "Same_sex_unmarried_partner",
"Fam1E": "Family_households",
"FamChU18E": "With_own_children_under_18_years",
"MrdFamE": "Married_couple_family",
"MrdChU18E": "With_own_children_under_18_years",
"MHnSE": "Male_householder_no spouse_present_family",
"MHnSChU18E": "With_own_children_under_18_years",
"FHnSE": "Female_householder_no_spouse_present_family",
"FHnSChU18E": "With_own_children_under_18_years",
"NFam1E": "Nonfamily_households",
"NFamAE": "Householder_living alone",
"NFamA65plE": "65_years_and_over",
'GeoID':'area', 
'HHPopE':'people_num', 
'HH1E':'household_num',
"AvgHHSzE" : 'Average_household_size',
"AvgFmSzE" : 'Average_family_size'
}

attributes = list(mapping.keys())
df_housing = dfh[attributes]
df_housing.rename(columns=mapping, inplace=True)
df_housing.to_pickle('housing_v2.pkl')

df_g = dfh[['GeoID','Borough']]
df_g['region'] = 'NYC'
df_g['super_area_code'] = df_g['Borough']
df_g['super_area_code'] = df_g['super_area_code'].map({'Manhattan':1, 'Bronx':2, 'Brooklyn':3, 'Queens':4, 'Staten Island':5})
df_g.rename(columns={'GeoID':'area', 'Borough':'super_area_name'}, inplace=True)
df_g.to_pickle("geo.pkl")