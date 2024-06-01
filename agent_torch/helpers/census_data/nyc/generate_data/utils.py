import numpy as np
import pandas as pd

demographic_file = "../nta/demo_2021acs5yr_nta.xlsx"
economic_file = "../nta/econ_2021acs5yr_nta.xlsx"
social_file = "../nta/soc_2021acs5yr_nta.xlsx"
housing_file = "../nta/hous_2021acs5yrr_nta.xlsx"


def compute_stats(nta_df, attr, estimation_vars=["E", "M", "C", "P", "Z"], filter=True):
    if filter:
        filter_vars = ["E", "P"]
        estimation_vars = filter_vars
    stats = [nta_df["{}{}".format(attr, var)].values for var in estimation_vars]

    return stats


def merge_nta_stats(nta_df, attr_vals):
    """estimate and percentage"""
    args = [compute_stats(nta_df, attr, filter=True) for attr in attr_vals]
    ret_e, ret_p = 0, 0
    for val in args:
        e, p = val
        ret_e += e
        ret_p += p

    return [ret_e, ret_p]


relationship_file = "../nyc2020census_tract_nta_cdta_relationships.xlsx"
relation_df = pd.read_excel(relationship_file)
map_df = relation_df.filter(items=["NTACode", "CountyFIPS", "BoroName", "NTAType"])

# Metadata 1: nta_metadata - NTA_ID -> [CountyFIPS, CountyName, NTAType]
nta_metadata = dict(
    zip(map_df.NTACode, list(zip(map_df.CountyFIPS, map_df.BoroName, map_df.NTAType)))
)

# Metadata 2: county_nta - CountyFIPS -> [list of NTA_ID]
county_to_nta = {}
for county in map_df.CountyFIPS.unique():
    nta_list = list(map_df[map_df["CountyFIPS"] == county]["NTACode"].unique())

    county_to_nta[county] = nta_list
