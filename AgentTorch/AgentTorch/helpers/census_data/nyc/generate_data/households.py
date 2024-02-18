import pandas as pd
import numpy as np

from utils import *

income_mapping_legacy = {'U25': ['HHIU10', 'HHI10t14', 'HHI15t24'], '25t49': ['HHI25t34', 'HHI35t49'], 
                        '50t99': ['HHI50t74', 'HHI75t99'], '100t199': ['HI100t149', 'HI150t199'], '200A': ['HHI200pl']}

income_mapping = {'U15': ['HHIU10', 'HHI10t14'], '15t35': ['HHI15t24', 'HHI25t34'], '35t49': ['HHI35t49'], '50t74': ['HHI50t74'], '75t99': ['HHI75t99'], '100t149': ['HI100t149'], '150t199': ['HI150t199'], '200A': ['HHI200pl']}

income_to_expense_mapping = {'U15':['02'], '15t35': ['03', '04'], '35t49': ['05'], '50t74': ['06', '07'], '75t99': ['08'], '100t149': ['09'], '150t199': ['10'], '200A': ['11']}
income_to_assets_mapping = {'U15': ['A1'], '15t35': ['A1'], '35t49': ['A2'], '50t74': ['A3'], '75t99': ['A3'], '100t149': ['A4'], '150t199': ['A5'], '2OOA': ['A5']}

ownership_mapping = {'renter': ['ROcHU1'], 'no_mortgage_owner': ['HUnoMrtg1'], 'mortgage_owner': ['HUwMrtg']}

# Property 1: household income
def get_nta_household_income(df, nta_id, income_mapping):
    nta_df = df[df['GeoID'] == nta_id]
    
    nta_household_income = {}

    for band in income_mapping:
        nta_household_income[band] = {}
        estimate, percentage = merge_nta_stats(nta_df, income_mapping[band])

        nta_household_income[band]['estimate'] = estimate
        nta_household_income[band]['probability'] = percentage / 100.0

    return nta_household_income

# Property 2: household ownership and mortgage status
def get_nta_household_ownership(df, nta_id, ownership_mapping):
    nta_df = df[df['GeoID'] == nta_id]
    
    nta_household_ownership = {}

    # compute estimates
    for ownership_type in ownership_mapping:
        nta_household_ownership[ownership_type] = {}
        estimate, percentage = merge_nta_stats(nta_df, ownership_mapping[ownership_type])

        nta_household_ownership[ownership_type]['estimate'] = estimate

    all_households = sum([nta_household_ownership[o_type]['estimate'] for o_type in ownership_mapping])
    
    # compute probability
    for ownership_type in ownership_mapping:
        nta_household_ownership[ownership_type]['probability'] = nta_household_ownership[ownership_type]['estimate'] / all_households

    return nta_household_ownership