import pandas as pd
import numpy as np
import json

def can(df_path = '/tmp/syspop_test/NYC/1/tmp/synpop.pickle'):
    df = pd.read_pickle(df_path)
    attributes = df.keys()
    mapping_collection = {}
    for attribute in attributes:
        df[attribute],mapping = pd.factorize(df[attribute])
        df[attribute].to_pickle(f'/Users/shashankkumar/Documents/GitHub/MacroEcon/{attribute}.pickle')
        mapping_collection[attribute] = mapping.tolist()
    with open('mapping.json', 'w') as f:
        json.dump(mapping_collection, f)


if __name__ == '__main__':
    can("/Users/shashankkumar/Documents/GitHub/MacroEcon/base_population.pkl")