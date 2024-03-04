import pandas as pd
import numpy as np
import json

def can(df_path = '/tmp/syspop_test/NYC/1/tmp/synpop.pickle'):
    df = pd.read_pickle(df_path)
    attributes = df['synpop'].keys()
    mapping_collection = {}
    for attribute in attributes:
        df['synpop'][attribute],mapping = pd.factorize(df['synpop'][attribute])
        df['synpop'][attribute].to_pickle(f'/tmp/syspop_test/NYC/1/tmp/{attribute}.pickle')
        mapping_collection[attribute] = mapping.tolist()
    with open('mapping.json', 'w') as f:
        json.dump(mapping_collection, f)
    return df

if __name__ == '__main__':
    loadAndSaveAttributes()