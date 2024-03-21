import pandas as pd
import json
import os

def preprocess_data(df_path,output_dir):
    df = pd.read_pickle(df_path)
    attributes = df.keys()
    mapping_collection = {}
    for attribute in attributes:
        df[attribute],mapping = pd.factorize(df[attribute])
        output_att_path = os.path.join(output_dir, attribute)
        df[attribute].to_pickle(f'{output_att_path}.pickle')
        mapping_collection[attribute] = mapping.tolist()
    output_mapping_path = os.path.join(output_dir, 'mapping.json')
    with open(output_mapping_path, 'w') as f:
        json.dump(mapping_collection, f)


if __name__ == '__main__':
    preprocess_data("/Users/shashankkumar/Documents/GitHub/MacroEcon/simulator_data/NZ/output_pop_data/NZ_POP.pkl","/Users/shashankkumar/Documents/GitHub/MacroEcon/simulator_data/NZ")