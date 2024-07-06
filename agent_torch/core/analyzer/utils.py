import pickle
import pandas as pd
from pandasai import Agent


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_state_trace(sim_data_path):
    with open(sim_data_path, 'rb') as handle:
        sim_data_dict = pickle.load(handle)
    agent_prop_df_list = []
    # Loop through each episode in the simulation data dictionary
    for episode in sim_data_dict.keys():
        agent_prop_dict = sim_data_dict[episode]['agents']

        # Loop through each step and agent properties in the episode
        for step, agent_prop in agent_prop_dict.items():
            processed_data = {'consumers': {}}

            # Extract consumer data from the agent properties
            for key, value in agent_prop['consumers'].items():
                value = value.flatten().squeeze()
                processed_data['consumers'][key] = value.numpy()

            population_size = len(processed_data['consumers']['ID'])
            # Limit the 'assets' column to the first population_size entries
            processed_data['consumers']['assets'] = processed_data['consumers']['assets'][:population_size]

            # Create a DataFrame from the processed consumer data
            consumer_df = pd.DataFrame(processed_data['consumers'])

            # Explode (flatten) the nested columns
            consumer_df = consumer_df.explode(['assets', 'consumption_propensity', 'monthly_income', 'post_tax_income', 'will_work'])

            # Ensure consistent data types for the DataFrame columns
            consumer_df = consumer_df.astype({'ID': int, 'age': float, 'area': float, 'assets': float, 'consumption_propensity': float,
                                                'ethnicity': float, 'gender': float, 'monthly_income': float, 'post_tax_income': float,
                                                'will_work': float})

            # Remove unnecessary columns
            consumer_df = consumer_df.drop(['work_propensity', 'monthly_consumption'], axis=1)

            # Add month and year columns based on the current step and episode
            consumer_df['month'] = step
            consumer_df['year'] = episode

            # Mapping for categorical variables
            mapping = {
                "age": ["20t29", "30t39", "40t49", "50t64", "65A", "U19"],
                "gender": ["male", "female"],
                "ethnicity": ["hispanic", "asian", "black", "white", "other", "native"],
                "region": ["Bay of Plenty", "Otago", "Tasman", "Waikato", "Wellington"]
            }

            # Reverse the mapping for replacement
            reverse_mapping = {col: {i: val for i, val in enumerate(vals)} for col, vals in mapping.items()}

            # Replace numerical values with categorical labels in the DataFrame
            consumer_df.replace(reverse_mapping, inplace=True)

            # Append the current consumer DataFrame to the list
            agent_prop_df_list.append(consumer_df)

    return agent_prop_df_list


def get_pandas_agent(agent_prop_df_list, llm):
    # Create and return the PandasAI Agent instance
    return Agent(agent_prop_df_list, config={"llm": llm},description="You are a data analysis agent. Your main goal is to help non-technical users to analyze data")
