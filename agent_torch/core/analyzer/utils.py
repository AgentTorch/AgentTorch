import pickle
import pandas as pd
from pandasai import Agent
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.prompts import ChatPromptTemplate
import uuid
from langgraph.checkpoint.base import (
    BaseCheckpointSaver
)

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_state_trace(sim_data_path,description):
    df_name = sentence_to_snake_case(description)
    locals()[df_name] = []
    with open(sim_data_path, 'rb') as handle:
        sim_data_dict = pickle.load(handle)
    # Loop through each episode in the simulation data dictionary
    sim_data_dict = sim_data_dict[next(reversed(sim_data_dict))]
    agent_prop_dict = sim_data_dict['agents']

    # Loop through each step and agent properties in the episode
    for step, agent_prop in agent_prop_dict.items():
        processed_data = {'consumers': {}}

        # Extract consumer data from the agent properties
        for key, value in agent_prop['consumers'].items():
            value = value.flatten().squeeze()
            processed_data['consumers'][key] = value.detach().numpy()

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
        # consumer_df['year'] = episode

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
        locals()[df_name].append(consumer_df)

    return locals()[df_name], df_name


def get_pandas_agent(agent_prop_df_list, llm):
    # Create and return the PandasAI Agent instance
    return Agent(agent_prop_df_list, config={"llm": llm},description="You are a data analysis agent. Your main goal is to help non-technical users to analyze data. You have access to state data of a simulation that user ran. Give well explained and reasoned results. Don't display any image, give outputs as only string.",memory_size=3)

def sentence_to_snake_case(sentence):
    # Split the sentence into words, strip any leading/trailing whitespace from each word
    words = [word.strip() for word in sentence.split()]
    # Combine words using '_' as the separator
    snake_case_sentence = '_'.join(words)
    return snake_case_sentence

def empty_checkpoint():
    return {
        "id": str(uuid.uuid4()), 
        "question": "",
        "generation": "",
        "documents": [],
        "retries_limit": 0,
        "messages": [],
        "datasource": ""
    }
    
def clear_memory(memory: BaseCheckpointSaver, thread_id: str) -> None:
    checkpoint = empty_checkpoint()
    memory.put(config={"configurable": {"thread_id": thread_id}}, checkpoint=checkpoint, metadata={})
    


def generate_description(field_name,llm):
    system = f"Generate a succinct  description for the field named '{field_name}', in under 15 words:"
    # Prompt
    description_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{field_name}"),
        ]
    )

    description_generator = description_prompt | llm
    description = description_generator.invoke({"field_name": field_name})
    return description.content


def generate_attribute_info_list(data,llm):
    attribute_info_list = []
    filtered_data = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
    for key, value in filtered_data.items():
        if isinstance(value, int):
            value_type = "integer"
        elif isinstance(value, float):
            value_type = "float"
        elif isinstance(value, bool):
            value_type = "boolean"
        else:
            value_type = "string"

        description = generate_description(key,llm)

        attribute_info = AttributeInfo(
            name=key, description=description, type=value_type
        )
        attribute_info_list.append(attribute_info)

    return attribute_info_list

def initialize_agents_dict(state_trace, llm):
    """
    Initialize a dictionary of agents based on the keys in state_trace.

    Parameters:
    state_trace (dict): The state trace dictionary containing agent properties.
    llm: The language model to be used for creating agents.

    Returns: 
    dict: A dictionary of agents.
    """
    # Retrieve the keys from state_trace
    state_trace_keys = state_trace.keys()

    # Initialize a dictionary of agents
    agents_dict = {}

    # Loop through each key and create an agent using get_pandas_agent
    for key in state_trace_keys:
        agents_dict[key] = get_pandas_agent(
            agent_prop_df_list=state_trace[key],
            llm=llm,
        )

    return agents_dict