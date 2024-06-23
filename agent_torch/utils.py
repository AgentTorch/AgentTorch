import glob
import inspect
import json
import wandb
import types
import os
import pandas as pd
import torch


def get_config_values(conf, keys):
    return {key: conf.get(f"simulation_metadata.{key}") for key in keys}


def add_metadata(conf, params):
    for key, value in params.items():
        conf.add_metadata(key, value)


def set_custom_transition_network_factory(custom_transition_network):
    def set_custom_transition_network(cls):
        class CustomTransition(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.custom_transition_network = custom_transition_network

        return CustomTransition

    return set_custom_transition_network


def set_custom_observation_network_factory(custom_observation_network):
    def set_custom_observation_network(cls):
        class CustomObservation(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.custom_observation_network = custom_observation_network

        return CustomObservation

    return set_custom_observation_network


def set_custom_action_network_factory(custom_action_network):
    def set_custom_action_network(cls):
        class CustomAction(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.custom_action_network = custom_action_network

        return CustomAction

    return set_custom_action_network


def initialise_wandb(entity, project, name, config):
    wandb.init(entity=entity, project=project, name=name, config=config)


def create_dicts_list(params):
    # Find the key with a list value
    list_key = next(
        (key for key, value in params.items() if isinstance(value, list)), None
    )

    # If no key with a list value is found, return the input dictionary as the only element in a list
    if list_key is None:
        return [params]

    # Create a list of dictionaries based on the length of the list value
    list_value = params[list_key]
    dict_list = []
    for i in range(len(list_value)):
        new_dict = params.copy()
        new_dict[list_key] = list_value[i]
        dict_list.append(new_dict)

    return dict_list


def assign_method(runner, method_name, method):
    setattr(runner, method_name, types.MethodType(method, runner))


def is_async_method(cls, method_name):
    method = getattr(cls, method_name)
    return inspect.iscoroutinefunction(method)


def preprocess_data(df_path, output_dir):
    df = pd.read_pickle(df_path)
    attributes = df.keys()
    mapping_collection = {}
    for attribute in attributes:
        df[attribute], mapping = pd.factorize(df[attribute])
        output_att_path = os.path.join(output_dir, attribute)
        df[attribute].to_pickle(f"{output_att_path}.pickle")
        mapping_collection[attribute] = mapping.tolist()
    output_mapping_path = os.path.join(output_dir, "mapping.json")
    with open(output_mapping_path, "w") as f:
        json.dump(mapping_collection, f)
