import os
import json
import re
import itertools
from typing import List, Dict, Any, Tuple

class PromptManager:
    def __init__(self, user_prompt: str, population: Any):
        self.prompt = user_prompt
        self.population = population
        self.population_folder_path = self.population.population_folder_path
        mapping_path = os.path.join(self.population_folder_path, "mapping.json")
        self.mapping = self.load_mapping(mapping_path)
        self.variables = self.get_variables(self.prompt)
        self.filtered_mapping = self.filter_mapping(self.mapping, self.variables)
        self.combinations_of_prompt_variables = list(itertools.product(*self.filtered_mapping.values()))
        self.combinations_of_prompt_variables_with_index = [
            {k: v for k, v in zip(self.filtered_mapping.keys(), combo)} 
            for combo in self.combinations_of_prompt_variables
        ]
        self.distinct_groups = len(self.combinations_of_prompt_variables)

    @staticmethod
    def load_mapping(path: str) -> Dict[str, List[Any]]:
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def get_variables(prompt_template: str) -> List[str]:
        return re.findall(r"\{(.+?)\}", prompt_template)

    @staticmethod
    def filter_mapping(mapping: Dict[str, List[Any]], variables: List[str]) -> Dict[str, List[Any]]:
        return {k: mapping[k] for k in variables if k in mapping}

    def get_prompt_variables_dict(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: getattr(self.population, key, kwargs.get(key))
            for key in self.variables
        }

    def get_prompt_list(self, kwargs: Dict[str, Any]) -> List[str]:
        dict_variables_with_values = self.get_prompt_variables_dict(kwargs)
        prompt_list = []
        for combination in self.combinations_of_prompt_variables_with_index:
            prompt_values = {**combination, **dict_variables_with_values}
            prompt = self.prompt.format(**prompt_values)
            prompt_list.append(prompt)
        return prompt_list
    
