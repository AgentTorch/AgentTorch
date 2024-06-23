import os
import json
import re
import itertools


class PromptManager:
    def __init__(self, user_prompt, population):
        self.prompt = user_prompt
        self.population = population
        self.population_folder_path = self.population.population_folder_path
        mapping_path = os.path.join(self.population_folder_path, "mapping.json")
        self.mapping = self.load_mapping(mapping_path)
        self.variables = self.get_variables(self.prompt)
        self.filtered_mapping = self.filter_mapping(self.mapping, self.variables)
        (
            self.combinations_of_prompt_variables,
            self.combinations_of_prompt_variables_with_index,
        ) = self.get_combinations_of_prompt_variables(self.filtered_mapping)

    def load_mapping(self, path):
        with open(path, "r") as f:
            mapping = json.load(f)
        return mapping

    def get_variables(self, prompt_template):
        variables = re.findall(r"\{(.+?)\}", prompt_template)
        return variables

    def filter_mapping(self, mapping, variables):
        return {k: mapping[k] for k in variables if k in mapping}

    def get_combinations_of_prompt_variables(self, mapping):
        combinations = list(itertools.product(*mapping.values()))
        dict_combinations = [
            dict(zip(mapping.keys(), combination)) for combination in combinations
        ]
        index_combinations = [
            {k: mapping[k].index(v) for k, v in combination.items()}
            for combination in dict_combinations
        ]
        return dict_combinations, index_combinations

    def get_prompt_variables_dict(self, kwargs):
        variables = {}
        for key in self.variables:
            variables[key] = (
                getattr(self.population, key)
                if hasattr(self.population, key)
                else kwargs.get(key)
            )
        return variables

    def get_prompt_list(self, kwargs):
        self.dict_variables_with_values = self.get_prompt_variables_dict(kwargs=kwargs)
        prompt_list = []
        for en, _ in enumerate(self.combinations_of_prompt_variables_with_index):
            prompt_values = self.combinations_of_prompt_variables[en]
            for key, value in self.dict_variables_with_values.items():
                if isinstance(value, (int, str, float, bool)):
                    prompt_values[key] = value
            prompt = self.prompt.format(**prompt_values)
            prompt_list.append(prompt)
        return prompt_list
