from agent_torch.llm.archetype import Archetype
from agent_torch.llm.llm import DspyLLM
from agent_torch.llm.prompt_manager import PromptManager
from agent_torch.utils import LoadPopulation
import torch


class Behavior:
    def __init__(self, archetype, region):
        self.archetype = archetype
        self.population = LoadPopulation(region)
        self.prompt_manager = PromptManager(self.archetype.user_prompt, self.population)

    def sample(self, kwargs=None):
        print("Behavior: Decision")

        # tensor to store the sampled behavior for each agent
        sampled_behavior = torch.zeros(self.population.population_size, 1).to(
            kwargs["device"]
        )

        # Get list of prompts for each group
        prompt_list = self.prompt_manager.get_prompt_list(kwargs=kwargs)
        masks = self.get_masks_for_each_group(
            self.prompt_manager.dict_variables_with_values
        )

        for num_retries in range(10):
            try:
                # last_k : Number of previous conversations to add in history
                agent_output = self.archetype(prompt_list, last_k=3)
                break

            except Exception as e:
                print(f"Error in sampling behavior: {e}")
                print("Retrying")
                continue

        sampled_behavior = self.get_sampled_behavior(
            sampled_behavior, masks, agent_output
        )

        # Save current step's conversation history to file
        # file_dir : Path to export current step's conversation history
        self.archetype.export_memory_to_file(
            file_dir=kwargs["current_memory_dir"], last_k=len(prompt_list)
        )

        return sampled_behavior

    def get_sampled_behavior(self, sampled_behavior, masks, agent_output):
        for en, output_value in enumerate(agent_output):
            value_for_group = float(output_value)
            sampled_behavior_for_group = masks[en] * value_for_group
            sampled_behavior = torch.add(sampled_behavior, sampled_behavior_for_group)
        return sampled_behavior

    def get_masks_for_each_group(self, variables):
        masks = []
        for en, target_values in enumerate(
            self.prompt_manager.combinations_of_prompt_variables_with_index
        ):
            mask = torch.tensor([True] * self.population.population_size)
            for key, value in target_values.items():
                if key in variables:
                    if key == "age" and value == 0:
                        mask = torch.zeros_like(mask)
                    else:
                        mask = torch.logical_and(mask, variables[key] == value)
            mask = mask.unsqueeze(1)
            float_mask = mask.float()
            masks.append(float_mask)

        return masks
