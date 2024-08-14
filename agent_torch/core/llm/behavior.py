from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.backend import DspyLLM
from agent_torch.core.llm.prompt_manager import PromptManager
from agent_torch.core.dataloader import LoadPopulation
import torch


class Behavior:
    def __init__(self, archetype, region):
        self.archetype = archetype
        self.population = LoadPopulation(region)
        self.prompt_manager = PromptManager(
            self.archetype[-1].user_prompt, self.population
        )
        [
            archetype.initialize_memory(num_agents=self.prompt_manager.distinct_groups)
            for archetype in self.archetype
        ]

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
        agent_outputs = []
        for num_retries in range(10):
            try:
                # last_k : Number of previous conversations to add in history
                for n_arch in range(self.archetype[-1].n_arch):
                    agent_outputs.append(self.archetype[n_arch](prompt_list, last_k=12))
                break

            except Exception as e:
                print(f"Error in sampling behavior: {e}")
                print("Retrying")
                continue

        sampled_behavior = self.get_sampled_behavior(
            sampled_behavior, masks, agent_outputs
        )

        # Save current step's conversation history to file
        # file_dir : Path to export current step's conversation history
        self.archetype[-1].export_memory_to_file(
            file_dir=kwargs["current_memory_dir"], last_k=len(prompt_list)
        )

        return sampled_behavior

    def get_sampled_behavior(self, sampled_behavior, masks, agent_outputs):
        for agent_output in agent_outputs:
            for en, output_value in enumerate(agent_output):
                value_for_group = float(output_value)
                sampled_behavior_for_group = masks[en] * value_for_group
                sampled_behavior = torch.add(
                    sampled_behavior, sampled_behavior_for_group
                )
        n = len(agent_outputs)
        average_sampled_behavior = sampled_behavior / n
        return average_sampled_behavior

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
