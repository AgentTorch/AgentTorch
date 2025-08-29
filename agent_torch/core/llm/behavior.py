from agent_torch.core.llm.prompt_manager import PromptManager
from agent_torch.core.dataloader import LoadPopulation
import torch


class Behavior:
    def __init__(self, archetype, region, template=None, population=None, optimization_interval: int = 3):
        self.archetype = archetype
        if population is None:
            self.population = LoadPopulation(region)
        else:
            # Accept either a preloaded population (has population_size) or a module to be loaded
            if hasattr(population, 'population_size'):
                self.population = population
            else:
                self.population = LoadPopulation(population)
        self.template = template
        self._memory_initialized = False
        if self.template is None:
            # Base PromptManager flow (string prompts)
            self.prompt_manager = PromptManager(self.archetype[-1].user_prompt, self.population)
            for arch in self.archetype:
                arch.initialize_memory(num_agents=self.prompt_manager.distinct_groups)
        else:
            # Template-based grouping: initialize memory sized by number of groups
            prompts, _, _ = self.template.get_grouped_prompts(self.population, kwargs={})
            for arch in self.archetype:
                arch.initialize_memory(num_agents=len(prompts))

    def pre_sample_hook(self, kwargs):
        """Hook called before sampling - subclasses can override for custom logic."""
        pass
    
    def post_sample_hook(self, sampled_behavior, kwargs):
        """Hook called after sampling - subclasses can override for custom logic."""
        pass

    def sample(self, kwargs=None):
        print("Behavior: Decision")
        self.pre_sample_hook(kwargs)

        device = kwargs["device"]
        sampled_behavior = torch.zeros(self.population.population_size, 1, device=device)

        if self.template is not None:
            # Template-based grouped flow
            prompt_list, group_keys, group_indices = self.template.get_grouped_prompts(self.population, kwargs or {})
            max_show = int(kwargs.get("print_examples", 0)) if isinstance(kwargs, dict) else 0
            if max_show > 0:
                print(f"\n=== Population Broadcast LLM Calls ===")
                print(f"Number of unique prompts: {len(prompt_list)}")
                print(f"Number of archetypes: {self.archetype[-1].n_arch}")
                for i, prompt in enumerate(prompt_list[:max_show]):
                    print(f"\nPrompt {i+1}:\n{prompt}")
            self.last_group_keys = group_keys
            agent_outputs = []
            for num_retries in range(10):
                try:
                    for n_arch in range(self.archetype[-1].n_arch):
                        outputs = self.archetype[n_arch](prompt_list, last_k=12)
                        agent_outputs.append(outputs)
                    break
                except Exception as e:
                    print(f"Error in sampling behavior: {e}")
                    print("Retrying")
                    continue
            group_values_accum = [0.0 for _ in range(len(prompt_list))]
            for arch_outputs in agent_outputs:
                for en, output_value in enumerate(arch_outputs):
                    try:
                        text_value = output_value["text"] if isinstance(output_value, dict) and "text" in output_value else output_value
                        value_for_group = float(text_value)
                        if torch.isnan(torch.tensor(value_for_group, device=device)):
                            value_for_group = 0.0
                    except Exception:
                        value_for_group = 0.0
                    group_values_accum[en] += value_for_group
                    idx = torch.tensor(group_indices[en], dtype=torch.long, device=device)
                    sampled_behavior[idx, 0] = sampled_behavior[idx, 0] + value_for_group
            n = len(agent_outputs) if agent_outputs else 1
            sampled_behavior = sampled_behavior / max(n, 1)
            self.last_group_outputs = [v / max(n, 1) for v in group_values_accum]
            if max_show > 0:
                print(f"=== End Population LLM Calls ===\n")
            self.archetype[-1].export_memory_to_file(file_dir=kwargs["current_memory_dir"], last_k=len(prompt_list))
            self.post_sample_hook(sampled_behavior, kwargs)
            return sampled_behavior

        # Base PromptManager flow
        prompt_list = self.prompt_manager.get_prompt_list(kwargs=kwargs)
        max_show = int(kwargs.get("print_examples", 0)) if isinstance(kwargs, dict) else 0
        if max_show > 0:
            print(f"\n=== Population Broadcast LLM Calls (base) ===")
            print(f"Number of prompts: {len(prompt_list)}")
            print(f"Number of archetypes: {self.archetype[-1].n_arch}")
            for i, prompt in enumerate(prompt_list[:max_show]):
                print(f"\nPrompt {i+1}:\n{prompt}")
        masks = self.get_masks_for_each_group(self.prompt_manager.dict_variables_with_values, kwargs)
        agent_outputs = []
        for num_retries in range(10):
            try:
                for n_arch in range(self.archetype[-1].n_arch):
                    agent_outputs.append(self.archetype[n_arch](prompt_list, last_k=12))
                break
            except Exception as e:
                print(f"Error in sampling behavior: {e}")
                print("Retrying")
                continue
        sampled_behavior = self.get_sampled_behavior(sampled_behavior, masks, agent_outputs)
        if max_show > 0:
            print(f"=== End Population LLM Calls ===\n")
        self.archetype[-1].export_memory_to_file(file_dir=kwargs["current_memory_dir"], last_k=len(prompt_list))
        self.post_sample_hook(sampled_behavior, kwargs)
        return sampled_behavior

    def get_sampled_behavior(self, sampled_behavior, masks, agent_outputs):
        for agent_output in agent_outputs:
            for en, output_value in enumerate(agent_output):
                try:
                    # Handle both string and dictionary formats
                    if isinstance(output_value, dict) and "text" in output_value:
                        text_value = output_value["text"]
                    else:
                        text_value = output_value
                    
                    value_for_group = float(text_value)
                    if torch.isnan(torch.tensor(value_for_group)):
                        value_for_group = 0.0
                except Exception:
                    value_for_group = 0.0
                sampled_behavior_for_group = masks[en] * value_for_group
                sampled_behavior = torch.add(
                    sampled_behavior, sampled_behavior_for_group
                )
        n = len(agent_outputs)
        average_sampled_behavior = sampled_behavior / n
        return average_sampled_behavior

    def get_masks_for_each_group(self, variables, kwargs=None):
        masks = []
        for en, target_values in enumerate(
            self.prompt_manager.combinations_of_prompt_variables_with_index
        ):
            # Get device from kwargs (which comes from YAML config via simulation_metadata)
            device = kwargs['device']
            mask = torch.tensor([True] * self.population.population_size, device=device)
            for key, value in target_values.items():
                if key in variables:
                    if key == "age" and value == 0:
                        mask = torch.zeros_like(mask)
                    else:
                        # Convert to tensor to handle CuPy arrays, use same device as mask
                        comparison = variables[key] == value
                        if not isinstance(comparison, torch.Tensor):
                            comparison = torch.tensor(comparison, device=device)
                        mask = torch.logical_and(mask, comparison)
            mask = mask.unsqueeze(1)
            float_mask = mask.float()
            masks.append(float_mask)

        return masks
