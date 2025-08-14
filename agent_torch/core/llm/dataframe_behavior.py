"""
DataFrameBehavior Extension
==========================

Extends Behavior to work seamlessly with DataFrameArchetype and DataFramePromptManager.
Bridges template-based behavior system with dynamic DataFrame-driven job profiles.

Key features:
- Uses DataFramePromptManager for SOC-code based prompt generation
- Works with DataFrameLLMArchetype for job profile processing  
- Maintains full compatibility with population masking and tensor operations
- Integrates GPU population data with DataFrame job profiles
"""

import torch
import pandas as pd
from typing import Dict, Any, Optional
from agent_torch.core.llm.behavior import Behavior
from agent_torch.core.llm.dataframe_prompt_manager import DataFramePromptManager
from agent_torch.core.dataloader import LoadPopulation


class DataFrameBehavior(Behavior):
    """
    Enhanced Behavior class for DataFrame-based job profile archetypes.
    
    Uses DataFramePromptManager to generate SOC-code based prompts instead
    of template-based prompts, enabling seamless integration with job profile data.
    """
    
    def __init__(self, archetype, region, population, job_df, template=None, archetype_mapping_manager=None):
        super().__init__(archetype, region)
        self.job_df = job_df
        self.population = population if population else self._create_fallback_population()
        
        # Initialize SOC code cache for efficiency
        self.soc_cache = {}  # {soc_code: final_score}
        
        # 🆕 Store archetype mapping manager (NEW)
        self.archetype_mapping_manager = archetype_mapping_manager
        
        # Data prompt manager
        user_prompt = f"You are analyzing job-specific behavior in {region}."
        try:
            self.prompt_manager = DataFramePromptManager(user_prompt, population, job_df, archetype_mapping_manager)
            # Set template if provided
            if template is not None:
                # 🆕 Handle both string and TemplateDefinition objects (NEW)
                if hasattr(template, 'name'):
                    # TemplateDefinition object
                    self.prompt_manager.template = template
                else:
                    # String template - try to resolve
                    try:
                        from .template import Template
                        resolved_template = Template.resolve_template(template)
                        self.prompt_manager.template = resolved_template
                    except Exception as e:
                        print(f"⚠️ Failed to resolve template '{template}': {e}")
                        # Keep as string for backward compatibility
                        self.prompt_manager.template = template
            # 🆕 Set archetype mapping manager if provided (NEW)
            if archetype_mapping_manager:
                self.prompt_manager.archetype_mapping_manager = archetype_mapping_manager
        except Exception as e:
            print(f"⚠️ DataFramePromptManager failed: {e}")
            self.prompt_manager = None
        
    def __call__(self, observation):
        """Make DataFrameBehavior callable for @with_behavior compatibility."""
        return self.sample(observation)

    def calculate_average_scores(self, agent_outputs):
        """Calculate average scores across archetypes for each agent"""
        pass # Placeholder for now
    
    def sample(self, observation: Dict[str, Any], **kwargs):
        """Sample behavior tensor using DataFrameArchetype with P3O step management."""
        device = observation.get('device', torch.device('cpu'))
        population_size = observation.get('num_agents', 0)
        
        if population_size == 0:
            return torch.empty(0, 1, device=device)
        
        # 🔥 NEW: Start P3O step if backend supports it
        if hasattr(self.archetype[0].llm, 'start_step'):
            self.archetype[0].llm.start_step()
        
        # Create behavior tensor 
        sampled_behavior = torch.zeros(population_size, 1, device=device)
        
        # Check for generalizable tensor reuse (using any categorical field)
        categorical_tensor_cache = {}
        
        for agent_idx in range(population_size):
            agent_data = observation['agent_data_list'][agent_idx]
            
            # 🧠 EXTERNAL DATA LOOKUP-BASED CACHE: Use external data lookup fields for caching
            # This makes behavioral sense - agents with same external data keys get same behavior
            cache_key = None
            
            # Get mapping manager to access external sources configuration
            mapping_manager = getattr(self, 'archetype_mapping_manager', None)
            cache_parts = []
            
            if mapping_manager and hasattr(mapping_manager, 'field_sources'):
                # 🎯 PRIMARY: Use external data lookup fields for caching
                external_sources = mapping_manager.field_sources.get('external_sources', {})
                
                # Build cache key from lookup fields in mapping.json order
                for source_name, source_config in external_sources.items():
                    lookup_field = source_config.get('lookup_field')
                    if lookup_field and lookup_field in agent_data:
                        field_value = agent_data[lookup_field]
                        # Handle list values by converting to string for cache key
                        if isinstance(field_value, list):
                            field_value = str(hash(tuple(field_value)))
                        # Include None values explicitly
                        if field_value is None:
                            field_value = "None"
                        cache_parts.append(f"{lookup_field}:{field_value}")
            
            # Create cache key from external lookup fields
            if cache_parts:
                cache_key = "_".join(cache_parts)
            else:
                # 🔄 FALLBACK: Use population fields if no external sources
                population_fields = set()
                if mapping_manager and hasattr(mapping_manager, 'field_sources'):
                    population_fields = set(mapping_manager.field_sources.get('population_fields', {}).keys())
                
                population_values = []
                for field_name, field_value in agent_data.items():
                    # Only use population fields (no hardcoded context field exclusions)
                    if population_fields and field_name in population_fields:
                        # Handle list values by converting to string for cache key
                        if isinstance(field_value, list):
                            field_value = str(hash(tuple(field_value)))
                        if field_value is None:
                            field_value = "None"
                        population_values.append(f"{field_name}:{field_value}")
                    elif not population_fields:
                        # If we can't determine population fields, skip only agent_id
                        if field_name != 'agent_id':
                            if isinstance(field_value, list):
                                field_value = str(hash(tuple(field_value)))
                            if field_value is None:
                                field_value = "None"
                            population_values.append(f"{field_name}:{field_value}")
                
                # Create multi-field cache key from population fields (limit to first 3 for efficiency)
                if population_values:
                    cache_key = "_".join(population_values[:3])
                else:
                    cache_key = "default"  # Final fallback
            
            # Check if we already processed this categorical value
            if cache_key in categorical_tensor_cache:
                sampled_behavior[agent_idx] = categorical_tensor_cache[cache_key]
                cached_value = categorical_tensor_cache[cache_key]
                # Handle both tensor and float values for display
                display_value = cached_value.item() if hasattr(cached_value, 'item') else float(cached_value)
                print(f"  REUSE: {cache_key} from agent[{categorical_tensor_cache[f'{cache_key}_first_idx']}] → tensor[{agent_idx}] = {display_value:.3f}")
                continue
            
            # Process new categorical value
            agent_scores = []
            
            for arch_idx in range(len(self.archetype)):
                agent_prompt = self.prompt_manager.generate_prompt_from_data(agent_data)
                arch_output = self.archetype[arch_idx]([agent_prompt], last_k=kwargs.get('last_k', 12))
                
                if arch_output is not None and len(arch_output) > 0:
                    agent_scores.append(arch_output[0])
                else:
                    raise ValueError(f"Archetype {arch_idx} returned None for agent {agent_idx}")
            
            # Average archetype scores
            if agent_scores:
                avg_score = sum(agent_scores) / len(agent_scores)
                sampled_behavior[agent_idx] = avg_score
                
                # 🆕 GENERALIZABLE: Cache for categorical-based reuse
                categorical_tensor_cache[cache_key] = avg_score
                categorical_tensor_cache[f'{cache_key}_first_idx'] = agent_idx
                print(f"  FILLED: tensor[{agent_idx}] = {avg_score:.3f}")
                
                # 🔥 NEW: Trigger P3O optimization after each completed agent (not each archetype call)
                if hasattr(self.archetype[0].llm, 'p3o_enabled') and self.archetype[0].llm.p3o_enabled:
                    if hasattr(self.archetype[0].llm, '_trigger_agent_level_p3o'):
                        self.archetype[0].llm._trigger_agent_level_p3o(agent_idx + 1, avg_score)
            else:
                raise ValueError(f"No valid archetype outputs for agent {agent_idx}")
        
        # Count reused agents vs new LLM calls
        agents_reused = population_size - len(categorical_tensor_cache)  # Agents that used cached categorical values
        llm_calls_made = len(categorical_tensor_cache) * len(self.archetype)  # Unique categories × archetypes per category
        
        print(f"Completed: {agents_reused} reused, {llm_calls_made} LLM calls")
        
        # 🔥 NEW: End P3O step if backend supports it
        if hasattr(self.archetype[0].llm, 'end_step'):
            self.archetype[0].llm.end_step()
        
        return sampled_behavior
    
    def _fill_individual_agent_scores(self, sampled_behavior: torch.Tensor, agent_outputs: list, prompt_list: list) -> torch.Tensor:
        """
        Fill population tensor with individual agent scores, averaging across archetypes.
        
        Args:
            sampled_behavior: Population tensor to fill [population_size, 1]
            agent_outputs: List of archetype outputs [arch1_outputs, arch2_outputs, arch3_outputs]
            prompt_list: List of agent prompts with metadata
            
        Returns:
            Filled population tensor with averaged archetype scores per agent
        """
        num_agents_processed = len(prompt_list)
        num_archetypes = len(agent_outputs)
        
        print(f"Tensor filling: {num_agents_processed} agents, {num_archetypes} archetypes")
        
        # Track which actual agent IDs were processed (since we skip agents without SOC codes)
        processed_agent_ids = []
        
        # Find which agents were actually processed by checking SOC codes
        agent_id = 0
        for prompt_idx, prompt_dict in enumerate(prompt_list):
            # Find the next agent with a SOC code
            while agent_id < self.population.population_size:
                if hasattr(self.population, 'soc_codes') and hasattr(self.population.soc_codes, '__getitem__'):
                    if self.population.soc_codes[agent_id]:
                        processed_agent_ids.append(agent_id)
                        agent_id += 1
                        break
                else:
                    # No SOC codes in population, assume sequential processing
                    processed_agent_ids.append(agent_id)
                    agent_id += 1
                    break
                agent_id += 1
        
        # Fill tensor for each processed agent
        for prompt_idx in range(num_agents_processed):
            agent_id = processed_agent_ids[prompt_idx] if prompt_idx < len(processed_agent_ids) else prompt_idx
            
            # Collect scores from all archetypes for this agent
            archetype_scores = []
            for arch_idx in range(num_archetypes):
                if prompt_idx < len(agent_outputs[arch_idx]):
                    score = float(agent_outputs[arch_idx][prompt_idx])
                    archetype_scores.append(score)
            
            # Average archetype scores for this agent
            if archetype_scores:
                final_score = sum(archetype_scores) / len(archetype_scores)
                sampled_behavior[agent_id, 0] = final_score
                
                # Only log final score
                print(f"Agent {agent_id + 1}: final={final_score:.3f}")
            else:
                print(f"Agent {agent_id + 1}: no archetype scores available")
        
        # Summary of tensor filling
        non_zero_agents = int((sampled_behavior != 0).sum().item())
        print(f"   📊 TENSOR FILLING SUMMARY:")
        print(f"      ✅ Filled scores: {non_zero_agents}/{self.population.population_size} agents")
        print(f"      📈 Score range: {float(sampled_behavior.min()):.3f} - {float(sampled_behavior.max()):.3f}")
        print(f"      📊 Mean score: {float(sampled_behavior.mean()):.3f}")
        
        return sampled_behavior
    
    def get_masks_for_each_group(self, variables: Dict[str, Any]) -> list:
        """
        Generate masks for each SOC code group.
        
        For DataFrame behavior, masks are based on SOC code assignments
        rather than template variable combinations.
        
        Args:
            variables: Dictionary of variables for masking
            
        Returns:
            List of masks for each SOC code group
        """
        masks = []
        
        # Get SOC code assignments for population
        if 'soc_code' in variables:
            soc_assignments = variables['soc_code']
            
            # Create mask for each agent (individual processing)
            for target_soc in set(soc_assignments):
                
                # Create boolean mask for agents with this SOC code
                if hasattr(soc_assignments, '__iter__'):
                    # Array-like SOC assignments
                    mask = torch.tensor([soc == target_soc for soc in soc_assignments])
                else:
                    # Single SOC assignment
                    mask = torch.tensor([soc_assignments == target_soc] * self.population.population_size)
                
                mask = mask.unsqueeze(1).float()
                masks.append(mask)
        else:
            # Fallback: create uniform masks (not used in individual processing)
            print("⚠️ No SOC code variable found, creating uniform masks")
            uniform_mask = torch.ones(self.population.population_size, 1)
            masks = [uniform_mask.clone()]
        
        print(f"🎭 Generated {len(masks)} masks for SOC code groups")
        print(f"   🔧 TENSOR FILLING & BITMASK DETAILS:")
        
        if masks:
            total_coverage = 0
            for i, mask in enumerate(masks):
                active_agents = int(mask.sum().item())
                total_coverage += active_agents
                if i < 3:  # Show details for first 3 masks
                    soc_code = f"agent_group_{i}"
                    print(f"   📊 Mask {i} ({soc_code}): shape={mask.shape}, active_agents={active_agents}")
                    if active_agents > 0:
                        active_indices = torch.where(mask.squeeze())[0][:3]  # Show first 3 active indices
                        print(f"      🎯 Active agent indices: {active_indices.tolist()}...")
                elif i == 3 and len(masks) > 3:
                    print(f"   ... and {len(masks) - 3} more masks")
            
            print(f"   ✅ Total coverage: {total_coverage} agent-mask assignments")
            print(f"   📐 Population size: {self.population.population_size}")
            print(f"   📊 Expected calls: {len(masks)} SOC codes × 3 archetypes = {len(masks) * 3} LLM calls")
        
        return masks
    
    def add_soc_codes_to_population(self, soc_assignments=None):
        """
        Add SOC code assignments to population for masking.
        
        Args:
            soc_assignments: Array of SOC codes for each agent, or None for random assignment
        """
        if soc_assignments is None:
            # Randomly assign SOC codes to population
            import numpy as np
            soc_assignments = np.random.choice(
                self.prompt_manager.available_soc_codes,
                size=self.population.population_size,
                replace=True
            )
            print(f"🎲 Randomly assigned SOC codes to {self.population.population_size:,} agents")
        
        # Add to population object
        setattr(self.population, 'soc_codes', soc_assignments)
        
        # Update prompt manager variables
        self.prompt_manager.dict_variables_with_values = {
            'soc_code': soc_assignments
        }
        
        print(f"✅ SOC codes added to population")
        soc_counts = {}
        for soc in soc_assignments:
            soc_counts[soc] = soc_counts.get(soc, 0) + 1
        print(f"   Distribution: {dict(list(soc_counts.items())[:3])}...")


def create_dataframe_behavior(archetype_list, region, job_df: pd.DataFrame) -> DataFrameBehavior:
    """
    Factory function to create DataFrameBehavior
    
    Args:
        archetype_list: List of DataFrameLLMArchetype instances
        region: Population region
        job_df: DataFrame with job profiles
        
    Returns:
        DataFrameBehavior instance
    """
    return DataFrameBehavior(archetype_list, region, job_df)


if __name__ == "__main__":
    """
    Test DataFrameBehavior functionality
    """
    import pandas as pd
    from agent_torch.populations import NYC
    from agent_torch.core.llm.df_archetype import DataFrameArchetype
    from agent_torch.core.llm.claude_llm import ClaudeLocal
    
    print("🧪 Testing DataFrameBehavior...")
    
    # Load job data
    job_df = pd.read_pickle("./data/job_data_processed/job_data_processed.pkl").set_index("soc_code")
    
    # Create archetype
    df_arch = DataFrameArchetype(df_pkl_path="./data/job_data_processed/job_data_processed.pkl", n_arch=2)
    
    # Create LLM
    llm = ClaudeLocal(
        system_prompt="You are a NYC resident analyzing job willingness during COVID-19.",
        temperature=0.7,
        max_tokens=100
    )
    
    # Build archetypes
    archetypes = df_arch.llm(llm=llm)
    
    # Create DataFrameBehavior
    df_behavior = DataFrameBehavior(archetypes, NYC, job_df)
    
    # Add SOC codes to population
    df_behavior.add_soc_codes_to_population()
    
    # Test single SOC code sampling
    test_kwargs = {
        "soc_code": "35-2015.00",  # Cooks, Short Order
        "covid_cases": 5000,
        "unemployment_rate": 0.08,
        "month": "April",
        "year": 2020,
        "device": "cpu",
        "current_memory_dir": "./memories"
    }
    
    result_tensor = df_behavior.sample(test_kwargs)
    
    print(f"\n✅ DataFrameBehavior test complete!")
    print(f"📊 Result shape: {result_tensor.shape}")
    print(f"📈 Mean willingness: {float(result_tensor.mean()):.3f}") 