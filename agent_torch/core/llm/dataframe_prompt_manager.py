"""
DataFramePromptManager - Template-Based Prompt Management
=======================================================

True subclass of PromptManager that delegates all logic to Template.
Only overrides __init__ and the two getter methods - no regex, no mapping logic.

Key principle: Template.render() handles all placeholder filling and data loading.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from agent_torch.core.llm.prompt_manager import PromptManager
from agent_torch.core.llm.template import Template


def make_one_hot_mask(agent_id: int, population_size: int) -> torch.Tensor:
    # Unused in this manager; kept for backward compatibility if referenced elsewhere
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mask = torch.zeros(population_size, 1, dtype=torch.bool, device=device)
    if 0 <= agent_id < population_size:
        mask[agent_id, 0] = True
    return mask


class DataFramePromptManager(PromptManager):
    """
    Template-aware prompt manager that properly extends base PromptManager.
    
    Implements group-based processing where agents are grouped by template.grouping_key
    and one representative prompt is generated per group.
    """
    
    def __init__(self, template: Template, population):
        """
        Initialize with template support.
        
        Args:
            template: Template object with prompt structure and data sources
            population: Population object for agent data
        """
        # Store template and population
        self.template = template
        self.population = population
        
        # Initialize basic attributes that base PromptManager would set
        self.user_prompt = template.get_base_prompt_manager_template()
        
        # Load mapping.json for value transformations
        self.mapping = self._load_mapping()
        
        # Store raw contexts for group-based processing
        self._raw_contexts = []
        # Cache external data frame once
        self._external_df = self.template._load_external_data()
        
        # Initialize attributes that base class expects
        self.variables = {}
        self.dict_variables_with_values = {}
    
    def _apply_mapping(self, field_name: str, raw_value):
        """Apply mapping for any field using mapping.json or return raw value."""
        if field_name in self.mapping and isinstance(raw_value, (int, float)):
            mapping_values = self.mapping[field_name]
            if 0 <= int(raw_value) < len(mapping_values):
                return mapping_values[int(raw_value)]
        return raw_value
    
    def _load_mapping(self) -> Dict[str, Any]:
        """Load mapping.json for value transformations."""
        mapping_path = os.path.join(os.path.dirname(self.template.src), "mapping.json") if self.template.src else None
        
        if mapping_path and os.path.exists(mapping_path):
            try:
                with open(mapping_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading mapping.json: {e}")
        
        return {}
    
    @property
    def combinations_of_prompt_variables_with_index(self):
        """
        Group agents by template.grouping_key and return (prompt, indices) pairs.
        
        This is the key method that enables group-based processing.
        Returns a list of (prompt_string, agent_indices_list) tuples.
        """
        # Return cached result if already computed
        if hasattr(self, '_cached_combinations'):
            return self._cached_combinations
            
        if not self._raw_contexts:
            return []
        
        buckets = self._get_groups()

        # Generate combinations in the format expected by base Behavior
        # Base behavior expects: [target_values_dict, target_values_dict, ...]
        # where each target_values_dict contains the grouping criteria for that group
        combinations = []
        
        for _, agent_indices in buckets.items():
            rep_agent_id = agent_indices[0]
            rep_profile = self._build_agent_profile({"agent_id": rep_agent_id})
            raw_grouping_value = rep_profile.get(self.template.grouping_logic, "unknown")
            mapped_value = self._apply_mapping(self.template.grouping_logic, raw_grouping_value)
            combinations.append({self.template.grouping_logic: mapped_value})
        
        # Cache the result to avoid recomputation
        self._cached_combinations = combinations
        return combinations
    
    def _build_agent_profile(self, ctx):
        """Build agent profile dict for grouping from context."""
        agent_id = ctx.get("agent_id", 0)
        
        # Load population data
        profile = {"agent_id": agent_id}
        if self.population:
            # Add population attributes (dynamic based on template requirements)
            population_fields = self.template.fields("archetype")
            for field_name, is_learnable in population_fields:
                if hasattr(self.population, field_name):
                    attr_data = getattr(self.population, field_name)
                    if hasattr(attr_data, '__getitem__') and agent_id < len(attr_data):
                        raw_value = attr_data[agent_id]
                        if hasattr(raw_value, 'item'):
                            raw_value = raw_value.item()
                        profile[field_name] = raw_value
        
        # Load external data (dynamic based on template requirements)
        external_df = self._external_df
        if external_df is not None and agent_id < len(external_df):
            row = external_df.iloc[agent_id]
            external_fields = self.template.fields("external")
            for field_name, is_learnable in external_fields:
                # Use Series index for membership
                if field_name in row.index:
                    profile[field_name] = row[field_name]
        
        return profile
    
    def _get_groups(self):
        """Get agent groupings (extracted from combinations_of_prompt_variables_with_index logic)."""
        from collections import defaultdict
        
        if not self._raw_contexts:
            return {}
        
        # Group agents by grouping key
        buckets = defaultdict(list)
        for i, ctx in enumerate(self._raw_contexts):
            # Create agent profile for grouping
            agent_profile = self._build_agent_profile(ctx)
            group_key = self.template.grouping_key(agent_profile)
            buckets[group_key].append(i)
        
        return buckets

    def get_prompt_list(self, kwargs=None):
        """
        Generate group-based prompts using combinations_of_prompt_variables_with_index.
        
        This builds raw contexts for all agents, then returns one prompt per group.
        """
        if not self.template:
            return super().get_prompt_list(kwargs)
        
        # Build population data for mask generation (only the grouping key)
        pop_size = self.population.population_size
        population_data = {}
        gkey = self.template.grouping_logic
        if hasattr(self.population, gkey):
            attr_data = getattr(self.population, gkey)
            if hasattr(attr_data, '__getitem__'):
                values = []
                for i in range(pop_size):
                    if i < len(attr_data):
                        raw_value = attr_data[i]
                        if hasattr(raw_value, 'item'):
                            raw_value = raw_value.item()
                        values.append(self._apply_mapping(gkey, raw_value))
                    else:
                        values.append(None)
                population_data[gkey] = np.array(values)
        else:
            external_df = self._external_df
            if external_df is not None and gkey in list(external_df.columns):
                values = []
                for i in range(pop_size):
                    if i < len(external_df):
                        values.append(external_df.iloc[i][gkey])
                    else:
                        values.append(None)
                population_data[gkey] = np.array(values)
        
        # Set dict_variables_with_values with actual population data for masking
        self.dict_variables_with_values = population_data
        
        # Build raw contexts for all agents
        self._raw_contexts = [
            {"agent_id": i, **(kwargs or {})}
            for i in range(pop_size)
        ]
        
        # Clear cached combinations when contexts change
        if hasattr(self, '_cached_combinations'):
            delattr(self, '_cached_combinations')
        
        # Generate fresh prompts each time (picks up latest P3O optimized slots)
        prompt_list = []
        for group_index, (group_key, agent_indices) in enumerate(self._get_groups().items()):
            # Use first agent as representative for this group
            rep_agent_id = agent_indices[0]
            rep_ctx = self._raw_contexts[rep_agent_id]

            # Always render using provided kwargs; Template will handle any missing fields gracefully
            prompt = self.template.render(
                agent_id=rep_agent_id,
                population=self.population,
                mapping=self.mapping,
                config_kwargs=rep_ctx
            )
            prompt_list.append(prompt)
        
        return prompt_list

    @property
    def distinct_groups(self) -> int:
        if not self._raw_contexts:
            return 0
        return len(self._get_groups())
    
 