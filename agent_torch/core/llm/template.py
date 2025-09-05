"""
Template System for AgentTorch
=============================

Clean, single-responsibility template system for organizing prompts with external data sources.
Users have full control over prompt text with {placeholder, learnable=True/False} syntax.

Single Public API: Template.render(agent_id, population, mapping, config_kwargs)
"""

import os
import json
import pandas as pd
import pickle
import re
import torch
from typing import Dict, Any, Optional, Union, List, Tuple, Literal
from dataclasses import dataclass

# Import Slot class
from agent_torch.core.llm.Variable import Variable


@dataclass
class Template:
    """
    Template structure for organizing prompt data sources.
    

    """
    src: Optional[str] = None
    grouping_logic: Optional[Union[str, List[str]]] = None
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        self.optimized_slots: Dict[str, int] = {}  # Store P3O slot choices
        # Optional externally provided datasets and configs
        self._external_df: Optional[pd.DataFrame] = getattr(self, "_external_df", None)
        self._match_on: Optional[str] = getattr(self, "_match_on", None)
        # Ground-truth is owned by Archetype; no internal defaults here
        # Register class-declared Variables (descriptor instances)
        self._variables: Dict[str, Variable] = {}
        for name, attr in vars(self.__class__).items():
            if isinstance(attr, Variable):
                self._variables[name] = attr
                # Initialize instance value from default if not already set
                if name not in self.__dict__:
                    if attr.default is not None:
                        self.__dict__[name] = attr.default
    
    def set_optimized_slots(self, slot_choices: Dict[str, int]):
        """
        Set P3O optimized slot choices. Template will use these for presentation.
        
        Args:
            slot_choices: Dictionary mapping field names to slot choice indices
                         e.g., {"age": 3, "primary_tasks": 1, "abilities": 2}
        """
        self.optimized_slots = slot_choices.copy()
    
    def clear_optimized_slots(self):
        """Clear P3O slot choices, reverting to default presentation."""
        self.optimized_slots = {}
    
    def _load_external_data(self) -> Optional[pd.DataFrame]:
        """
        Load external data from the src path (strict mode).
        
        Returns:
            DataFrame with external data, or None if not configured
        Raises:
            FileNotFoundError/ValueError on invalid paths or formats
        """
        # Prefer programmatically provided DataFrame
        if isinstance(getattr(self, "_external_df", None), pd.DataFrame):
            return self._external_df

        if not self.src:
            return None
        if not os.path.exists(self.src):
            raise FileNotFoundError(f"External data file not found: {self.src}")
            
            if self.src.endswith('.pkl'):
                with open(self.src, 'rb') as f:
                    return pickle.load(f)
        if self.src.endswith('.csv'):
                return pd.read_csv(self.src)
        raise ValueError(f"Unsupported external data format: {self.src}")
    
    def load_ground_truth_dict(self) -> Dict[str, Any]:
        """Deprecated: ground truth handling moved to Archetype.configure."""
        raise NotImplementedError("Use Archetype.configure(..., ground_truth_src=...) for ground truth.")
    
    def parse_template_fields(self, template_string: str) -> List[Tuple[str, bool]]:
        """
        Parse template string to extract field names and learnable flags.
        
        Args:
            template_string: String with {field, learnable=True/False} syntax
            
        Returns:
            List of (field_name, is_learnable) tuples
            
        Example:
            parse_template_fields("I am {age, learnable=True} and work as {job, learnable=False}")
            # Returns: [('age', True), ('job', False)]
        """
        # Pattern to match {field_name, learnable=True/False}
        pattern = r'\{([^,}]+)(?:,\s*learnable\s*=\s*(True|False))?\}'
        matches = re.findall(pattern, template_string)
        
        result = []
        for field_name, learnable_str in matches:
            field_name = field_name.strip()
            # Default to False if learnable not specified
            is_learnable = learnable_str.strip() == 'True' if learnable_str else False
            result.append((field_name, is_learnable))
            
        return result
    
    def fields(self, section: Literal["archetype", "external", "config"]) -> List[Tuple[str, bool]]:
        """Return fields referenced by the effective base prompt (section ignored)."""
        text = self.get_base_prompt_manager_template()
        return self.parse_template_fields(text)
    
    def create_slots(self) -> Dict[str, Variable]:
        """Backward-compatible API: return variables for learnables."""
        base_text = self.get_base_prompt_manager_template()
        fields = self.parse_template_fields(base_text)
        vars_map: Dict[str, Variable] = {}
        for var_name, var in getattr(self, "_variables", {}).items():
            if var.learnable:
                vars_map[var_name] = var
        # include parsed learnables not declared explicitly by creating dynamic Variables
        for field_name, is_learnable in fields:
            if is_learnable and field_name not in vars_map:
                dyn = Variable(desc=f"dynamic {field_name}", learnable=True)
                vars_map[field_name] = dyn
        return vars_map
    
    def create_p3o_placeholder_choices(self, mapping: Dict[str, Any] = None) -> Dict[str, Tuple[int, Any]]:
        """
        Convert Template fields to P3O placeholder_choices format using Slot objects.
        
        Args:
            mapping: Mapping dictionary from mapping.json for value transformations
            
        Returns:
            Dictionary compatible with P3O PromptGeneratorModule placeholder_choices
        """
        slots = self.create_slots()
        
        # Convert Slot objects to P3O format with mapping
        choices = {}
        for field_name, slot in slots.items():
            choices[field_name] = slot.get_p3o_choice(mapping=mapping)
                
        return choices
    
    def get_slot_parameters(self) -> List[torch.Tensor]:
        """
        Get all learnable parameters from slots for P3O optimization.
            
        Returns:
            List of PyTorch parameters that can be optimized
        """
        parameters: List[torch.Tensor] = []
        for name, var in self.create_slots().items():
            param = var.get_parameter(self)
            if isinstance(param, torch.nn.Parameter):
                parameters.append(param)
        return parameters
    
    def get_base_prompt_manager_template(self) -> str:
        """
        Return template compatible with base PromptManager (no learnable syntax).
        Converts {field, learnable=True/False} to {field}.
        """
        # Invoke optional hooks to allow users to populate fields
        sys_hook = getattr(self, "__system_prompt__", None)
        data_hook = getattr(self, "__data__", None)
        prompt_hook = getattr(self, "__prompt__", None)
        out_hook = getattr(self, "__output__", None)

        if callable(data_hook):
            data_hook()
        if callable(prompt_hook):
            prompt_hook()

        # Resolve system prompt (hook wins; else class attribute)
        sys_text = str(sys_hook()) if callable(sys_hook) else None
        if not sys_text:
            sp = getattr(self, "system_prompt", None)
            sys_text = sp.strip() if isinstance(sp, str) and sp.strip() else None

        # Resolve base prompt
        prompt_text = self.prompt_string if isinstance(getattr(self, "prompt_string", None), str) else None

        # Resolve output instruction (hook wins; else any supported class attribute)
        out_text = str(out_hook()) if callable(out_hook) else None
        if not out_text:
            for name in ("output", "output_instruction", "output_prompt"):
                cand = getattr(self, name, None)
                if isinstance(cand, str) and cand.strip():
                    out_text = cand.strip()
                    break

        parts = [p for p in (sys_text, prompt_text, out_text) if isinstance(p, str) and p]
        text = " ".join(parts)
        return self._normalize_self_placeholders(self._strip_learnable_syntax(text))

    def _strip_learnable_syntax(self, text: str) -> str:
        pattern = r'\{([^,}]+)(?:,\s*learnable\s*=\s*(?:True|False|true|false))?\}'
        return re.sub(pattern, r'{\1}', text)

    def _normalize_self_placeholders(self, text: str) -> str:
        # Convert {self.var} -> {var}
        return re.sub(r"\{\s*self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\}", r"{\1}", text)
    
    def _process_field_value(self, field_name: str, value: Any) -> str:
        """
        Process field values from DataFrame columns into string format.
        
        Args:
            field_name: Name of the field being processed
            value: Value from DataFrame column
            
        Returns:
            Clean string representation for LLM prompts
        """
        if pd.isna(value) or value is None:
            return ""
        
        # DataFrame columns should already be clean - just convert to string
        return str(value)
    
    def _fill_section(self, template_section: str, data: Dict[str, Any], slot_values: Dict[str, Any] = None) -> str:
        """
        Fill a template section with data, handling both regular and P3O slots.
        
        Args:
            template_section: Template string to fill
            data: Data dictionary with field values
            slot_values: Optional P3O slot values
            
        Returns:
            Filled template section
        """
        filled_section = template_section
        
        # Use provided slot_values or stored optimized_slots from P3O
        active_slot_values = slot_values or self.optimized_slots
        
        # Pattern to match placeholders like {field} (learnable inferred from Variable registry)
        pattern = r'\{([^,}]+)(?:,\s*learnable\s*=\s*(True|False|true|false))?\}'

        # Cache slots once per call
        slots = self.create_slots()
        
        def replace_placeholder(match):
            field_name = match.group(1).strip()
            # Infer learnable from declared Variables, ignore inline flags
            var = slots.get(field_name)
            is_learnable = bool(var and getattr(var, 'learnable', False))

            # Get the value for this field
            if is_learnable and active_slot_values and field_name in active_slot_values:
                # P3O mode: Use slot choice to format the actual data
                slot_choice = active_slot_values[field_name]
                
                # Skip P3O formatting if data is missing - fall back to normal mode
                if field_name not in data:
                    raise KeyError(f"Field '{field_name}' missing from data")
                
                # Apply P3O presentation choice using Slot lambda function
                if var is not None:
                    # Use mapping if available (from population mapping.json or src directory)
                    mapping = getattr(self, '_mapping', None)
                    _, lambda_func = var.get_p3o_choice(mapping)
                    formatted_value = lambda_func(slot_choice, data)
                    return formatted_value
                else:
                    return str(data[field_name])
            elif field_name in data:
                # Normal mode: Use data value directly
                return str(data[field_name])
            else:
                # Field missing from data
                raise KeyError(f"Field '{field_name}' missing from data")
        
        # Replace all placeholders
        filled_section = re.sub(pattern, replace_placeholder, filled_section)
        
        return filled_section
    
    def grouping_key(self, agent_profile: Dict[str, Any]) -> str:
        """
        Get grouping key for an agent based on grouping_logic.
        
        Args:
            agent_profile: Complete agent profile dictionary
            
        Returns:
            String key for grouping agents with similar characteristics
        """
        if not self.grouping_logic:
            # Default: group by population attributes referenced in the template if available
            if agent_profile:
                parts: List[str] = []
                for key in sorted(agent_profile.keys()):
                    parts.append(str(agent_profile.get(key, "")))
                return "|".join(parts) if any(parts) else "default_group"
            # Fallback: if no profile provided, try learnable fields; else single default group
            learnable_fields = [name for name, slot in self.create_slots().items() if slot.learnable]
            if learnable_fields:
                return "_".join([str(agent_profile.get(field, "")) for field in learnable_fields])
                return "default_group"
        
        # Accept list of fields for composite grouping
        if isinstance(self.grouping_logic, (list, tuple)):
            parts: List[str] = []
            for field in self.grouping_logic:
                parts.append(str(agent_profile.get(field, "")))
            return "|".join(parts)
        
        # Direct field grouping (e.g., "job_title" -> group by job_title)
        return str(agent_profile.get(self.grouping_logic, "unknown"))
    
    # --- Mapping and grouped prompt generation (unifies former DataFramePromptManager) ---
    def _load_mapping(self, population=None) -> Dict[str, Any]:
        mapping: Dict[str, Any] = {}
        base_dir: Optional[str] = None
        if population is not None:
            if hasattr(population, 'population_folder_path'):
                base_dir = str(population.population_folder_path)
            elif hasattr(population, '__path__'):
                try:
                    base_dir = population.__path__[0]
                except Exception:
                    base_dir = None
        if base_dir is None and getattr(self, 'src', None):
            base_dir = os.path.dirname(self.src)
        if not base_dir:
            return {}
        mapping_path = os.path.join(base_dir, 'mapping.json')
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def get_grouped_prompts(self, population, kwargs: Dict[str, Any]) -> tuple[list[str], list[str], list[list[int]]]:
        """Group agents by grouping_logic and return prompts, keys, and indices per group."""
        pop_size = getattr(population, 'population_size', 0)
        mapping = self._load_mapping(population)
        # Persist mapping for downstream rendering (e.g., P3O variant prints)
        try:
            self._mapping = mapping
        except Exception:
            pass
        external_df = self._load_external_data()

        # Build buckets by grouping key
        buckets: Dict[str, list[int]] = {}
        # Determine grouping keys
        if self.grouping_logic:
            keys = self.grouping_logic if isinstance(self.grouping_logic, (list, tuple)) else [self.grouping_logic]
            keys = [k for k in keys if isinstance(k, str) and k]
        else:
            # Derive default keys from placeholders referenced in the template that exist on the population
            base_text_for_fields = self.get_base_prompt_manager_template()
            requested_fields = set(re.findall(r"\{([^,}]+)\}", base_text_for_fields))
            keys = [fld for fld in requested_fields if hasattr(population, fld)]
        for i in range(pop_size):
            profile: Dict[str, Any] = {}
            for key in keys:
                val = None
                if hasattr(population, key):
                    attr = getattr(population, key)
                    if hasattr(attr, '__getitem__') and i < len(attr):
                        v = attr[i]
                        if hasattr(v, 'item'):
                            v = v.item()
                        # apply mapping if available
                        if key in mapping and isinstance(v, (int, float)):
                            mv = mapping[key]
                            try:
                                idx = int(v)
                                if 0 <= idx < len(mv):
                                    v = mv[idx]
                            except Exception:
                                pass
                        val = v
                if val is None and external_df is not None and i < len(external_df) and key in list(external_df.columns):
                    val = external_df.iloc[i][key]
                profile[key] = val
            gkey = self.grouping_key(profile)
            buckets.setdefault(gkey, []).append(i)

        # Create prompts per group using representative agent
        prompt_list: list[str] = []
        group_keys: list[str] = []
        group_indices: list[list[int]] = []
        for gkey, indices in buckets.items():
            rep_id = indices[0]
            ctx = {**(kwargs or {}), 'agent_id': rep_id}
            prompt = self.render(agent_id=rep_id, population=population, mapping=mapping, config_kwargs=ctx)
            prompt_list.append(prompt)
            group_keys.append(gkey)
            group_indices.append(indices)

        return prompt_list, group_keys, group_indices
    
    # Removed legacy make_agent_mask; masking is handled at behavior layer.
    
    def render(self, agent_id: int, population, mapping: Dict[str, Any], config_kwargs: Dict[str, Any]) -> str:
        """
        Single Responsibility: Generate a fully resolved prompt for an agent.
        
        This is the ONLY public API for prompt generation. Handles ALL data loading, merging, and placeholder resolution:
        - Population attributes (age, gender, etc.)
        - External data (CSV/PKL rows) 
        - Config values (from YAML or kwargs)
        - Resolving every placeholder {field, learnable=True/False}
        - Inserting output_instruction
        
        Args:
            agent_id: Agent ID to generate prompt for
            population: Population object with agent attributes
            mapping: Mapping dict for converting raw values to human-readable text
            config_kwargs: Config values from simulation
            
        Returns:
            Fully resolved prompt string ready for LLM
        """
        # Assemble complete data for this agent
        all_data = self.assemble_data(agent_id=agent_id, population=population, mapping=mapping, config_kwargs=config_kwargs)
        
        # Class-based hook mode only
        if hasattr(self, "__data__") and callable(getattr(self, "__data__")):
            self.__data__()
        if hasattr(self, "__prompt__") and callable(getattr(self, "__prompt__")):
            self.__prompt__()
        parts = []
        # Optional system prompt hook takes precedence if provided
        sys_text = None
        if hasattr(self, "__system_prompt__") and callable(getattr(self, "__system_prompt__")):
            try:
                sys_text = str(self.__system_prompt__())
            except Exception:
                sys_text = None
        if not sys_text:
            system_prompt = getattr(self, "system_prompt", None)
            if isinstance(system_prompt, str) and system_prompt.strip():
                sys_text = system_prompt.strip()
        if sys_text:
            parts.append(sys_text)
        base_text = getattr(self, "prompt_string", "")
        base_text = self._normalize_self_placeholders(base_text)
        # Replace placeholders using all_data
        rendered = self._fill_section(base_text, all_data)
        parts.append(rendered)
        # Optional output hook or class attribute
        out_text = None
        if hasattr(self, "__output__") and callable(getattr(self, "__output__")):
            try:
                out_text = str(self.__output__())
            except Exception:
                out_text = None
        if not out_text:
            for attr_name in ("output", "output_instruction", "output_prompt"):
                cand = getattr(self, attr_name, None)
                if isinstance(cand, str) and cand.strip():
                    out_text = cand.strip()
                    break
        if out_text:
            parts.append(out_text)
        return " ".join(p for p in parts if p).strip()

    def assemble_data(self, agent_id: int, population, mapping: Dict[str, Any] = None, config_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Assemble and return the complete data dict used to fill the template for a given agent.
        Includes population, external, and config data.
        """
        mapping = mapping or {}
        # 1. Population data
        population_data: Dict[str, Any] = {}
        if population:
            # Determine all fields referenced in the base prompt (handles class-based hooks)
            base_text_for_fields = self.get_base_prompt_manager_template()
            requested_fields = set(re.findall(r"\{([^,}]+)\}", base_text_for_fields))
            for field_name in requested_fields:
                if hasattr(population, field_name):
                    field_data = getattr(population, field_name)
                    if hasattr(field_data, '__getitem__') and agent_id < len(field_data):
                        raw_value = field_data[agent_id]
                        if hasattr(raw_value, 'item'):
                            raw_value = raw_value.item()
                        # Normalize common cases and types
                        if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
                            normalized_value = ""
                        else:
                            normalized_value = raw_value
                            # Strings like "0" → 0
                            if isinstance(normalized_value, str) and normalized_value.isdigit():
                                try:
                                    normalized_value = int(normalized_value)
                                except Exception:
                                    pass
                            # Float that is integral → int
                            if isinstance(normalized_value, float) and hasattr(normalized_value, 'is_integer') and normalized_value.is_integer():
                                normalized_value = int(normalized_value)

                        if field_name in mapping and isinstance(normalized_value, (int, float)):
                            mapping_values = mapping[field_name]
                            try:
                                idx = int(normalized_value)
                                if 0 <= idx < len(mapping_values):
                                    population_data[field_name] = mapping_values[idx]
                                else:
                                    population_data[field_name] = str(normalized_value)
                            except Exception:
                                population_data[field_name] = str(normalized_value)
                        else:
                            population_data[field_name] = normalized_value
        # 2a. Self-declared variables (from class-based templates)
        self_vars: Dict[str, Any] = {}
        for var_name, var in getattr(self, "_variables", {}).items():
            value = getattr(self, var_name, var.default)
            if value is not None:
                self_vars[var_name] = value

        # 2. External data
        external_data: Dict[str, Any] = {}
        external_df = self._load_external_data()
        if external_df is not None:
            base_text_for_fields = self.get_base_prompt_manager_template()
            requested_fields = set(re.findall(r"\{([^,}]+)\}", base_text_for_fields))
            # Determine matching row based on configured key if available
            match_on_key: Optional[str] = getattr(self, "_match_on", None)
            matched_row: Optional[pd.Series] = None
            if match_on_key:
                # get key value from precedence: self_vars > population_data > config_data
                key_value = None
                if match_on_key in self_vars:
                    key_value = self_vars[match_on_key]
                elif match_on_key in population_data:
                    key_value = population_data[match_on_key]
                elif match_on_key in (config_kwargs or {}):
                    key_value = (config_kwargs or {}).get(match_on_key)
                # Fallback: read directly from population even if not requested
                if key_value is None and population is not None and hasattr(population, match_on_key):
                    try:
                        attr_data = getattr(population, match_on_key)
                        if hasattr(attr_data, '__getitem__') and agent_id < len(attr_data):
                            kv = attr_data[agent_id]
                            if hasattr(kv, 'item'):
                                kv = kv.item()
                            key_value = kv
                    except Exception:
                        key_value = None
                # Final fallback: read from external_df at agent index if column exists
                if key_value is None and match_on_key in external_df.columns:
                    try:
                        if 0 <= agent_id < len(external_df):
                            key_value = external_df.iloc[agent_id][match_on_key]
                    except Exception:
                        key_value = None
                # Ensure the match key itself is available in final data for prompt filling
                if key_value is not None and match_on_key not in self_vars and match_on_key not in population_data:
                    external_data[match_on_key] = key_value
                if key_value is not None and match_on_key in external_df.columns:
                    try:
                        # Robust match: compare as strings to avoid dtype mismatches
                        key_str = str(key_value)
                        col_as_str = external_df[match_on_key].astype(str)
                        candidates = external_df[col_as_str == key_str]
                        if not candidates.empty:
                            matched_row = candidates.iloc[0]
                        else:
                            # Fallback: index-aligned selection if available
                            if 0 <= agent_id < len(external_df):
                                matched_row = external_df.iloc[agent_id]
                    except Exception:
                        matched_row = None
            # Fallback selection: respect match_on semantics
            if matched_row is None:
                # Pre-broadcast (population is None): always allow index-aligned fallback
                if population is None and 0 <= agent_id < len(external_df):
                    matched_row = external_df.iloc[agent_id]
                else:
                    if match_on_key:
                        # When population is present and match_on is specified but no match, skip external fill
                        matched_row = None
                    else:
                        if len(external_df) == 0:
                            raise ValueError("External data frame is empty")
                        if not (0 <= agent_id < len(external_df)):
                            raise IndexError(f"agent_id {agent_id} out of range for external_df length {len(external_df)}")
                        matched_row = external_df.iloc[agent_id]
            if matched_row is not None:
                # Prefer copying all columns from the matched row to maximize availability pre-broadcast
                for col in list(getattr(matched_row, 'index', [])):
                    if col in self_vars or col in population_data:
                        continue
                    try:
                        external_data[col] = self._process_field_value(col, matched_row[col])
                    except Exception:
                        pass
                # Also ensure any requested fields not covered above are attempted
                for field_name in requested_fields:
                    if field_name in external_data or field_name in self_vars or field_name in population_data:
                        continue
                    if field_name in getattr(matched_row, 'index', []):
                        try:
                            external_data[field_name] = self._process_field_value(field_name, matched_row[field_name])
                        except Exception:
                            pass
        # 3. Config data
        config_data = dict(config_kwargs or {})
        # No special-case aliasing; users must provide canonical keys
        # Merge with precedence: external < population < self_vars < config
        all_data: Dict[str, Any] = {}
        all_data.update(external_data)
        all_data.update(population_data)
        all_data.update(self_vars)
        all_data.update(config_data)
        return all_data

        # Removed unreachable legacy hook utilities.
    
    # Removed deprecated configure; ground truth is owned by Archetype.


def load_file(filepath: str):
    raise NotImplementedError("Use pandas to load data within __data__ hooks directly.")
        