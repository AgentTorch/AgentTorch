"""
Template System for AgentTorch
=============================

Clean, single-responsibility template system for organizing prompts with external data sources.
Users have full control over prompt text with {placeholder, learnable=True/False} syntax.

Single Public API: Template.render(agent_id, population, mapping, config_kwargs)
"""

import os
import pandas as pd
import pickle
import re
import torch
from typing import Dict, Any, Optional, Union, List, Tuple, Literal
from dataclasses import dataclass

# Import Slot class
from agent_torch.core.llm.slot import Slot


@dataclass
class Template:
    """
    Template structure for organizing prompt data sources.
    
    Structure:
        src: Path to external data source (pkl/csv file) - N rows, each row = agent data
        ground_truth_src: Path to ground truth data (csv file) for P3O optimization
        archetype_data: Template string for population/agent data with {field, learnable=True/False}
        external_data: Template string for external data source with {field, learnable=True/False}
        config_data: Template string for simulation config data with {field, learnable=True/False}
        output_format: Expected output format configuration (type, range, etc.)
        output_instruction: The instruction/question text for the LLM
        grouping_logic: How to group agents for behavior application (e.g., "by_job_title", "by_age_group")
        
    Example:
        template = Template(
            src="agent_torch/populations/mock_test_18/job_data.pkl",
            ground_truth_src="agent_torch/core/llm/data/ground_truth_willingness.csv",
            archetype_data="This is a {age, learnable=True} year old {gender, learnable=False}.",
            external_data="Working as {job_name, learnable=True} with skills {Skills, learnable=True}.",
            config_data="COVID cases: {covid_cases, learnable=False}.",
            output_format={"type": "float", "range": [0.0, 1.0]},
            output_instruction="Rate your willingness to continue normal activities (0.0-1.0):",
            grouping_logic="job_title"  # Apply same behavior to agents with same job
        )
    """
    src: Optional[str] = None
    ground_truth_src: Optional[str] = None
    ground_truth_column: str = "willingness_score"  # Configurable ground truth column name
    archetype_data: str = ""
    external_data: str = ""
    config_data: str = ""
    output_format: Optional[Dict[str, Any]] = None
    output_instruction: str = ""  # The instruction/question for the LLM
    grouping_logic: Optional[str] = None
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        self.optimized_slots: Dict[str, int] = {}  # Store P3O slot choices
    
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
        Load external data from the src path.
        
        Returns:
            DataFrame with external data, or None if not available
        """
        if not self.src or not os.path.exists(self.src):
            return None
            
        try:
            if self.src.endswith('.pkl'):
                with open(self.src, 'rb') as f:
                    return pickle.load(f)
            elif self.src.endswith('.csv'):
                return pd.read_csv(self.src)
            else:
                print(f"Template: Unsupported file format: {self.src}")
                return None
        except Exception as e:
            print(f"Template: Error loading external data from {self.src}: {e}")
            return None
    
    def load_ground_truth_dict(self) -> Dict[str, Any]:
        """
        Load ground truth data and return in P3O-compatible format.
        
        Returns:
            Dictionary with format {"ground_truth": {agent_id: float_value}}
        """
        if not self.ground_truth_src:
            return {}
        
        try:
            if self.ground_truth_src.endswith('.csv'):
                df = pd.read_csv(self.ground_truth_src)
                # Convert to dictionary format expected by P3O
                ground_truth_dict = {}
                for idx, row in df.iterrows():
                    agent_id = idx  # Use row index as agent_id
                    willingness = float(row.get(self.ground_truth_column, 0.5))
                    ground_truth_dict[agent_id] = willingness
                return {"ground_truth": ground_truth_dict}
            else:
                # Handle other formats if needed
                return {}
        except Exception as e:
            print(f"Failed to load ground truth data from {self.ground_truth_src}: {e}")
            return {}
    
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
        """
        Get fields for a specific template section.
        
        Args:
            section: Template section name ("archetype", "external", "config")
            
        Returns:
            List of (field_name, is_learnable) tuples
        """
        text = getattr(self, f"{section}_data")
        return self.parse_template_fields(text)
    
    def create_slots(self) -> Dict[str, 'Slot']:
        """
        Create Slot objects for all learnable fields in the template.
            
        Returns:
            Dictionary mapping field names to Slot objects
        """
        from agent_torch.core.llm.slot import create_slots_from_fields
        
        # Parse all template sections for learnable fields
        all_template_text = " ".join([
            self.archetype_data,
            self.external_data, 
            self.config_data
        ])
        
        fields = self.parse_template_fields(all_template_text)
        return create_slots_from_fields(fields)
    
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
        slots = self.create_slots()
        parameters = []
        
        for slot in slots.values():
            if slot.learnable and slot.theta is not None:
                parameters.append(slot.theta)
                
        return parameters
    
    def create_p3o_template_string(self) -> str:
        """
        Convert template sections into a single P3O-compatible template string.
        
        Converts {field, learnable=True} to {{field}} for P3O placeholders.
        Converts {field, learnable=False} to direct substitution.
        
        Returns:
            Template string with {{placeholder}} syntax for P3O
        """
        def convert_placeholders(text):
            """Convert {field, learnable=True/False} to appropriate format."""
            def replace_match(match):
                field_name = match.group(1).strip()
                learnable_str = match.group(2)
                is_learnable = learnable_str and learnable_str.strip() == 'True'
                
                if is_learnable:
                    # Convert to P3O placeholder format
                    return f"{{{{{field_name}}}}}"
                else:
                    # Keep as regular placeholder for direct substitution
                    return f"{{{field_name}}}"
            
            pattern = r'\{([^,}]+)(?:,\s*learnable\s*=\s*(True|False))?\}'
            return re.sub(pattern, replace_match, text)
        
        # Combine all template sections
        sections = []
        if self.archetype_data.strip():
            sections.append(convert_placeholders(self.archetype_data))
        if self.external_data.strip():
            sections.append(convert_placeholders(self.external_data))
        if self.config_data.strip():
            sections.append(convert_placeholders(self.config_data))
        
        return " ".join(sections)
    
    def get_base_prompt_manager_template(self) -> str:
        """
        Return template compatible with base PromptManager (no learnable syntax).
        Converts {field, learnable=True/False} to {field}.
        """
        # Build template sections directly
        sections = []
        if self.archetype_data.strip():
            sections.append(self.archetype_data)
        if self.external_data.strip():
            sections.append(self.external_data)
        if self.config_data.strip():
            sections.append(self.config_data)
        
        template = " ".join(sections)
        
        # Remove learnable syntax for base prompt manager compatibility
        pattern = r'\{([^,}]+)(?:,\s*learnable\s*=\s*(?:True|False))?\}'
        clean_template = re.sub(pattern, r'{\1}', template)
        return clean_template
    
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
        
        # Pattern to match {field} and {field, learnable=True/False} formats
        pattern = r'\{([^,}]+)(?:,\s*learnable\s*=\s*(True|False|true|false))?\}'
        
        def replace_placeholder(match):
            field_name = match.group(1).strip()
            learnable_str = match.group(2)
            is_learnable = learnable_str and learnable_str.strip().lower() == 'true'
            
            # Get the value for this field
            if is_learnable and active_slot_values and field_name in active_slot_values:
                # P3O mode: Use slot choice to format the actual data
                slot_choice = active_slot_values[field_name]
                
                # Skip P3O formatting if data is missing - fall back to normal mode
                if field_name not in data:
                    raise KeyError(f"Field '{field_name}' missing from data")
                
                # Apply P3O presentation choice using Slot lambda function
                slots = self.create_slots()
                if field_name in slots:
                    slot = slots[field_name]
                    _, lambda_func = slot.get_p3o_choice()
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
            # Default: group by all learnable fields
            learnable_fields = [name for name, slot in self.create_slots().items() if slot.learnable]
            if learnable_fields:
                return "_".join([str(agent_profile.get(field, "")) for field in learnable_fields])
            else:
                return "default_group"
        
        # Direct field grouping (e.g., "job_title" -> group by job_title)
        return str(agent_profile.get(self.grouping_logic, "unknown"))
    
    def make_agent_mask(self, agent_id: int, population_size: int) -> torch.Tensor:
        """
        Create a one-hot mask for a specific agent.
        
        Args:
            agent_id: Agent ID to create mask for
            population_size: Total population size
            
        Returns:
            Boolean tensor of shape (population_size, 1) with True only at agent_id
        """
        # Create zero tensor
        mask = torch.zeros(population_size, 1, dtype=torch.bool)
        
        # Set the specific agent to True
        if 0 <= agent_id < population_size:
            mask[agent_id, 0] = True
        
        return mask
    
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
        
        # 5. Fill template sections using unified _fill_section method
        filled_archetype = self._fill_section(self.archetype_data, all_data) if self.archetype_data else ""
        filled_external = self._fill_section(self.external_data, all_data) if self.external_data else ""
        filled_config = self._fill_section(self.config_data, all_data) if self.config_data else ""
        
        # 6. Combine into full prompt
        full_prompt = f"{filled_archetype} {filled_external} {filled_config} {self.output_instruction or ''}".strip()
        
        return full_prompt

    def assemble_data(self, agent_id: int, population, mapping: Dict[str, Any] = None, config_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Assemble and return the complete data dict used to fill the template for a given agent.
        Includes population, external, and config data.
        """
        mapping = mapping or {}
        # 1. Population data
        population_data: Dict[str, Any] = {}
        if population:
            for field_name, is_learnable in self.fields("archetype"):
                if hasattr(population, field_name):
                    field_data = getattr(population, field_name)
                    if hasattr(field_data, '__getitem__') and agent_id < len(field_data):
                        raw_value = field_data[agent_id]
                        if hasattr(raw_value, 'item'):
                            raw_value = raw_value.item()
                        if isinstance(raw_value, float) and hasattr(raw_value, 'is_integer') and raw_value.is_integer():
                            raw_value = int(raw_value)
                        if field_name in mapping and isinstance(raw_value, (int, float)):
                            mapping_values = mapping[field_name]
                            if 0 <= int(raw_value) < len(mapping_values):
                                population_data[field_name] = mapping_values[int(raw_value)]
                            else:
                                population_data[field_name] = str(raw_value)
                        else:
                            population_data[field_name] = raw_value
        # 2. External data
        external_data: Dict[str, Any] = {}
        external_df = self._load_external_data()
        if external_df is not None and agent_id < len(external_df):
            row = external_df.iloc[agent_id]
            for field_name, is_learnable in self.fields("external"):
                if field_name in row.index:
                    external_data[field_name] = self._process_field_value(field_name, row[field_name])
        # 3. Config data
        config_data = dict(config_kwargs or {})
        if 'covid_cases' not in config_data and 'base_covid_cases' in config_data:
            config_data['covid_cases'] = config_data['base_covid_cases']
        if 'unemployment_rate' not in config_data and 'base_unemployment_rate' in config_data:
            config_data['unemployment_rate'] = config_data['base_unemployment_rate']
        # Merge
        all_data: Dict[str, Any] = {}
        all_data.update(population_data)
        all_data.update(external_data)
        all_data.update(config_data)
        return all_data