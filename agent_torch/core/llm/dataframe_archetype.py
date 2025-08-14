"""
Simplified DataFrame Archetype using Template System
===================================================

DataFrameArchetype now uses Template objects for clean integration
with the base archetype system. No more separate DataFrame abstractions.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from agent_torch.core.llm.archetype import Archetype, LLMArchetype
from agent_torch.core.llm.template import Template
from agent_torch.core.llm.dataframe_prompt_manager import DataFramePromptManager


class DataFrameArchetype(Archetype):
    """
    Archetype that uses Template objects and external data sources.
    
    Integrates seamlessly with the base Archetype system - no separate abstractions needed.
    Template handles data loading, P3O handles optimization.
    """
    
    def __init__(self, template: Template, n_arch: int = 1):

        super().__init__(n_arch=n_arch)
        self.template = template
        
        if not template.src:
            raise ValueError("Template must have 'src' field pointing to external data")

    def llm(self, llm, user_prompt: str = None, **kwargs):

        # Build template-based user prompt if none provided
        if user_prompt is None:
            user_prompt = self._build_template_prompt()
        
        llm.initialize_llm()
        
        return [
            TemplateLLMArchetype(llm=llm, template=self.template, population=None, n_arch=self.n_arch) 
            for _ in range(self.n_arch)
        ]
    
    def _build_template_prompt(self) -> str:
        """Build user prompt from template sections."""
        sections = []
        if self.template.archetype_data.strip():
            sections.append(self.template.archetype_data)
        if self.template.external_data.strip():
            sections.append(self.template.external_data)
        if self.template.config_data.strip():
            sections.append(self.template.config_data)
        return " ".join(sections)
    
    def enhance_profile(self, agent_id: int, population_data: Dict[str, Any]) -> Dict[str, Any]:
        enhanced_profile = population_data.copy()
        template_data = self.template.get_agent_data(agent_id, population_data)
        enhanced_profile.update(template_data)
        return enhanced_profile


class TemplateLLMArchetype(LLMArchetype):
    """
    LLM Archetype that uses Template system for data-driven prompts.
    
    Extends base LLMArchetype to work with Template objects and external data.
    Handles agent data assembly and P3O integration.
    """
    
    def __init__(self, llm, template: Template, population=None, n_arch: int = 1):
        self.template = template
        self.population = population  # Store population for data access
        self.profile = {}  # Will be set by DataFrameArchetype
        
        user_prompt = self._build_user_prompt()
        
        super().__init__(llm, user_prompt, n_arch)
        
        if hasattr(self.llm, 'output_format') and template.output_format:
            self.llm.output_format = template.output_format
        
        # Template-specific attributes (kept for compatibility)
        self.p3o_enabled = self._has_learnable_fields()
    
    # Removed legacy build_prompt; Template + PromptManager own prompt assembly
    
    # Removed legacy mapping helper; PromptManager/Template handle mapping
        
    def _build_user_prompt(self) -> str:
        """
        Build user prompt from Template sections.
        Converts {field, learnable=True/False} to simple {field} placeholders.
        """
        def convert_to_simple_placeholders(text):
            """Convert {field, learnable=True/False} to {field}."""
            pattern = r'\{([^,}]+)(?:,\s*learnable\s*=\s*(True|False))?\}'
            return re.sub(pattern, r'{\1}', text)
        
        sections = []
        
        if self.template.archetype_data.strip():
            sections.append(convert_to_simple_placeholders(self.template.archetype_data))
        if self.template.external_data.strip():
            sections.append(convert_to_simple_placeholders(self.template.external_data))
        if self.template.config_data.strip():
            sections.append(convert_to_simple_placeholders(self.template.config_data))
            
        return " ".join(sections) if sections else "Rate your response (0.0-1.0):"
    
    def _has_learnable_fields(self) -> bool:
        """Check if template has any learnable fields for P3O."""
        all_text = f"{self.template.archetype_data} {self.template.external_data} {self.template.config_data}"
        fields = self.template.parse_template_fields(all_text)
        return any(is_learnable for _, is_learnable in fields)
    
    def __call__(self, prompt_list, last_k: int = 2):
        """
        Process prompts using Template system and DataFramePromptManager.
        Compatible with both string prompts (base system) and dict prompts (enhanced system).
        
        Args:
            prompt_list: List of prompt strings (base system) OR context dictionaries (enhanced system)
            last_k: Memory context length
            
        Returns:
            List of LLM responses
        """
        if not prompt_list:
            return []
        
        # Base flow expects string prompts; delegate directly
        if isinstance(prompt_list[0], str):
            return self._process_string_prompts(prompt_list, last_k)
        # If dicts are passed inadvertently, convert via Template and continue
        if isinstance(prompt_list[0], dict):
            converted = []
            for i, ctx in enumerate(prompt_list):
                agent_id = ctx.get('agent_id', i)
                filled = self.template.render(
                    agent_id=agent_id,
                    population=self.population,
                    mapping={},
                    config_kwargs=ctx
                )
                converted.append(filled)
            return self._process_string_prompts(converted, last_k)
        return []
    
    def _process_string_prompts(self, prompt_list: List[str], last_k: int) -> List[List[str]]:
        """Process string prompts from base system (master branch compatibility)."""
        # Check if this is a P3O optimization call (will have the P3O marker)
        is_p3o_call = len(prompt_list) == 1 and "[P3O_OPTIMIZATION]" in str(prompt_list[0]) if prompt_list else False
        
        # PromptManager already resolved everything via Template.generate_fully_resolved_prompt()
        # No need to fill again - just use the prompts as-is
        filled_prompt_list = prompt_list
        
        if not is_p3o_call:
            print(f"---LLM RECEIVES {len(filled_prompt_list)} FILLED PROMPTS:---")
            for i, prompt in enumerate(filled_prompt_list[:3]):  # Show first 3
                print(f"         Prompt {i+1}: {prompt}")
        
        clean_prompt_list = []
        for prompt in filled_prompt_list:
            if "[P3O_OPTIMIZATION]" in prompt:
                clean_prompt = prompt.replace("[P3O_OPTIMIZATION] ", "")
                clean_prompt_list.append(clean_prompt)
            else:
                clean_prompt_list.append(prompt)
        
        # Use base LLMArchetype processing for string prompts
        return super().__call__(clean_prompt_list, last_k)
    
    # Removed P3O-specific and direct assembly methods; LLM calls use base flow
    
    def get_p3o_components(self, agent_data_list: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get P3O components for optimization.
        
        Args:
            agent_data_list: List of enhanced agent data dictionaries for P3O
        
        Returns:
            Dictionary with P3O placeholder choices, template, parameters, and agent data
        """
        if not self.p3o_enabled:
            return {}
        
        return {
            'placeholder_choices': self.template.create_p3o_placeholder_choices(),
            'template_string': self.template.create_p3o_template_string(),
            'parameters': self.template.get_slot_parameters(),
            'agent_data_list': agent_data_list or [],
            'non_learnable_template': self._get_non_learnable_template()
        }
    
    def _get_non_learnable_template(self) -> str:
        """Get template with only non-learnable placeholders for pre-filling."""
        # Get learnable fields to avoid including them
        all_text = f"{self.template.archetype_data} {self.template.external_data} {self.template.config_data}"
        fields = self.template.parse_template_fields(all_text)
        learnable_fields = {field_name for field_name, is_learnable in fields if is_learnable}
        
        # Create template string with only non-learnable placeholders
        template_string = self.template.create_p3o_template_string()
        
        # This template can be pre-filled with non-learnable data
        # P3O will handle the {{learnable}} placeholders
        return template_string
    
    def prepare_for_p3o(self, prompt_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare data for P3O optimization instead of direct LLM calls.
        
        Args:
            prompt_list: List of context dictionaries
            
        Returns:
            Dictionary with everything P3O needs for optimization
        """
        if not self.p3o_enabled:
            return {}
        
        # Generate enhanced prompt data using DataFramePromptManager or direct method
        enhanced_agent_data = []
        
        if self.prompt_manager:
            # Use DataFramePromptManager
            for i, context in enumerate(prompt_list):
                agent_id = context.get('agent_id', i)
                context_with_id = {**context, 'agent_id': agent_id}
                prompt_data = self.prompt_manager.generate_prompt_from_data(context_with_id)
                enhanced_agent_data.append(prompt_data)
        else:
            # Direct method
            for i, context in enumerate(prompt_list):
                agent_id = context.get('agent_id', i)
                agent_data = self.template.get_agent_data(agent_id, context)
                enhanced_agent_data.append(agent_data)
        
        # Return P3O components with enhanced agent data
        return self.get_p3o_components(enhanced_agent_data)
    
    def enhance_profile(self, agent_id: int, population_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance agent profile using template's external data (clean system approach).
        
        Args:
            agent_id: Agent identifier
            population_data: Basic population data (age, gender, etc.)
            
        Returns:
            Enhanced agent profile with template data
        """
        # Start with population data
        enhanced_profile = population_data.copy()
        
        # Load template's external data for this agent
        template_data = self.template.get_agent_data(agent_id, population_data)
        
        # Merge template data with population data
        enhanced_profile.update(template_data)
        
        return enhanced_profile
    
    def load_ground_truth(self) -> Optional[pd.DataFrame]:
        """Load ground truth data for P3O optimization."""
        return self.template.load_ground_truth_data()
    
