#!/usr/bin/env python3
"""
AgentTorch P3O Template System
==============================

Centralized template management for P3O prompt optimization.
Provides built-in templates and utilities for creating custom templates.

Design Principles:
- Templates focus on DECISION-MAKING, not model-specific context
- Model-agnostic templates work across COVID, economics, social models
- System automatically injects model-specific context
- Easy for users to create and customize templates

Usage Examples:
--------------

# Use built-in template
template = Template.COVID_WILLINGNESS

# Create custom template
my_template = Template.create_custom(
    name="my_decision", 
    template="{{context}}\n\n{{decision}}",
    placeholders={
        "context": ["You are {job_name}...", "As {job_name}..."],
        "decision": ["How willing are you? (0.0-1.0)", "Rate willingness:"]
    }
)

# Validate template
Template.validate(my_template)
"""

from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass
import json
import re


@dataclass
class TemplateDefinition:
    """
    Structure for a P3O template definition.
    
    Attributes:
        name: Template identifier
        template: Template string with {{placeholder}} markers
        placeholders: Dict mapping placeholder names to lists of variations
        description: Human-readable description of template purpose
        output_format: Expected output format configuration
    """
    name: str
    template: str
    placeholders: Dict[str, List[str]]
    description: str = ""
    output_format: Dict[str, Any] = None
    
    def __post_init__(self):
        """Set default output format if not provided."""
        if self.output_format is None:
            self.output_format = {
                "type": "float",
                "range": [0.0, 1.0],
                "patterns": [
                    r'\(([0-9]*\.?[0-9]+)\)',  # (0.45)
                    r'([0-9]*\.?[0-9]+)$',     # Final number
                    r'([0-9]*\.?[0-9]+)'       # Any decimal
                ]
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for P3O backend."""
        return {
            "template": self.template,
            "placeholders": self.placeholders,
            "output_format": self.output_format
        }
    
    def extract_required_fields(self) -> Dict[str, Set[str]]:
        """
        Smart field extraction based on standardized 3-section structure.
        
        Maps template sections to their corresponding data sources:
        - personal_context → population_fields (from mapping.json)
        - agent_context → external_fields (from mapping.json external_sources)
        - decision_prompt → context_fields (from simulation/environment)
        
        Returns:
            Dictionary with categorized field sets:
            {
                'population_fields': set(['age', 'gender', 'ethnicity', ...]),
                'external_fields': set(['job_name', 'Skills', 'hobby_name', ...]),
                'context_fields': set(['covid_cases', 'unemployment_rate', ...])
            }
        """
        field_categories = {
            'population_fields': set(),
            'external_fields': set(), 
            'context_fields': set()
        }
        
        # Smart mapping: Template sections to data sources (no hardcoding)
        section_mapping = {
            'personal_context': 'population_fields',
            'agent_context': 'external_fields', 
            'decision_prompt': 'context_fields'
        }
        
        # Extract fields from each template section
        for placeholder_name, variations in self.placeholders.items():
            if placeholder_name in section_mapping:
                category = section_mapping[placeholder_name]
                for variation in variations:
                    fields = re.findall(r'\{(\w+)(?:[^}]*)?\}', variation)
                    field_categories[category].update(fields)
        
        return field_categories
    
    def validate_template_structure(self) -> None:
        """
        Structure validation: Ensure template follows required 3-section structure.
        
        Raises:
            ValueError: If template is missing required sections
        """
        required_sections = {'personal_context', 'agent_context', 'decision_prompt'}
        template_sections = set(self.placeholders.keys())
        
        if not required_sections.issubset(template_sections):
            missing = required_sections - template_sections
            raise ValueError(
                f"Template '{self.name}' missing required sections: {missing}. "
                f"All templates must have: personal_context, agent_context, decision_prompt"
            )
    
    def assemble_agent_data(self, agent_idx: int, mapping_manager, simulation_context: Dict[str, Any], 
                           environment_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assemble all data needed for this agent's prompt from multiple sources.
        
        🧠 SMART ASSEMBLY: Uses template structure to guide data resolution
        1. Validates template structure (3-section requirement)
        2. Extracts categorized field requirements
        3. Delegates to MappingManager for smart assembly
        
        Args:
            agent_idx: Agent index
            mapping_manager: MappingManager instance for all data resolution
            simulation_context: Model-specific simulation state data
            environment_context: Model-specific environment data
            
        Returns:
            Complete agent data dictionary for prompt generation
        """
        try:
            # Validate template structure first
            self.validate_template_structure()
            
            # Delegate to MappingManager for smart, categorized assembly
            return mapping_manager.get_template_data(
                agent_id=agent_idx,
                template=self,
                simulation_context=simulation_context,
                environment_context=environment_context
            )
        except Exception as e:
            print(f"Error: Template: Error assembling data for agent {agent_idx}: {e}")
            raise
    
    def extract_output_score(self, text: str) -> float:
        """
        Extract numerical score from LLM text output based on template's output format.
        
        Args:
            text: LLM response text
            
        Returns:
            Extracted numerical score normalized to [0.0, 1.0] range
        """
        import re
        
        patterns = self.output_format.get("patterns", [r'([0-9]*\.?[0-9]+)'])
        output_range = self.output_format.get("range", [0.0, 1.0])
        output_type = self.output_format.get("type", "float")
        
        # Try each pattern to find a valid score
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    try:
                        if output_type == "int":
                            score = int(match)
                        else:
                            score = float(match)
                        
                        # Normalize to [0.0, 1.0] range
                        min_val, max_val = output_range
                        if min_val <= score <= max_val:
                            # Normalize to 0-1 range for internal use
                            normalized = (score - min_val) / (max_val - min_val)
                            return float(normalized)
                    except (ValueError, IndexError, ZeroDivisionError):
                        continue
        
        # Fallback: return neutral score if no valid score found
        return 0.5


class Template:
    """
    Central template registry and management system.
    
    Stores built-in templates and provides utilities for creating,
    validating, and managing custom templates for P3O optimization.
    """
    
    # ========================================================================
    # BUILT-IN TEMPLATES
    # ========================================================================
    
    COVID_WILLINGNESS = TemplateDefinition(
        name="covid_willingness",
        template="{{personal_context}} {{agent_context}} {{decision_prompt}}",
        placeholders={
            "personal_context": [
                # Index 0: Default/primary personal context (P3O will select this with uniform probabilities)
                "You are a {age}-year-old {gender} {ethnicity} living in {area} with a household of {household_size}.",
                # Additional variations for optimization
                "As a {age}-year-old {ethnicity} {gender} in {area} ({household_size} in household), you consider your personal risk factors.", 
                "Your personal profile: {age} years old, {gender}, {ethnicity} background, living in {area} with {household_size} people.",
                "You are {age} years old ({gender}, {ethnicity}) living in {area} with {household_size} household members."
            ],
            "agent_context": [
                # Job context (external data sources only)
                "You work as a {job_name} with key skills: {Skills}. Your main tasks: {Tasks}.",
                "As a {job_name}, your daily work involves {Tasks} and requires {Abilities} in {WorkContext}.",
                "Your profession: {job_name}. Skills: {Skills}. Work environment: {WorkContext}.",
                "Working as a {job_name}, you have expertise in {Skills} and work in {WorkContext} conditions."
            ],
            "decision_prompt": [
                # Only context fields (simulation/environment data)
                "Given COVID-19 cases at {covid_cases:,} and {unemployment_rate:.1%} unemployment, how willing are you to continue working outside your home? (0.0-1.0)",
                "With current conditions (COVID: {covid_cases:,} cases, unemployment: {unemployment_rate:.1%}), rate your willingness to keep working (0.0-1.0):",
                "Considering the pandemic (COVID cases: {covid_cases:,}, unemployment: {unemployment_rate:.1%}), what's your work willingness? (0.0-1.0)",
                "Given health/economic conditions (COVID: {covid_cases:,}, unemployment: {unemployment_rate:.1%}), how willing are you to maintain work? (0.0-1.0)"
            ]
        },
        description="COVID work willingness template - follows 3-section structure for clear data source separation"
    )

    COVID_HOBBY_WILLINGNESS = TemplateDefinition(
        name="covid_hobby_willingness",
        template="{{personal_context}} {{agent_context}} {{decision_prompt}}",
        placeholders={
            "personal_context": [
                # Index 0: Default/primary personal context (P3O will select this with uniform probabilities)
                "You are a {age}-year-old {gender} {ethnicity} living in {area} with a household of {household_size}.",
                "As a {age}-year-old {ethnicity} {gender} in {area} ({household_size} in household), you consider your personal risk factors.", 
                "Your personal profile: {age} years old, {gender}, {ethnicity} background, living in {area} with {household_size} people.",
                "You are {age} years old ({gender}, {ethnicity}) living in {area} with {household_size} household members."
            ],
            "agent_context": [
                "You work as a {job_name} with key skills: {Skills}. Your main tasks: {Tasks}. Outside of work, you enjoy {hobby_name} as your hobby, which involves {HobbyActivities} and requires {HobbySkills}.",
                "As a {job_name}, your daily work involves {Tasks} and requires {Abilities} in {WorkContext}. Your hobby is {hobby_name}, a {Time_Commitment} activity that involves {HobbyActivities} and uses {Equipment}.",
                "Your profession: {job_name}. Skills: {Skills}. Work environment: {WorkContext}. You practice {hobby_name} as a hobby, which is a {Cost_Level} cost activity with {Social_Aspect} social aspects.",
                "Working as a {job_name}, you have expertise in {Skills} and work in {WorkContext} conditions. In your free time, you do {hobby_name}, a {Indoor_Outdoor} activity that involves {HobbyActivities}."
            ],
            "decision_prompt": [
                "Given COVID-19 cases at {covid_cases:,} and {unemployment_rate:.1%} unemployment, how willing are you to continue working outside your home? (0.0-1.0)",
                "With current conditions (COVID: {covid_cases:,} cases, unemployment: {unemployment_rate:.1%}), rate your willingness to keep working (0.0-1.0):",
                "Considering the pandemic (COVID cases: {covid_cases:,}, unemployment: {unemployment_rate:.1%}), what's your work willingness? (0.0-1.0)",
                "Given health/economic conditions (COVID: {covid_cases:,}, unemployment: {unemployment_rate:.1%}), how willing are you to maintain work? (0.0-1.0)"
            ]
        },
        description="COVID work willingness template - follows 3-section structure for clear data source separation"
    )
    
    ECONOMIC_DECISION = TemplateDefinition(
        name="economic_decision", 
        template="{{personal_context}} {{agent_context}} {{decision_prompt}}",
        placeholders={
            "personal_context": [
                "You are a {age}-year-old {gender} {ethnicity} living in {area} with {household_size} household members.",
                "As a {age}-year-old {ethnicity} {gender} in {area}, you manage household finances for {household_size} people.",
                "Your personal profile: {age} years old, {gender}, {ethnicity} background, residing in {area}."
            ],
            "agent_context": [
                "You work as a {job_name} with key skills in {Skills}. Your main professional tasks include {Tasks}.",
                "As a {job_name}, you have expertise in {Skills} and work in {WorkContext} conditions.",
                "Your profession: {job_name}. Skills: {Skills}. Work environment: {WorkContext}."
            ],
            "decision_prompt": [
                "Given current market conditions with {inflation_rate:.1%} inflation and {interest_rate:.1%} interest rates, how confident are you in making major economic decisions? (0.0-1.0)",
                "With economic indicators showing {gdp_growth:.1%} GDP growth and {unemployment_rate:.1%} unemployment, rate your comfort level with financial decision-making (0.0-1.0):",
                "Considering current economic conditions (inflation: {inflation_rate:.1%}, unemployment: {unemployment_rate:.1%}), what's your confidence in economic choices? (0.0-1.0)"
            ]
        },
        description="Economic decision confidence template following 3-section structure for clear data source separation"
    )
    
    SOCIAL_BEHAVIOR = TemplateDefinition(
        name="social_behavior",
        template="{{personal_context}} {{agent_context}} {{decision_prompt}}",
        placeholders={
            "personal_context": [
                "You are a {age}-year-old {gender} {ethnicity} living in {area} with {household_size} household members.",
                "As a {age}-year-old {ethnicity} {gender} in {area}, you are part of a {household_size}-person household.",
                "Your personal background: {age} years old, {gender}, {ethnicity}, residing in {area}."
            ],
            "agent_context": [
                "You work as a {job_name} with expertise in {Skills}. Your main tasks involve {Tasks}.",
                "As a {job_name}, your daily work requires {Abilities} in {WorkContext} environments.",
                "Your profession: {job_name}. Skills: {Skills}. Work setting: {WorkContext}."
            ],
            "decision_prompt": [
                "Given current community conditions with {community_events} upcoming events and {social_restrictions} social guidelines, how likely are you to participate in community activities? (0.0-1.0)",
                "With {volunteer_opportunities} volunteer opportunities and {social_safety_level} safety measures in place, rate your willingness to engage in social community events (0.0-1.0):",
                "Considering current social conditions (events: {community_events}, safety: {social_safety_level}), what's your motivation for community participation? (0.0-1.0)"
            ]
        },
        description="Social engagement and participation template following 3-section structure for clear data source separation"
    )
    
    RISK_ASSESSMENT = TemplateDefinition(
        name="risk_assessment",
        template="{{personal_context}} {{agent_context}} {{decision_prompt}}",
        placeholders={
            "personal_context": [
                "You are a {age}-year-old {gender} {ethnicity} living in {area} with {household_size} household members.",
                "As a {age}-year-old {ethnicity} {gender} in {area}, you support a household of {household_size} people.",
                "Your personal situation: {age} years old, {gender}, {ethnicity}, residing in {area}."
            ],
            "agent_context": [
                "You work as a {job_name} with specialized skills in {Skills}. Your main responsibilities include {Tasks}.",
                "As a {job_name}, your work requires {Abilities} and takes place in {WorkContext} conditions.",
                "Your professional expertise: {job_name}. Key skills: {Skills}. Work environment: {WorkContext}."
            ],
            "decision_prompt": [
                "Given current risk factors with {safety_incidents} recent safety incidents and {risk_level} risk assessments, how comfortable are you proceeding with work activities? (0.0-1.0)",
                "With workplace conditions showing {hazard_reports} hazard reports and {safety_measures} safety protocols in place, rate your confidence in safely performing your job duties (0.0-1.0):",
                "Considering current risk environment (incidents: {safety_incidents}, measures: {safety_measures}), what's your willingness to engage in work tasks? (0.0-1.0)"
            ]
        },
        description="Risk-based decision making template following 3-section structure for clear data source separation"
    )
    
    BIRD_MIGRATION_DECISION = TemplateDefinition(
        name="bird_migration_decision",
        template="{{personal_context}} {{agent_context}} {{decision_prompt}}",
        placeholders={
            "personal_context": [
                "You are a {age_class} {species} located in the {region} region.",
                "As a {age_class} member of the {species} species from the {region}, you are considering migration.",
                "This {age_class} {species} resides in {region} and is assessing environmental cues."
            ],
            "agent_context": [
                "You have been tracking seasonal patterns and food availability in your territory.",
                "Your migration experience includes knowledge of routes and timing for your species.",
                "You possess instinctual knowledge about optimal migration conditions and pathways."
            ],
            "decision_prompt": [
                "Current conditions: daylength {daylength_hr} hours, temperature {air_temp_c}°C, fat reserves {fat_score}/5, food availability {food_avail_index}, wind speed {wind_speed_ms} m/s. Based on these environmental cues, should you begin migration? (0.0 - 1.0)",
                "Environmental assessment: {daylength_hr} hr daylight, {air_temp_c}°C temperature, fat score {fat_score}/5, food index {food_avail_index}, wind {wind_speed_ms} m/s. Rate the likelihood of initiating migration now. (0.0 to 1.0)",
                "Migration decision factors: daylight {daylength_hr}hr, temperature {air_temp_c}°C, energy reserves {fat_score}/5, food {food_avail_index}, wind conditions {wind_speed_ms} m/s. Should you migrate at this point? Provide a probability (0.0 to 1.0)."
            ]
        },
        description="Bird migration decision template following 3-section structure for clear data source separation"
    )
    
    # Example template with different output format (0-100 scale)
    CONFIDENCE_PERCENTAGE = TemplateDefinition(
        name="confidence_percentage",
        template="{{personal_context}} {{agent_context}} {{decision_prompt}}",
        placeholders={
            "personal_context": [
                "You are a {age}-year-old {gender} {ethnicity} living in {area} with {household_size} household members.",
                "As a {age}-year-old {ethnicity} {gender} in {area}, you consider your personal situation.",
                "Your profile: {age} years old, {gender}, {ethnicity}, residing in {area}."
            ],
            "agent_context": [
                "You work as a {job_name} with expertise in {Skills}. Your main tasks include {Tasks}.",
                "As a {job_name}, you have skills in {Skills} and work in {WorkContext} conditions.",
                "Your profession: {job_name}. Key abilities: {Abilities}. Work environment: {WorkContext}."
            ],
            "decision_prompt": [
                "Given current conditions (COVID: {covid_cases:,} cases, unemployment: {unemployment_rate:.1%}), what is your confidence percentage in continuing your work routine? Answer as a number from 0 to 100:",
                "With these conditions (COVID cases: {covid_cases:,}, unemployment: {unemployment_rate:.1%}), rate your work confidence on a scale of 0-100%:",
                "Considering the situation (COVID: {covid_cases:,}, unemployment: {unemployment_rate:.1%}), provide your confidence level as a percentage (0-100):"
            ]
        },
        description="Confidence assessment template using 0-100 percentage scale",
        output_format={
            "type": "float",
            "range": [0.0, 100.0],
            "patterns": [
                r'([0-9]+)%',                    # 85%
                r'([0-9]+)\s*percent',           # 85 percent
                r'([0-9]+)$',                    # 85 (at end)
                r'([0-9]*\.?[0-9]+)'            # Any number
            ]
        }
    )
    
    # ========================================================================
    # TEMPLATE REGISTRY AND UTILITIES
    # ========================================================================
    
    @classmethod
    def get_builtin_templates(cls) -> Dict[str, TemplateDefinition]:
        """Get all built-in templates as a dictionary."""
        return {
            "covid_willingness": cls.COVID_WILLINGNESS,
            "covid_hobby_willingness": cls.COVID_HOBBY_WILLINGNESS,
            "economic_decision": cls.ECONOMIC_DECISION,
            "social_behavior": cls.SOCIAL_BEHAVIOR,
            "risk_assessment": cls.RISK_ASSESSMENT,
            "bird_migration_decision": cls.BIRD_MIGRATION_DECISION,
            "confidence_percentage": cls.CONFIDENCE_PERCENTAGE
        }
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """List all available built-in template names."""
        return list(cls.get_builtin_templates().keys())
    
    @classmethod
    def get_template(cls, name: str) -> Optional[TemplateDefinition]:
        """Get a built-in template by name."""
        templates = cls.get_builtin_templates()
        return templates.get(name)
    
    @classmethod
    def create_custom(
        cls,
        name: str,
        template: str,
        placeholders: Dict[str, List[str]],
        description: str = "Custom user template"
    ) -> TemplateDefinition:
        """
        Create a custom template with validation.
        
        Args:
            name: Template identifier
            template: Template string with {{placeholder}} markers
            placeholders: Dict mapping placeholder names to variation lists
            description: Template description
            
        Returns:
            Validated TemplateDefinition
            
        Raises:
            ValueError: If template is invalid
        """
        # Create template definition
        template_def = TemplateDefinition(
            name=name,
            template=template,
            placeholders=placeholders,
            description=description
        )
        
        # Validate before returning
        cls.validate(template_def)
        return template_def
    
    @classmethod
    def validate(cls, template: TemplateDefinition) -> bool:
        """
        Validate a template definition.
        
        Args:
            template: Template to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If template is invalid
        """
        # Check required fields
        if not template.name:
            raise ValueError("Template name is required")
        if not template.template:
            raise ValueError("Template string is required")
        if not template.placeholders:
            raise ValueError("Template placeholders are required")
        
        # Extract placeholders from template string
        template_placeholders = set(re.findall(r'\{\{(\w+)\}\}', template.template))
        defined_placeholders = set(template.placeholders.keys())
        
        # Check placeholder consistency
        missing_placeholders = template_placeholders - defined_placeholders
        if missing_placeholders:
            raise ValueError(f"Template uses undefined placeholders: {missing_placeholders}")
        
        extra_placeholders = defined_placeholders - template_placeholders
        if extra_placeholders:
            raise ValueError(f"Unused placeholder definitions: {extra_placeholders}")
        
        # Validate placeholder variations
        for name, variations in template.placeholders.items():
            if not isinstance(variations, list) or len(variations) == 0:
                raise ValueError(f"Placeholder '{name}' must have at least one variation")
            
            for i, variation in enumerate(variations):
                if not isinstance(variation, str):
                    raise ValueError(f"Placeholder '{name}' variation {i} must be a string")
        
        return True
    
    @classmethod
    def resolve_template(cls, template_input: Union[str, Dict[str, Any], TemplateDefinition]) -> TemplateDefinition:
        """
        Resolve template input to TemplateDefinition.
        
        Args:
            template_input: Template name (str), dict, or TemplateDefinition
            
        Returns:
            Resolved TemplateDefinition
            
        Raises:
            ValueError: If template cannot be resolved
        """
        if isinstance(template_input, TemplateDefinition):
            cls.validate(template_input)
            return template_input
            
        elif isinstance(template_input, str):
            # Built-in template name
            template = cls.get_template(template_input)
            if template is None:
                available = cls.list_templates()
                raise ValueError(f"Unknown built-in template '{template_input}'. Available: {available}")
            return template
            
        elif isinstance(template_input, dict):
            # Custom template dictionary
            if "template" not in template_input or "placeholders" not in template_input:
                raise ValueError("Custom template dict must have 'template' and 'placeholders' keys")
            
            return cls.create_custom(
                name=template_input.get("name", "custom"),
                template=template_input["template"],
                placeholders=template_input["placeholders"],
                description=template_input.get("description", "Custom template")
            )
            
        else:
            raise TypeError(f"Template input must be str, dict, or TemplateDefinition, got {type(template_input)}")
    
    @classmethod
    def print_template_info(cls, template: Union[str, TemplateDefinition]) -> None:
        """Print detailed information about a template."""
        if isinstance(template, str):
            template = cls.get_template(template)
            if template is None:
                print(f"Error: Template '{template}' not found")
                return
        
        print(f"   TEMPLATE: {template.name}")
        print(f"   Description: {template.description}")
        print(f"   Structure: {template.template}")
        print(f"   Placeholders: {len(template.placeholders)}")
        
        for name, variations in template.placeholders.items():
            print(f"     • {name}: {len(variations)} variations")
            for i, variation in enumerate(variations):
                preview = variation[:50] + "..." if len(variation) > 50 else variation
                print(f"       {i+1}. {preview}")
        print()
    
    @classmethod
    def print_all_templates(cls) -> None:
        """Print information about all built-in templates."""
        print("AVAILABLE P3O TEMPLATES")
        print("=" * 50)
        
        templates = cls.get_builtin_templates()
        for name, template in templates.items():
            cls.print_template_info(template)


# Convenience functions for backward compatibility
def get_builtin_templates() -> Dict[str, Dict[str, Any]]:
    """Get built-in templates in dictionary format (backward compatibility)."""
    templates = Template.get_builtin_templates()
    return {name: tmpl.to_dict() for name, tmpl in templates.items()} 