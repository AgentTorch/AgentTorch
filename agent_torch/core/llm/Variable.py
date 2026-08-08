"""
User-facing Variable descriptor
===============================

Descriptor to declare variables inside Template subclasses.
- Carries metadata like description, learnable flag, and default value
- Stores resolved values per-instance
- When learnable=True, owns a trainable tensor parameter attached to the
  Template instance; no separate Slot class is needed.
"""

from typing import Any, Optional, Tuple, Callable, Dict, List
import torch
import torch.nn as nn


class Variable:
    def __init__(self, desc: Optional[str] = None, learnable: bool = False, default: Any = None,
                 presentations: Optional[List[str]] = None):
        """
        Args:
            desc: Description of what this variable represents
            learnable: Whether this variable's presentation can be optimized
            default: Default value when variable is not set
            presentations: List of format strings for presentation choices.
                          presentations[0] should always be "" (skip)
                          presentations[1+] are user-defined formats with {value} placeholder
                          
        Example:
            skill = lm.Variable(
                desc="Programming expertise",
                learnable=True,
                presentations=[
                    "",                           # Choice 0: Skip
                    "Skill: {value}",            # Choice 1: Formal
                    "Expert in {value}",         # Choice 2: Expertise
                    "{value} experience"         # Choice 3: Casual
                ]
            )
        """
        self.desc = desc
        self.learnable = learnable
        self.default = default
        self._name: Optional[str] = None
        
        # Set up presentation choices
        if presentations is None:
            # Default behavior: binary choice (skip or include with current format)
            self.presentations = [
                "",                    # Choice 0: Skip
                "- {value}: {value}"   # Choice 1: Default format
            ]
        else:
            # User-provided presentations
            if not presentations or presentations[0] != "":
                # Ensure first choice is always "skip"
                presentations = [""] + (presentations or ["- {value}: {value}"])
            self.presentations = presentations
        
        # Number of presentation options for P3O
        self.num_options: int = len(self.presentations)

    def __set_name__(self, owner, name: str):
        self._name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self._name, self.default)

    def __set__(self, instance, value: Any):
        instance.__dict__[self._name] = value

    @property
    def name(self) -> Optional[str]:
        return self._name

    # --- Learnable parameter support (replaces Slot) ---
    def get_parameter(self, instance: Any) -> Optional[nn.Parameter]:
        """Return/create the learnable parameter (logits over options) for this variable on the given instance."""
        if not self.learnable:
            return None
        
        # Auto-discover name if not set
        if self._name is None:
            for name, var in getattr(instance, '_variables', {}).items():
                if var is self:
                    self._name = name
                    break
        
        if self._name is None:
            return None
            
        param_attr = f"__var_param__{self._name}"
        param = getattr(instance, param_attr, None)

        if not isinstance(param, nn.Parameter):
            # Match original experiment: unbiased initialization with torch.zeros()
            # This gives 50/50 probability for binary choices, letting P3O learn naturally
            init = torch.zeros(self.num_options, dtype=torch.float32)
            param = nn.Parameter(init, requires_grad=True)
            setattr(instance, param_attr, param)
        return param

    def get_p3o_choice(self, mapping: Optional[dict] = None) -> Tuple[int, Callable[[int, dict], str]]:
        """Return P3O placeholder choice tuple: (num_options, lambda(category, data) -> str).

        The lambda implements presentation choices:
          0: skip → ""
          1: direct → value
          2: labeled → "{field_name}: value"
          3: contextual → "with value"
          4: descriptive → "The {field_name} is value"
        For non-learnable variables, the lambda always returns the direct value.
        """

        field_name = self._name

        def map_value(raw: Any) -> str:
            # Map numeric categorical values using mapping.json when available
            if mapping and field_name in mapping and isinstance(raw, (int, float)):
                try:
                    idx = int(raw)
                    choices = mapping[field_name]
                    if 0 <= idx < len(choices):
                        return str(choices[idx])
                except Exception:
                    pass
            # Already string or anything else → stringify
            return "" if raw is None else str(raw)

        def fmt(category: int, data: dict) -> str:
            if not field_name:
                return ""
            raw_value = data.get(field_name)
            
            # Handle sparse skill data: if skill is not relevant (0 or missing), always skip
            if field_name != 'soc_code' and field_name != 'job_title':
                # For skill fields, check if this skill is relevant to this job
                if raw_value is None or raw_value == 0 or raw_value == '0':
                    return ""  # Skill not relevant for this job, always skip
            
            if raw_value is None:
                return ""
            value = map_value(raw_value)

            # If not learnable, always use the first non-skip presentation
            if not self.learnable:
                if len(self.presentations) > 1:
                    return self.presentations[1].format(value=value)
                return value

            # For skills: only render if choice=1 AND skill is relevant (value=1) 
            if field_name != 'soc_code' and field_name != 'job_title':
                if category == 0:
                    return ""  # P3O chose to skip this skill
                elif category == 1 and raw_value == 1:
                    # P3O chose to include AND skill is relevant for this job
                    if 1 < len(self.presentations):
                        return self.presentations[1].format(value=field_name.replace('_', ' ').title())
                    return f"- {field_name.replace('_', ' ').title()}: {field_name.replace('_', ' ').title()}"
                else:
                    return ""  # Skill not relevant or P3O chose to skip

            # For non-skill fields (soc_code, etc.), use normal presentation logic
            if 0 <= category < len(self.presentations):
                presentation = self.presentations[category]
                if presentation == "":
                    return ""
                return presentation.format(value=value)
            
            # Fallback to last presentation if category is out of range
            if self.presentations:
                last_presentation = self.presentations[-1]
                return last_presentation.format(value=value) if last_presentation else ""
            
            return value

        return self.num_options, fmt

    # --- DSPy Conversion Utilities ---
    @classmethod
    def from_dspy_field(cls, field_name: str, field_annotation, dspy_field, **kwargs) -> 'Variable':
        """Convert a DSPy InputField or OutputField to an lm.Variable.
        
        Args:
            field_name: Name of the field in the DSPy signature
            field_annotation: Type annotation (e.g., str, JobMetrics)
            dspy_field: The dspy.InputField() or dspy.OutputField() instance
            **kwargs: Additional Variable constructor arguments
            
        Returns:
            Variable instance configured for use in AgentTorch templates
            
        Example:
            # From DSPy signature:
            # job_info: str = dspy.InputField(desc="Job description")
            
            var = lm.Variable.from_dspy_field(
                "job_info", str, dspy.InputField(desc="Job description"),
                learnable=True  # Make it optimizable
            )
        """
        # Extract description from DSPy field
        desc = getattr(dspy_field, 'desc', None) or f"Converted from DSPy field: {field_name}"
        
        # InputFields are typically learnable (content we want to optimize)
        # OutputFields are typically not learnable (LLM generates them)
        default_learnable = 'InputField' in str(type(dspy_field))
        learnable = kwargs.pop('learnable', default_learnable)
        
        # Create Variable with DSPy metadata
        return cls(
            desc=desc,
            learnable=learnable,
            default=kwargs.pop('default', None),
            **kwargs
        )
    
    @classmethod
    def from_dspy_signature(cls, signature_class) -> Dict[str, 'Variable']:
        """Convert an entire DSPy Signature to a dictionary of lm.Variables.
        
        Args:
            signature_class: A DSPy Signature class
            
        Returns:
            Dictionary mapping field names to Variable instances
            
        Example:
            class JobSignature(dspy.Signature):
                job_info: str = dspy.InputField(desc="Job skills")
                prediction: JobMetrics = dspy.OutputField(desc="Predictions")
            
            variables = lm.Variable.from_dspy_signature(JobSignature)
            # Returns: {"job_info": Variable(...), "prediction": Variable(...)}
        """
        import inspect
        variables = {}
        
        # Get signature fields
        if hasattr(signature_class, '__annotations__'):
            for field_name, field_type in signature_class.__annotations__.items():
                if hasattr(signature_class, field_name):
                    dspy_field = getattr(signature_class, field_name)
                    # Skip non-field attributes
                    if hasattr(dspy_field, 'desc') or 'Field' in str(type(dspy_field)):
                        variables[field_name] = cls.from_dspy_field(
                            field_name, field_type, dspy_field
                        )
        
        return variables

    # --- Helpers for P3O optimization over Variable options ---
    def sample_index(self, instance: Any) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample a presentation index using the instance-bound logits.

        Returns (sampled_index, log_prob, entropy).
        For non-learnable variables, returns (1, 0.0, 0.0) which corresponds to direct value.
        """
        if not self.learnable:
            return 1, torch.tensor(0.0), torch.tensor(0.0)
        logits = self.get_parameter(instance)
        if logits is None:
            return 1, torch.tensor(0.0), torch.tensor(0.0)

        probs = torch.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample()
        return int(idx.item()), dist.log_prob(idx), dist.entropy()

    def get_probabilities(self, instance: Any) -> torch.Tensor:
        """Return softmax probabilities over presentation options for this variable."""
        if not self.learnable:
            probs = torch.zeros(self.num_options, dtype=torch.float32)
            if self.num_options > 1:
                probs[1] = 1.0  # direct value
            return probs
        logits = self.get_parameter(instance)
        return torch.softmax(logits, dim=0) if logits is not None else torch.zeros(self.num_options, dtype=torch.float32)

    def get_best_index(self, instance: Any) -> int:
        """Return argmax option index based on current probabilities."""
        if not self.learnable:
            return 1
        probs = self.get_probabilities(instance)
        return int(torch.argmax(probs).item()) if probs.numel() > 0 else 1

