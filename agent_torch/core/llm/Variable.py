"""
User-facing Variable descriptor
===============================

Descriptor to declare variables inside Template subclasses.
- Carries metadata like description, learnable flag, and default value
- Stores resolved values per-instance
- When learnable=True, owns a trainable tensor parameter attached to the
  Template instance; no separate Slot class is needed.
"""

from typing import Any, Optional
import torch
import torch.nn as nn


class Variable:
    def __init__(self, desc: Optional[str] = None, learnable: bool = False, default: Any = None):
        self.desc = desc
        self.learnable = learnable
        self.default = default
        self._name: Optional[str] = None

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
        """Return/create the learnable parameter for this variable on the given instance."""
        if not self.learnable or self._name is None:
            return None
        param_attr = f"__var_param__{self._name}"
        param = getattr(instance, param_attr, None)
        if not isinstance(param, nn.Parameter):
            # Simple scalar parameter; can be extended to multi-choice logits if needed
            param = nn.Parameter(torch.zeros(1), requires_grad=True)
            setattr(instance, param_attr, param)
        return param

    def get_p3o_choice(self, mapping: Optional[dict] = None):
        """Return a placeholder choice tuple for P3O compatibility.

        Returns (num_choices, formatter_lambda). We keep a minimal implementation
        that simply returns the raw data value for this variable.
        """
        def fmt(choice_index: int, data: dict) -> str:
            value = data.get(self._name, "") if self._name else ""
            return str(value)
        return 1, fmt


