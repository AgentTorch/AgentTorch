"""
Simple Mock LLM for testing the unified Archetype API.

Minimal protocol:
 - initialize_llm()  # optional
 - __call__(prompt_inputs) -> list[dict|str]  # required
 - prompt(prompt_list) -> list[dict|str]      # optional adapter
"""

from typing import List, Union, Dict, Any
import random


class MockLLM:
    def __init__(self, low: float = 0.1, high: float = 0.9, seed: int | None = None):
        self.low = float(low)
        self.high = float(high)
        self._rng = random.Random(seed) if seed is not None else random

    def initialize_llm(self):
        return self

    def prompt(self, prompt_list: List[Union[str, Dict[str, Any]]]):
        # Accept both raw strings and dicts with chat_history; ignore history in mock
        vals = []
        for _ in prompt_list:
            v = self._rng.uniform(self.low, self.high)
            vals.append({"text": f"{v:.3f}"})
        return vals

    def __call__(self, prompt_inputs: List[Union[str, Dict[str, Any]]]):
        return self.prompt(prompt_inputs)

