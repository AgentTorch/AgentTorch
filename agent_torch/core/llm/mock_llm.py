"""
Simple Mock LLM for testing the unified Archetype API.

Minimal protocol:
 - initialize_llm()  # optional
 - __call__(prompt_inputs) -> list[dict|str]  # required
 - prompt(prompt_list) -> list[dict|str]      # optional adapter
"""

from typing import List, Union, Dict, Any


class MockLLM:
    def __init__(self, value: float = 0.7):
        self.fixed_value = value

    def initialize_llm(self):
        return self

    def prompt(self, prompt_list: List[Union[str, Dict[str, Any]]]):
        # Accept both raw strings and dicts with chat_history; ignore history in mock
        return [{"text": str(self.fixed_value)} for _ in prompt_list]

    def __call__(self, prompt_inputs: List[Union[str, Dict[str, Any]]]):
        return self.prompt(prompt_inputs)


