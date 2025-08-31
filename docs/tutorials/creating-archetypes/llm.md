## Writing an LLM backend for Archetype (New API)

Archetype expects a minimal LLM protocol so different backends can be plugged in.

### Required hooks

- __call__(prompt_inputs: list[str|dict]) -> list[str|dict]
  - Given a list of prompts, return a list of outputs. Each item can be a string or a dict like {"text": "0.73"}.

### Optional hooks

- initialize_llm() -> self
  - If your backend needs initialization (API keys, clients), expose this. Archetype will call it if present.

- prompt(prompt_list: list[str|dict]) -> list[str|dict]
  - Convenience alias. If __call__ is not defined, Archetype will try .prompt.

### Minimal example (MockLLM)

```python
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
        vals = []
        for _ in prompt_list:
            v = self._rng.uniform(self.low, self.high)
            vals.append({"text": f"{v:.3f}"})
        return vals

    def __call__(self, prompt_inputs: List[Union[str, Dict[str, Any]]]):
        return self.prompt(prompt_inputs)
```

### Passing your LLM into Archetype

```python
from agent_torch.core.llm.archetype import Archetype
import agent_torch.core.llm.template as lm

class MyTemplate(lm.Template):
    def __prompt__(self):
        self.prompt_string = "You are {age}, working as {soc_code}."
    def __output__(self):
        return "Return a number in [0, 1]"

llm = MockLLM()
arch = Archetype(prompt=MyTemplate(), llm=llm, n_arch=3)
```

Archetype will call `initialize_llm()` if present, then use your object via `llm(prompt_inputs)` when sampling.


