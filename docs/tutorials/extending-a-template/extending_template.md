## Extending a Template (Unified API)

This guide shows how to create a class-based Template that works with the unified `Archetype` API. A Template controls:
- Variables (learnable and non-learnable)
- How the prompt is constructed (`__prompt__`)
- What the LLM should output (`__output__`)
- Optional data sourcing (`__data__`)

### Minimal Template

```python
import agent_torch.core.llm.template as lm
import os

class MyPromptTemplate(lm.Template):
    # Optional system message
    system_prompt = "You are evaluating willingness based on job profile and context."

    # Declare Variables (replace Slots)
    age = lm.Variable(desc="agent age", learnable=True)
    gender = lm.Variable(desc="agent gender", learnable=False, default="unknown")
    soc_code = lm.Variable(desc="job id", learnable=False)
    abilities = lm.Variable(desc="abilities required", learnable=True)
    work_context = lm.Variable(desc="work context", learnable=True)

    def __prompt__(self):
        # Build the prompt string using placeholders for declared Variables
        self.prompt_string = (
            "You are in your {age}'s and are a {gender}. "
            "As a {soc_code}...you have {abilities} and work in {work_context}."
        )

    def __output__(self):
        # Instruction to the LLM on how to respond
        return "Return a number in [0, 1]"

    def __data__(self):
        # Optional: provide a default data source for this template
        # Archetype.configure(external_df=...) can override this
        base_dir = os.path.dirname(__file__)
        self.src = os.path.join(base_dir, "../../..", "soc_external_df.pkl")
```

Notes
- Placeholders are field names (`{age}`, `{soc_code}`), not `{self.age}`.
- `learnable=True` Variables expose parameters through `arch.parameters()`.
- `default` provides a fallback value if the field isn’t found in data.

### Grouping and Matching (optional)

Templates can guide how agents are grouped for population-wide sampling.

```python
class MyPromptTemplate(lm.Template):
    # ... variables and hooks ...
    grouping_logic = "soc_code"            # or ["soc_code", "age"] for composite

tpl = MyPromptTemplate()
tpl.grouping_logic = "soc_code"            # can be set at runtime as well
```

During `broadcast(population, match_on=...)`, `match_on` defines how population keys line up with your external data. If `grouping_logic` is not set, it falls back to learnable fields.

### Using your Template with Archetype

```python
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.mock_llm import MockLLM
import agent_torch.populations.astoria as astoria
import pandas as pd, os

tpl = MyPromptTemplate()
llm = MockLLM()
arch = Archetype(prompt=tpl, llm=llm, n_arch=3)

# Attach external data (DataFrame) to drive prompt rendering
external_df = pd.read_pickle(os.path.join(os.getcwd(), "soc_external_df.pkl"))
arch.configure(external_df=external_df, split=3)  # preview with 3 rows

# Single-shot (pre-broadcast); prints prompts/responses if verbose
arch.sample(verbose=True)

# Population-wide sampling
arch.configure(external_df=external_df)          # full data
arch.broadcast(population=astoria, match_on="soc_code")
outputs = arch.sample(verbose=False)             # tensor of shape (n_agents,)
```

### Learnable Variables and optimization

```python
params = list(arch.parameters())  # tensors from Variables with learnable=True

from agent_torch.optim import P3O
# ground_truth_list aligned to external_df rows
opt = P3O(archetype=arch, ground_truth=ground_truth_list, verbose=True)
arch.sample()     # populate last group outputs/keys
opt.step()
opt.zero_grad()
```

That’s it—declare Variables, implement the hooks you need, and run through `Archetype`.


