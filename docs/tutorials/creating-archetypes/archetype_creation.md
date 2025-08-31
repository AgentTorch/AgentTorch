## Creating Archetypes (New API)

This tutorial reflects the new unified Archetype API used in `example_new_api.py`.

### 1) Define a Template with Variables

```python
import agent_torch.core.llm.template as lm

class MyPromptTemplate(lm.Template):
    system_prompt = "You are evaluating willingness based on job profile and context."

    age = lm.Variable(desc="agent age", learnable=True)
    gender = lm.Variable(desc="agent gender", learnable=False)
    soc_code = lm.Variable(desc="job id", learnable=False)
    abilities = lm.Variable(desc="abilities required", learnable=True)
    work_context = lm.Variable(desc="work context", learnable=True)

    def __prompt__(self):
        self.prompt_string = (
            "You are in your {age}'s and are a {gender}. "
            "As a {soc_code}...you have {abilities} and work in {work_context}."
        )

    def __output__(self):
        return "Rate your willingness to continue normal activities, respond in [0, 1] binary decision only."

    def __data__(self):
        # optional: point to a default data source; external_df can override
        import os
        base_dir = os.path.dirname(__file__)
        self.src = os.path.join(base_dir, "../../..", "job_data_clean.pkl")
```

### 2) Create an Archetype

```python
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.mock_llm import MockLLM
import agent_torch.populations.astoria as astoria
import pandas as pd, os

prompt_template = MyPromptTemplate()
llm = MockLLM()
arch = Archetype(prompt=prompt_template, llm=llm, n_arch=3)

base_dir = os.path.dirname(__file__)
jobs_df = pd.read_pickle(os.path.join(base_dir, "../../..", "job_data_clean.pkl"))
gt_csv = pd.read_csv(os.path.join(base_dir, "../../..", "agent_torch/core/llm/test_data/test_data.csv"))
soc_to_val = {str(r["soc_code"]): float(r["willingness"]) for _, r in gt_csv.iterrows()}
ground_truth_list = [soc_to_val.get(str(row.get("soc_code")), 0.0) for _, row in jobs_df.iterrows()]

# preview with a smaller slice before broadcast
arch.configure(external_df=jobs_df, split=2)
arch.sample()
```

### 3) Broadcast to a population and sample

```python
arch.configure(external_df=jobs_df)
arch.broadcast(population=astoria, match_on="soc_code")
arch.sample()  # returns (n_agents,) tensor
```

### 4) Optimize with P3O (optional)

```python
from agent_torch.optim import P3O
opt = P3O(arch.parameters(), archetype=arch)
for _ in range(2):
    arch.sample(print_examples=1)
    opt.step()
    opt.zero_grad()
```

See `docs/tutorials/creating-archetypes/llm.md` for implementing custom LLMs compatible with this API.
