## Archetype Tutorial (New API)

This tutorial shows how to use the new unified Archetype + Template API. You will:
- Define a Template by extending the class and declaring Variables
- Create an Archetype with your Template and an LLM backend
- Configure external data and ground truth via `configure`
- Sample before and after broadcasting to a population

### 1) Setup

```python
import agent_torch.core.llm.template as lm
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.mock_llm import MockLLM
import agent_torch.populations.astoria as astoria
import pandas as pd
```

### 2) Define a Template

Declare the fields you will reference in the prompt. Names must match your data sources (population/external) exactly.

```python
class WillingnessTemplate(lm.Template):
    system_prompt = "You are evaluating willingness based on job profile and context."

    # Variables (case-sensitive; align with your dataframe column names)
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
        return (
            "Rate your willingness to continue normal activities, respond in [0, 1] binary decision only."
        )
```

Notes:
- You do NOT need to create a `Behavior` object. Archetype manages grouping, memory, and sampling.
- Default grouping uses population attributes referenced in your template. You can override via `configure(group_on=...)`.

### 3) Create an Archetype

```python
llm = MockLLM()  # Replace with your LLM backend if desired
arch = Archetype(prompt=WillingnessTemplate(), llm=llm, n_arch=3)
```

### 4) Configure external data and ground truth

You can drive prompts from an external dataframe, and pass ground-truth targets for optimization/evaluation.

```python
all_jobs_df = pd.read_pickle("job_data_clean.pkl")  # columns: job_title, abilities, work_context, soc_code, ...

# Build a ground-truth list aligned to the external_df rows
gt_csv = pd.read_csv("agent_torch/core/llm/data/ground_truth_willingness_all_soc.csv")
soc_to_val = {str(r['soc_code']): float(r['willingness']) for _, r in gt_csv.iterrows()}
ground_truth_list = [soc_to_val.get(str(row.get('soc_code')), 0.0) for _, row in all_jobs_df.iterrows()]

# Configure matching and grouping
arch.configure(
    external_df=all_jobs_df,
    ground_truth=ground_truth_list,
    match_on="soc_code",  # also sets grouping to 'soc_code' by default
)
```

Tips:
- `match_on` sets how external rows and targets map to groups/agents. If `group_on` is not provided, grouping defaults to `match_on`.
- To group by a different key, pass `group_on="job_title"` (or list of keys) to `configure`.

### 5) Sample before and after broadcast

Before calling `broadcast`, `sample()` runs a single-shot (or over all external rows if provided). After broadcast, it returns a tensor of shape `(n_agents,)`.

```python
# Single-shot preview
arch.sample(print_examples=3)

# Bind a population and sample per-agent
arch.broadcast(population=astoria)
arch.sample(print_examples=3)
```

### 6) Common pitfalls

- Field names are case-sensitive. If your dataframe has `abilities` and `work_context`, use the same casing in your template.
- If `print_examples` shows placeholders like `{abilities}`, ensure the external dataframe is configured and `match_on` aligns to a column present in that dataframe.
- If grouping/targets seem mismatched, set both `match_on` and `group_on` to the same key, or let `group_on` default to `match_on`.


