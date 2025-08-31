## Archetype (Unified API)

This guide is focused solely on the `Archetype` class: how to construct it, configure it, broadcast a population, and sample decisions. The same interface works whether your prompt is a plain string or a class-based Template.

### What is `Archetype`?

`Archetype` is a unified LLM runner that:
- Accepts `prompt` as either a string or a `Template` instance
- Manages one or more underlying LLM “archetypes” (`n_arch`)
- Optionally binds a population and handles grouped prompting
- Exposes learnable parameters when the prompt is a `Template`

### Constructor

```python
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.mock_llm import MockLLM

# String prompt
arch = Archetype(prompt="You are a helpful assistant.", llm=MockLLM(), n_arch=3)

# Or, with a Template (defined elsewhere)
# from my_templates import MyPromptTemplate
# tpl = MyPromptTemplate()
# arch = Archetype(prompt=tpl, llm=MockLLM(), n_arch=3)
```

### configure(external_df=None, split=None)

Attach an external DataFrame used to render prompts (for Template prompts), with an optional row limit for quick previews.

```python
arch.configure(external_df=jobs_df, split=5)  # take only first 5 rows for preview
```

### sample(verbose=False)

Calls the LLM and returns decisions.
- Before `broadcast(...)`:
  - String prompt: returns a tensor of shape (1,)
  - Template prompt: if `external_df` is set, runs one prompt per row and returns a tensor of shape (num_rows,); otherwise (1,)
- After `broadcast(...)`:
  - Runs one prompt per group, then fans group outputs to agents; returns a tensor of shape (n_agents,)

```python
# Pre-broadcast single-shot
arch.sample(verbose=True)
```

### broadcast(population, match_on=None, group_on=None)

Bind a population to enable population-wide sampling. For Template prompts, `match_on` defines how group keys map from population to external data rows.

```python
import agent_torch.populations.astoria as astoria

# Match population agents to external_df rows by 'soc_code'
arch.broadcast(population=astoria, match_on="soc_code")

# Population-wide sampling
arch.sample(verbose=False)
```

### parameters()

Expose learnable parameters (only when `prompt` is a Template). Returns a list of tensors suitable for optimization.

```python
params = list(arch.parameters())
```

### Minimal optimization loop (optional)

`Archetype` works with the P3O optimizer. After you align your ground truth list to the external_df row order, a minimal loop looks like:

```python
from agent_torch.optim import P3O

opt = P3O(archetype=arch, ground_truth=ground_truth_list)
arch.sample()      # populates last group outputs/keys used by the optimizer
opt.step()
opt.zero_grad()
```

That’s it: construct → (optional) configure → broadcast → sample. Use `verbose=True` to inspect prompts and outputs during development.