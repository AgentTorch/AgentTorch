## Optimizing on Prompts with P3O

P3O is a policy‑gradient optimizer (PSPGO‑style by default) that tunes discrete Template Variables (the fields you declare with `learnable=True`) to improve decisions produced by an `Archetype`.

### What P3O does (PSPGO default)

- Treats each learnable Variable as a categorical policy over its choices/logits.
- Runs the Archetype to get per‑group predictions and matches them to ground truth.
- Computes a shaped fitness‑based reward with running statistics and applies a REINFORCE‑style update with entropy regularization:
  - Fitness F(y) defaults to F(y) = −(y − t)^2, where y is the model output and t is the target.
  - Shaped reward per group: R_g = F(y) + λ (∂F/∂y) (y − ȳ), with ȳ a running mean of outputs.
  - Running stats: ȳ ← ρ ȳ + (1 − ρ) mean(y), baseline ← β baseline + (1 − β) R.
  - Advantage: A = R − baseline.
  - Loss per Variable: L = −A log π(idx) − entropy_coef · H(π).
- No changes to your LLM are required; no `loss.backward()` is needed from user code (P3O calls backward on Variable logits internally).

### Template reminders (no inline flags needed)

- Declare learnable fields as `Variable(learnable=True)`; you do not need to put `, learnable=True` in the prompt text anymore.
- Use regular placeholders like `{age}`, `{gender}`, `{soc_code}` in your Template prompt string. Learnability is inferred from the declared Variables during rendering.

### What is a Variable?

`Variable` is a descriptor you declare on a Template to define a field used in prompt rendering.

```python
import agent_torch.core.llm.template as lm

class MyTemplate(lm.Template):
    age = lm.Variable(desc="agent age", learnable=True)
    gender = lm.Variable(desc="agent gender", learnable=False, default="unknown")

    def __prompt__(self):
        self.prompt_string = "You are in your {age}'s and are a {gender}."

    def __output__(self):
        return "Return a number in [0, 1]"
```

- `learnable=True` exposes parameters via `arch.parameters()`; P3O treats them as categorical policies (logits → choice index).
- `default` provides a fallback when the value isn’t present in population/external data.
- Placeholders use field names: `{age}`, `{gender}`.

### When to use

- You have a Template with one or more `learnable=True` Variables
- You have ground‑truth targets aligned to your external data rows
- You want to automatically improve the selected Variable choices over time

### Minimal workflow

```python
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.mock_llm import MockLLM
from agent_torch.optim import P3O

# 1) Build Archetype with a Template that declares learnable Variables
arch = Archetype(prompt=my_template, llm=MockLLM(), n_arch=3)

# 2) Attach external_df (rows drive prompts) and align ground truth list
arch.configure(external_df=external_df)
ground_truth_list = aligned_targets  # len == len(external_df), same row order

# 3) (Optional) Broadcast to a population for group-based sampling
arch.broadcast(population=astoria, match_on="soc_code")

# 4) Create P3O and optimize
opt = P3O(archetype=arch, ground_truth=ground_truth_list)  # PSPGO defaults enabled
arch.sample()   # populates last group outputs/keys
opt.step()      # computes rewards, updates Variable logits
opt.zero_grad()
```

### Presentation variants and verbose diagnostics

With `verbose=True`, P3O prints per‑step diagnostics:
- average shaped reward and advantage
- selected indices for each learnable Variable and sample probabilities
- a sample modified prompt that shows the chosen presentation variants

Example (abbreviated):

```
P3O: reward=-0.0843, adv=-0.0680 over 923 groups (PSPGO default)
P3O: selected indices = {'age': 3, 'gender': 1, 'soc_code': 3}
P3O: variable choices (selected_index -> best_index, sample probs): {'age': {...}, 'gender': {...}, 'soc_code': {...}}
------------
P3O: sample modified prompt:
You are evaluating willingness based on job profile and context. You are in your age: 20–29's and are a male. As a soc_code: 13-2099.01 ...
------------
```

Notes:
- The printed prompt uses the variant‑aware renderer, applying the sampled indices to format fields (e.g., direct, labeled, contextual, descriptive).
- If a field still shows raw numbers (e.g., `gender: 0`), add a mapping to convert codes to labels.

### Mapping numeric codes to readable labels

To convert population codes (e.g., age buckets, gender ids) into user‑friendly text during presentation rendering, place a `mapping.json` adjacent to your population or data source. It will be picked up automatically.

Example `mapping.json`:

```json
{
  "age": ["20–29", "30–39", "40–49", "50–59", "60+"],
  "gender": ["male", "female"],
  "soc_code": []
}
```

With this file present, P3O’s sample modified prompt will show labels instead of numeric codes when formatting learnable fields.

### Aligning ground truth

Targets must correspond to external_df rows. If your ground truth is in a CSV keyed by a column like `soc_code`, align first:

```python
from helper_soc_data import align_ground_truth_to_external_df

external_df, ground_truth_list = align_ground_truth_to_external_df(
    gt_csv_path, external_df, key="soc_code", value_col="willingness", mode="fill"
)
```

### Interpreting updates

P3O updates the logits behind each `learnable=True` Variable. At runtime, those logits determine which choice (index) is rendered into prompts. By default, the same sampled indices apply to all groups within a `sample()` call (global, stable policy per iteration).

### Diagnostics (verbose)

Enable logging to see rewards/advantage, selected vs best indices, sample probabilities, and a sample modified prompt:

```python
opt = P3O(archetype=arch, ground_truth=ground_truth_list, verbose=True)
```

You’ll see lines like:
- PSPGO reward and advantage over groups
- Variable choices (selected vs best index, sample probabilities)
- A sample prompt that includes the chosen Variable values

### Advanced options

```python
P3O(
    archetype=arch,
    ground_truth=ground_truth_list,
    lr=0.01,
    weight_decay=0.0,
    # If provided, reward_fn is treated as fitness F(y); default is F(y)=−(y−t)^2
    reward_fn=None,
    auto_update_from_archetype=True,   # step() pulls latest predictions/keys
    fix_choices_after_step=False,      # optionally lock to greedy choices
    reducer="mean",                   # mean|median|first for key→targets aggregation
    # PSPGO controls
    entropy_coef=0.01,
    lambda_param=0.5,
    rho=0.9,
    beta=0.9,
    verbose=False,
)
```

### Tips

- Keep `arch.sample()` as the single source of LLM calls; P3O relies on its stored `last_group_outputs/keys` (and logs a sample prompt).
- Ensure `ground_truth_list` is the same length and ordering as `external_df`.
- Use `verbose=True` during development; turn it off for batch runs.
- For custom fitness F(y), pass `reward_fn`; gradient is approximated as −2(y − t) by default.
- If you want strictly non‑negative rewards, set `reward_fn=lambda y, t: max(0.0, 1.0 - (y - t)**2)`; PSPGO shaping still applies.


