# Optimizing Prompt Variables with P3O

P3O is a lightweight policy‑gradient (REINFORCE‑style) optimizer that tunes discrete Template Variables (declared with `learnable=True`) to improve the decisions produced by an `Archetype`.

### Why optimize prompts?

If you wish to reduce prompt engineering by learning how to present context fields (direct, labeled, descriptive, etc.) or improve task-specific metrics without retraining your LLM, consider using the __Variable__ learnable argument and __P3O__ import.

### How P3O works

When defining a Template object, you get to choose which __Variable__ objects the P3O optimizer gets to run on. By setting learnable to __True__, you allow P3O to receive these objects from the Template and optimize on them.

```python
age = lm.Variable(desc="agent age", learnable=True)
gender = lm.Variable(desc="agent gender", learnable = False)
```

Each learnable Variable holds trainable logits over a few presentation choices. 

```python
#as of right now, they are hard-coded in this format
# inside Variable.py, example presentation choices:
  return ""
  return value
  return f"{field_name}: {value}"
  return f"with {value}"
  return f"The {field_name} is {value}"
#in the future, it would be possible to add markers to data so p3o can optimize on which specific markers for external data it will sample and choose from rather than being vague presentation choices i.e choosing a certain skill over another
```
// TODO: Optimize the presentation choices for prompt data.


When you call `arch.sample(...)` to obtain predictions per group, you can construct a P3O object to use either:

Use a Rewards provider: this is your custom reward function. P3O calls it each step with (group_keys, group_preds, arch) and expects a list[float] of rewards (one per group, same order). Use it when you want RL/bandit-style optimization or KPIs without labeled targets.

Or use a targets provider: this includes target lookup. P3O calls it with (group_keys, arch) and expects a list[float] of targets (one per group). P3O then computes rewards via reward_fn(y, t) (default 1 − (y − t)^2). Use it for supervised-style optimization with labeled targets.

The sections below will go in-depth in how to use both of these.

#### Rewards Provider
Use a rewards provider when you want to optimize for your own objective (KPIs, safety, cost, UX) instead of classic ground‑truth targets. It lets you compute a scalar reward per group from the model’s predictions (and any external signals) without aligning or exposing targets. This is ideal for online feedback, composite metrics, or cases where “ground truth” is undefined.

This is how you may define a reward function and pass it to P3O.

Example:
```python
def rewards_provider(group_keys, group_preds, arch):
    # Simple KPI: reward higher scores closer to 0.5
    return [1.0 - (y - 0.5)**2 for y in group_preds]

opt = P3O(archetype=arch, rewards_provider=rewards_provider)
```

#### Targets Provider (optional alternative)

Define a targets provider when you already have labeled targets and want a reproducible, target‑based objective. It can be helpful when you need to switch or compare reward shapes easily via `reward_fn` without changing data plumbing. If you prefer clearer auditing of (key → target) mappings versus computing rewards inline, then opt for using a targets provider.

Provide a targets provider to P3O that returns one float target per group key. P3O combines it with an optional `reward_fn(y, t)` (default `1 - (y - t)**2`). 

Signature:
```python
def targets_provider(group_keys: list[str], arch) -> list[float]:
    ...
```

Example:
```python
def targets_provider(group_keys, arch):
    lookup = {"13-2099.01": 0.73}
    return [lookup.get(k, 0.0) for k in group_keys]

opt = P3O(archetype=arch, targets_provider=targets_provider, reward_fn=lambda y, t: 1.0 - (y - t)**2,)
```

Once more, define this only when you already have labeled targets and want a reproducible, target‑based objective.

Here is an example with pre-defined ground-truth (align targets from a keyed CSV):
```python
import pandas as pd

gt_df = pd.read_csv("ground_truth.csv")  # columns: soc_code, willingness
lookup = {str(r["soc_code"]): float(r["willingness"]) for _, r in gt_df.iterrows()}

def targets_provider(group_keys, arch):
    # Return one target per group in the same order
    return [lookup.get(str(k), 0.0) for k in group_keys]

opt = P3O(
    archetype=arch,
    targets_provider=targets_provider,
    reward_fn=lambda y, t: 1.0 - (y - t)**2,
)
```

#### Custom reward functions

If you want to use a custom rewards function:

Without targets (fully decoupled): define any reward on (group_keys, group_preds, arch)
```python
def rewards_provider(group_keys, group_preds, arch):
    # Example: maximize margin above a threshold and penalize variance
    thr = 0.6
    base = [max(0.0, y - thr) for y in group_preds]
    # optional stabilization
    mean_y = sum(group_preds) / max(1, len(group_preds))
    var_penalty = 0.1 * sum((y - mean_y)**2 for y in group_preds)
    return [b - var_penalty for b in base]

opt = P3O(archetype=arch, rewards_provider=rewards_provider)
```

With targets (supervised-style): keep targets separate and swap the fitness easily
```python
def targets_provider(group_keys, arch):
    # Pull from a service/DB/cache; one target per group
    return fetch_targets_for(group_keys)

# Quadratic by default; switch to absolute or custom metric anytime
opt = P3O(
    archetype=arch,
    targets_provider=targets_provider,
    reward_fn=lambda y, t: 1.0 - abs(y - t),
)
```
Notes:
- If both providers are passed, `rewards_provider` is used.
- Keep rewards bounded/normalized for stability (e.g., [0, 1] or [-1, 1]).

### Using the P3O Optimizer

See also: the companion notebook `p3o_demo.ipynb` in this folder for an end-to-end example of decoupled rewards and targets providers with `Archetype`.

Rewards-based variant:

```python
opt = P3O(archetype=arch, rewards_provider=rewards_provider) # create the P3O object and supply your rewards provider
for i in range(n): #optimization loop runs n times
  arch.sample(print_examples=0)   # populate last group outputs/keys
  opt.step()                      # computes rewards, updates Variable logits, and renders optimized Template object  
  opt.zero_grad() # clears gradients for next iteration
```

Targets-based variant:
```python
opt = P3O(
    archetype=arch,
    targets_provider=targets_provider,
    reward_fn=lambda y, t: 1.0 - (y - t)**2,
)
for i in range(n):
    arch.sample(print_examples=0)
    opt.step()
    opt.zero_grad()
```

Loop notes:
- Always call `arch.sample()` before `opt.step()` so group keys/preds are fresh.
- Use `print_examples=k` during debugging; keep it `0` for performance runs.
- If you need to observe choices, set `verbose=True` in P3O to print selected indices and rewards.

