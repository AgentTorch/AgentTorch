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

### DSPy Conversions

You can convert a DSPy `Predict(Signature)` or module into an AgentTorch `Template` and optimize it directly with P3O. This preserves scaffold (instruction, demos/few‑shot) and exposes learnable slots.

- `from_dspy(module, slots, title=None)`
  - Extracts scaffold from `module.predictor` (or the module) and registers one learnable `Variable` per provided slot.
  - If the module has a signature, input/output names are surfaced in the system/output text.

- `from_predict(module, slots, title=None, categories=None, input_field=None, output_field=None)`
  - Same idea, targeted at `Predict`‑like modules. If `categories` are provided, the output section enforces strict JSON keys.

Example:
```python
from agent_torch.integrations.dspy_to_template import from_dspy

template = from_dspy(
    module=compiled_program,     # DSPy Module or Predict owner
    slots=slot_universe,         # any list of feature/attribute names you want as learnables
    title="Job Knowledge (Hybrid)",
)
```

### Using the P3O Optimizer

Prefer the built‑in `train(...)` loop with exploration modes. You can still pass additional knobs, but the presets handle temperature schedules for you.

Signature (most relevant params):
```python
opt.train(
    steps=100,                 # total optimization steps (used with exploration/mode)
    log_interval=20,           # print/save cadence
    exploration="balanced",   # quick | balanced | aggressive | long (preset name)
    batch_size=50,             # number of groups sampled per step
    sample_kwargs=None,        # forwarded to archetype.sample(...)
    mode=None,                 # same choices as exploration; when set, overrides exploration
)
```

Config modes (what they imply):
- `quick`: small number of steps, faster exploitation; short temperature decay
- `balanced`: moderate exploration vs exploitation; medium decay
- `aggressive`: high exploration for most of training; slow decay (useful early when choices are unknown)
- `long`: long exploration window with late decay; best for thorough runs

Examples:
```python
# Minimal, mode-driven
opt = P3O(archetype=arch, verbose=True)
opt.train(mode="quick", batch_size=32)

# Explicit steps with a preset schedule
opt.train(steps=5, exploration="balanced", batch_size=32)

# Larger run with logs
opt.train(mode="long", log_interval=1, batch_size=48)
```

Notes:
- `train(...)` calls into `archetype.sample(...)` and handles updates internally. No manual step/zero_grad needed.
- `batch_size` enables group subsampling during Behavior broadcast.
- Set `verbose=True` on P3O to print selections/rewards and prompt snippets during training.

