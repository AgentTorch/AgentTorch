## P3O Optimization Loop (New API)

This guide focuses exclusively on using the P3O optimizer to tune learnable Template variables.

### Prerequisites

- You have an `Archetype` created with your Template and LLM backend
- You configured `external_df` and `ground_truth` via `Archetype.configure(...)`
- You have called `arch.broadcast(population=...)`

For setup details, see `archetype_tutorial.md`.

### What gets optimized

- Only Template variables declared with `learnable=True` are optimized
- `arch.parameters()` returns those learnable parameters
- Each optimization step uses: model predictions per group, matched ground-truth targets, and a reward function

### Minimal optimization loop

```python
from agent_torch.optim import P3O

# Inspect learnable parameters (optional)
params = list(arch.parameters())
print(f"Learnable parameters: {len(params)}")

# Create optimizer (default uses PSPGO-shaped reward internally)
opt = P3O(arch.parameters(), archetype=arch)

# Train for a few steps
for step in range(5):
    arch.sample()   # populates last_group_outputs/last_group_keys inside behavior
    opt.step()      # pulls group info from archetype and updates learnables
    opt.zero_grad()
```

### Customizing the reward

By default, P3O uses a PSPGO-style shaped reward: `F(y) + λ * dF/dy * (y - ȳ)` with `F(y) = 1 - (y - t)^2` and running averages for ȳ and baseline. To override:

```python
opt = P3O(
    arch.parameters(),
    archetype=arch,
    reward_fn=lambda pred, tgt: 1.0 - abs(pred - tgt),
)
```

### Tips

- Ensure your Template fields that you want optimized are defined with `learnable=True`
- Align grouping with targets using `configure(match_on=...)` (or `group_on=...`) so groups map to correct ground truth
- Use `arch.sample(print_examples=3)` occasionally to inspect prompts/outputs during training
