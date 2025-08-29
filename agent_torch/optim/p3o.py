"""
PyTorch-style P3O Optimizer for discrete prompt optimization.

Usage:
    from agent_torch.optim import P3O
    
    opt = P3O(arch.parameters(), archetype=arch, lr=0.05)
    
    # Typical loop (no separate update needed):
    arch.sample()              # populates last_group_* on behavior
    opt.step()                 # automatically pulls from archetype and updates
    opt.zero_grad()

    # Advanced: disable auto update and control timing manually
    opt = P3O(arch.parameters(), archetype=arch, auto_update_from_archetype=False)
    arch.sample()
    opt.update_from_archetype()
    opt.step()
    opt.zero_grad()
"""

from typing import Iterable, List, Optional, Callable, Any
import torch
import torch.nn as nn


class P3O:
    """PyTorch-style optimizer for discrete prompt parameters.
    
    This optimizer uses policy gradient methods (REINFORCE) to optimize
    discrete choices in prompt templates.
    """
    
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        archetype: Any | None = None,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        reward_fn: Optional[Callable[[float, float], float]] = None,
        auto_update_from_archetype: bool = True,
        fix_choices_after_step: bool = False,
    ):
        """Initialize P3O optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            momentum: Momentum factor (for future use)
            weight_decay: Weight decay (L2 penalty)
            archetype: Optional archetype instance to read group outputs/keys from
            reward_fn: Optional reward function mapping (pred, target) -> reward
            auto_update_from_archetype: If True, step() will internally call
                update_from_archetype() before applying parameter updates.
        """
        self.param_groups = [
            {
                'params': list(params),
                'lr': lr,
                'momentum': momentum,
                'weight_decay': weight_decay,
            }
        ]
        self.state = {}
        self.archetype = archetype
        # Default to a positive reward: 1 - squared error (in [ -inf, 1 ], but >=0 when |pred-target|<=1)
        self.reward_fn = reward_fn or (lambda pred, target: 1.0 - (pred - target) ** 2)
        self.auto_update_from_archetype = auto_update_from_archetype
        self.fix_choices_after_step = fix_choices_after_step

    def compute_group_targets(self, group_keys: list[str]) -> list[float]:
        """Map group keys to ground-truth targets using template config.

        Supports either:
          - _ground_truth_list: list of target values (index-aligned or matched via _match_on)
          - legacy _ground_truth_df with _gt_value_col
        """
        template = getattr(self.archetype, "_prompt", None)
        if template is None:
            return [0.0 for _ in group_keys]
        # New path: list-based ground truth
        if getattr(template, "_ground_truth_list", None) is not None:
            gt_list = getattr(template, "_ground_truth_list")
            match_on = getattr(template, "_match_on", None)
            external_df = getattr(template, "_external_df", None)
            reducer = getattr(template, "_gt_reducer", "mean")

            # If match_on is provided and we have an external_df, match by key (e.g., soc_code)
            if match_on and external_df is not None and match_on in getattr(external_df, 'columns', []):
                # Build index lists per key for efficient lookup
                try:
                    import pandas as _pd
                    key_to_indices = {}
                    # Group external_df rows by the match_on value
                    for idx, val in enumerate(_pd.Series(external_df[match_on]).tolist()):
                        key_to_indices.setdefault(str(val), []).append(idx)
                except Exception:
                    key_to_indices = {}

                targets: list[float] = []
                for key in group_keys:
                    idxs = key_to_indices.get(str(key), [])
                    if not idxs:
                        targets.append(0.0)
                        continue
                    values = [float(gt_list[i]) if 0 <= i < len(gt_list) else 0.0 for i in idxs]
                    if not values:
                        targets.append(0.0)
                    elif reducer == "median":
                        values_sorted = sorted(values)
                        mid = len(values_sorted) // 2
                        if len(values_sorted) % 2 == 1:
                            targets.append(values_sorted[mid])
                        else:
                            targets.append(0.5 * (values_sorted[mid-1] + values_sorted[mid]))
                    elif reducer == "first":
                        targets.append(values[0])
                    else:  # mean default
                        targets.append(sum(values) / max(1, len(values)))
                return targets

            # Otherwise, use index-based alignment (keys should be indices)
            targets: list[float] = []
            for key in group_keys:
                try:
                    idx = int(key) if str(key).isdigit() else 0
                    if 0 <= idx < len(gt_list):
                        targets.append(float(gt_list[idx]))
                    else:
                        targets.append(0.0)
                except Exception:
                    targets.append(0.0)
            return targets

        # Legacy CSV ground truth path
        if getattr(template, "_ground_truth_df", None) is not None:
            gt_df = getattr(template, "_ground_truth_df", None)
            value_col = getattr(template, "_gt_value_col", "willingness")
            match_on = getattr(template, "_match_on", None)
            reducer = getattr(template, "_gt_reducer", "mean")

            targets: list[float] = []
            for key in group_keys:
                try:
                    if match_on and match_on in gt_df.columns:
                        rows = gt_df[gt_df[match_on] == key]
                        if rows.empty:
                            targets.append(0.0)
                            continue
                        series = rows[value_col]
                        if reducer == "mean":
                            targets.append(float(series.mean()))
                        elif reducer == "median":
                            targets.append(float(series.median()))
                        else:
                            targets.append(float(series.iloc[0]))
                    else:
                        # Index-aligned fallback if lengths match
                        idx = int(key) if str(key).isdigit() else 0
                        if 0 <= idx < len(gt_df):
                            targets.append(float(gt_df.iloc[idx][value_col]))
                        else:
                            targets.append(0.0)
                except Exception:
                    targets.append(0.0)
            return targets

        return [0.0 for _ in group_keys]

    def reinforce_step(self, group_preds: list[float], group_keys: list[str]) -> None:
        """Compute rewards against ground truth and nudge parameters (mock REINFORCE)."""
        targets = self.compute_group_targets(group_keys)
        rewards = [self.reward_fn(p, t) for p, t in zip(group_preds, targets)]
        avg_reward = sum(rewards) / max(len(rewards), 1)
        # Simple signal print for visibility
        print(f"P3O: avg_reward={avg_reward:.4f} over {len(rewards)} groups")
        # Mock: scale grads by reward if present
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.mul_(0.0)  # no true grads from LLM pathway

    def update_from_archetype(self) -> None:
        """Pull last predictions/keys and apply a REINFORCE-style update on Variable logits.

        Expected behavior state after arch.sample():
          - last_group_keys: list[str]
          - last_group_outputs: list[float]
          - last_slot_choices: dict[field_name -> sampled_idx] (global choices used to render)
        """
        beh = getattr(self.archetype, "_behavior", None)
        if beh is None:
            print("P3O: no behavior bound; call arch.broadcast(...); arch.sample() first")
            return
        keys = getattr(beh, "last_group_keys", None)
        preds = getattr(beh, "last_group_outputs", None)
        slot_choices = getattr(beh, "last_slot_choices", None)
        if not keys or not preds:
            print("P3O: no group info available; call arch.sample() after broadcast")
            return

        # Compute rewards per-group
        targets = self.compute_group_targets([str(k) for k in keys])
        rewards = [self.reward_fn(float(p), float(t)) for p, t in zip(preds, targets)]

        avg_reward_dbg = sum(rewards) / max(1, len(rewards))
        print(f"P3O: avg_reward={avg_reward_dbg:.4f} over {len(rewards)} groups")

        # If there are no learnable variables or no choices, nothing to update
        if not slot_choices:
            print("P3O: no slot choices present; ensure template has learnable Variables")
            return

        # Build a small REINFORCE loss over the chosen categories across groups
        loss = None
        template = getattr(self.archetype, "_prompt", None)
        if template is None:
            return
        # Sum over fields, average over groups
        fields = list(slot_choices.keys())
        num_groups = max(1, len(rewards))
        for field_name in fields:
            var = getattr(template, "_variables", {}).get(field_name)
            if var is None or not getattr(var, 'learnable', False):
                # If not in declared variables, try create_slots map
                try:
                    var = template.create_slots().get(field_name)
                except Exception:
                    var = None
            if var is None or not getattr(var, 'learnable', False):
                continue
            # Compute log_prob for the sampled index using current logits
            idx = slot_choices[field_name]
            logits = var.get_parameter(template)
            if logits is None:
                continue
            probs = torch.softmax(logits, dim=0)
            # Numerical safety
            eps = 1e-8
            logp = torch.log(probs[idx].clamp_min(eps))
            # Use average reward across groups as a simple baseline-free signal
            avg_reward = sum(rewards) / num_groups
            field_loss = -(avg_reward * logp)
            loss = field_loss if loss is None else (loss + field_loss)

        if loss is not None:
            loss.backward()
        
    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.zero_()
    
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step.
        
        Args:
            closure: Optional closure to re-evaluate model and return loss
            
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # Optionally pull latest group predictions/targets and compute rewards
        # so users don't have to call update_from_archetype() explicitly.
        if self.auto_update_from_archetype and self.archetype is not None:
            try:
                self.update_from_archetype()
            except Exception:
                # Be robust to missing behavior/state
                pass
        
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                    
                grad = param.grad.data
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)
                
                # Apply update (simple SGD for now)
                param.data.add_(grad, alpha=-lr)
        
        # Optionally fix template placeholders to greedy best choices for deterministic prompts
        if self.fix_choices_after_step and self.archetype is not None:
            try:
                template = getattr(self.archetype, "_prompt", None)
                if template is not None and hasattr(template, "create_slots"):
                    slots = template.create_slots()
                    best = {}
                    for name, var in slots.items():
                        if getattr(var, 'learnable', False):
                            best[name] = var.get_best_index(template)
                    if best:
                        template.set_optimized_slots(best)
            except Exception:
                # Be robust; don't fail optimizer if fixing choices fails
                pass

        return loss
    
    def state_dict(self) -> dict:
        """Return state of the optimizer as a dict."""
        return {
            'state': self.state,
            'param_groups': self.param_groups,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state."""
        self.state = state_dict['state']
        self.param_groups = state_dict['param_groups']
