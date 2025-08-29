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
        self.reward_fn = reward_fn or (lambda pred, target: -(pred - target) ** 2)
        self.auto_update_from_archetype = auto_update_from_archetype

    def compute_group_targets(self, group_keys: list[str]) -> list[float]:
        """Map group keys to ground-truth targets using template config.

        Expects template to have _ground_truth_df, _gt_value_col, and optional _match_on.
        """
        template = getattr(self.archetype, "_prompt", None)
        if template is None:
            return [0.0 for _ in group_keys]
        if not isinstance(getattr(template, "_ground_truth_df", None), type(None)):
            gt_df = getattr(template, "_ground_truth_df", None)
        else:
            return [0.0 for _ in group_keys]

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
        """Convenience: pull last group outputs/keys from archetype and run a reinforce step.

        Call this right after arch.sample() post-broadcast.
        """
        beh = getattr(self.archetype, "_behavior", None)
        keys = getattr(beh, "last_group_keys", None)
        preds = getattr(beh, "last_group_outputs", None)
        if not keys or not preds:
            print("P3O: no group info available; call arch.broadcast(...); arch.sample() first")
            return
        self.reinforce_step(group_preds=[float(p) for p in preds], group_keys=[str(k) for k in keys])
        
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
