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

from typing import List, Optional, Callable, Any
import torch
import torch.nn as nn


class P3O:
    """PyTorch-style optimizer for discrete prompt parameters.
    
    This optimizer uses policy gradient methods (REINFORCE) to optimize
    discrete choices in prompt templates.
    """
    
    def __init__(
        self,
        *,
        archetype: Any,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        reward_fn: Optional[Callable[[float, float], float]] = None,
        auto_update_from_archetype: bool = True,
        fix_choices_after_step: bool = False,
        reducer: str = "mean",
        verbose: bool = False,
        # User-supplied reward/target providers
        rewards_provider: Optional[Callable[[List[str], List[float], Any], List[float]]] = None,
        targets_provider: Optional[Callable[[List[str], Any], List[float]]] = None,
        # PSPGO parameters
        entropy_coef: float = 0.01,
        lambda_param: float = 0.5,
        rho: float = 0.9,
        beta: float = 0.9,
    ):
        """Initialize P3O optimizer.
        
        Args:
            lr: Learning rate
            momentum: Momentum factor (for future use)
            weight_decay: Weight decay (L2 penalty)
            archetype: Optional archetype instance to read group outputs/keys from
            reward_fn: Optional reward function mapping (pred, target) -> reward
            auto_update_from_archetype: If True, step() will internally call
                update_from_archetype() before applying parameter updates.
        """
        # Bind archetype and derive parameters automatically from template variables
        self.archetype = archetype
        self.param_groups = [
            {
                'params': list(getattr(self.archetype, 'parameters', lambda: [])()),
                'lr': lr,
                'momentum': momentum,
                'weight_decay': weight_decay,
            }
        ]
        self.state = {}
        # reward_fn, if provided, is treated as F(y) (fitness). If None, defaults to F(y) = -(y - t)^2.
        self.reward_fn = reward_fn
        self.auto_update_from_archetype = auto_update_from_archetype
        self.fix_choices_after_step = fix_choices_after_step
        self.reducer = reducer
        self.verbose = bool(verbose)
        # Providers (decoupled)
        self.rewards_provider = rewards_provider
        self.targets_provider = targets_provider
        # PSPGO state
        self.entropy_coef = float(entropy_coef)
        self.lambda_param = float(lambda_param)
        self.rho = float(rho)
        self.beta = float(beta)
        self._baseline: float = 0.0
        self._y_bar: Optional[float] = None

    def compute_group_targets(self, group_keys: list[str]) -> list[float]:
        """Resolve targets using a user-provided targets_provider.

        If targets_provider is absent, raise to force decoupled reward usage.
        """
        if self.targets_provider is None:
            raise ValueError("P3O: no targets_provider provided; use rewards_provider instead")
        targets = self.targets_provider(group_keys, self.archetype)
        return [float(v) for v in targets]

    def reinforce_step(self, group_preds: list[float], group_keys: list[str]) -> None:
        """Compute rewards and nudge parameters (REINFORCE)."""
        # Determine rewards
        if self.rewards_provider is not None:
            rewards = self.rewards_provider(group_keys, [float(p) for p in group_preds], self.archetype)
        else:
            targets = self.compute_group_targets(group_keys)
            # Fitness F(y)
            if self.reward_fn is None:
                rewards = [1.0 - (float(y) - float(t)) ** 2 for y, t in zip(group_preds, targets)]
            else:
                rewards = [float(self.reward_fn(float(y), float(t))) for y, t in zip(group_preds, targets)]
        avg_reward = sum(rewards) / max(len(rewards), 1)
        if self.verbose:
            print(f"P3O: avg_reward={avg_reward:.4f} over {len(rewards)} groups")
        # Mock: scale grads by reward if present (no true grads from LLM pathway yet)
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.mul_(0.0)

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

        # Compute rewards per-group (decoupled)
        if self.rewards_provider is not None:
            rewards_list = self.rewards_provider([str(k) for k in keys], [float(p) for p in preds], self.archetype)
        else:
            targets = self.compute_group_targets([str(k) for k in keys])
            if self.reward_fn is None:
                rewards_list = [1.0 - (float(p) - float(t)) ** 2 for p, t in zip(preds, targets)]
            else:
                rewards_list = [float(self.reward_fn(float(p), float(t))) for p, t in zip(preds, targets)]

        # Update running stats (baseline/mean) and compute advantage
        R = sum(rewards_list) / max(1, len(rewards_list))
        if self._y_bar is None:
            self._y_bar = R
        self._y_bar = self.rho * self._y_bar + (1.0 - self.rho) * R
        self._baseline = self.beta * self._baseline + (1.0 - self.beta) * R
        advantage = R - self._baseline
        if self.verbose:
            print(f"P3O: reward={R:.4f}, adv={advantage:.4f} over {len(rewards_list)} groups")

        # Print selected indices per learnable variable for visibility
        try:
            if isinstance(slot_choices, dict) and slot_choices:
                if self.verbose:
                    print(f"P3O: selected indices = {slot_choices}")
        except Exception:
            pass

        # If there are no learnable variables or no choices, nothing to update
        if not slot_choices:
            if self.verbose:
                print("P3O: no slot choices present; ensure template has learnable Variables")
            return

        # Build a small REINFORCE loss over the chosen categories across groups
        loss = None
        template = getattr(self.archetype, "_prompt", None)
        if template is None:
            return
        # Sum over fields, average over groups
        fields = list(slot_choices.keys())
        num_groups = max(1, len(rewards_list))
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
            # Policy gradient with advantage and entropy regularization
            entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
            field_loss = -(advantage * logp) - (self.entropy_coef * entropy)
            loss = field_loss if loss is None else (loss + field_loss)

        if loss is not None:
            loss.backward()
            # Optional verbose diagnostics (probabilities, etc.) can be printed here
    
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
            except Exception as e:
                # Do not fail silently; surface the root cause and re-raise
                if self.verbose:
                    try:
                        import traceback as _tb
                        print(f"P3O: update_from_archetype failed: {e}")
                        _tb.print_exc()
                    finally:
                        raise
                else:
                    raise
        
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