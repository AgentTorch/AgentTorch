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
        ground_truth: List[float],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        reward_fn: Optional[Callable[[float, float], float]] = None,
        auto_update_from_archetype: bool = True,
        fix_choices_after_step: bool = False,
        reducer: str = "mean",
        verbose: bool = False,
        # PSPGO parameters
        entropy_coef: float = 0.01,
        lambda_param: float = 0.5,
        rho: float = 0.9,
        beta: float = 0.9,
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
        # Optimizer-owned ground truth (list of target values)
        if ground_truth is None or not isinstance(ground_truth, list) or len(ground_truth) == 0:
            raise ValueError("P3O: ground_truth must be a non-empty list of floats")
        self.ground_truth = ground_truth
        self.reducer = reducer
        self.verbose = bool(verbose)
        # PSPGO state
        self.entropy_coef = float(entropy_coef)
        self.lambda_param = float(lambda_param)
        self.rho = float(rho)
        self.beta = float(beta)
        self._baseline: float = 0.0
        self._y_bar: Optional[float] = None

    def compute_group_targets(self, group_keys: list[str]) -> list[float]:
        """Map group keys to optimizer-owned list ground truth.

        Behavior:
          - If group keys are numeric strings, interpret as indices into the list.
          - Otherwise, use template._match_on from broadcast and template._external_df from configure
            to map label keys to external_df rows, then aggregate list values for those rows.
        """
        gt = self.ground_truth
        template = getattr(self.archetype, "_prompt", None)
        match_on = getattr(template, "_match_on", None)
        external_df = getattr(template, "_external_df", None)

        # Fast path: all-numeric keys -> direct indexing
        all_numeric = all(str(k).isdigit() for k in group_keys)
        if all_numeric:
            out = []
            for k in group_keys:
                idx = int(str(k))
                if not (0 <= idx < len(gt)):
                    raise ValueError(f"P3O: index {idx} out of bounds for ground truth length {len(gt)}")
                out.append(float(gt[idx]))
            return out

        # Label keys: require broadcast(match_on) and configure(external_df)
        if not match_on or external_df is None or match_on not in getattr(external_df, 'columns', []):
            raise ValueError("P3O: non-index keys require broadcast(match_on=...) and configure(external_df=...) beforehand")

        # Build key->indices
        key_to_indices = {}
        for idx, val in enumerate(list(external_df[match_on])):
            key_to_indices.setdefault(str(val), []).append(idx)
        out = []
        for key in group_keys:
            idxs = key_to_indices.get(str(key))
            if not idxs:
                raise ValueError(f"P3O: missing ground truth indices for key {key}")
            values = [float(gt[i]) for i in idxs if 0 <= i < len(gt)]
            if not values:
                raise ValueError(f"P3O: empty ground truth slice for key {key}")
            if self.reducer == "median":
                s = sorted(values)
                m = len(s) // 2
                out.append(s[m] if len(s) % 2 == 1 else 0.5 * (s[m-1] + s[m]))
            elif self.reducer == "first":
                out.append(values[0])
            else:
                out.append(sum(values) / len(values))
        return out

    def reinforce_step(self, group_preds: list[float], group_keys: list[str]) -> None:
        """Compute rewards against ground truth and nudge parameters (mock REINFORCE)."""
        targets = self.compute_group_targets(group_keys)
        rewards = [self.reward_fn(p, t) for p, t in zip(group_preds, targets)]
        avg_reward = sum(rewards) / max(len(rewards), 1)
        # Simple signal print for visibility
        if self.verbose:
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
        # PSPGO-shaped rewards (scalar y per group)
        shaped_rewards: list[float] = []
        ys: list[float] = []
        for p, t in zip(preds, targets):
            y = float(p)
            tt = float(t)
            # Fitness F(y) and gradient \partial F/\partial y
            if self.reward_fn is None:
                # Default PSPGO: F(y) = -(y - t)^2; grad = -2 (y - t)
                F_y = -(y - tt) ** 2
                grad_F_y = -2.0 * (y - tt)
            else:
                # Use provided fitness; approximate grad with -2(y - t) by default
                F_y = float(self.reward_fn(y, tt))
                grad_F_y = -2.0 * (y - tt)
            if self._y_bar is None:
                self._y_bar = y
            shaped = F_y + (self.lambda_param * (grad_F_y * (y - self._y_bar)))
            shaped_rewards.append(shaped)
            ys.append(y)
        # Update running stats
        R = sum(shaped_rewards) / max(1, len(shaped_rewards))
        self._y_bar = self.rho * (self._y_bar if self._y_bar is not None else R) + (1.0 - self.rho) * (sum(ys) / max(1, len(ys)))
        self._baseline = self.beta * self._baseline + (1.0 - self.beta) * R
        advantage = R - self._baseline
        if self.verbose:
            print(f"P3O: reward={R:.4f}, adv={advantage:.4f} over {len(shaped_rewards)} groups (PSPGO default)")

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
        num_groups = max(1, len(shaped_rewards))
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
            # PSPGO: policy gradient with advantage and entropy regularization
            entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
            field_loss = -(advantage * logp) - (self.entropy_coef * entropy)
            loss = field_loss if loss is None else (loss + field_loss)

        if loss is not None:
            loss.backward()
            # Verbose diagnostics: show variable index changes and an example prompt
            if self.verbose:
                try:
                    beh = getattr(self.archetype, "_behavior", None)
                    template = getattr(self.archetype, "_prompt", None)
                    if beh is not None and template is not None:
                        slot_choices = getattr(beh, "last_slot_choices", {}) or {}
                        # Report chosen indices and resolved values
                        changes = {}
                        for field_name, idx in slot_choices.items():
                            var = getattr(template, "_variables", {}).get(field_name)
                            if var is None and hasattr(template, "create_slots"):
                                var = template.create_slots().get(field_name)
                            if var is None:
                                continue
                            logits = var.get_parameter(template)
                            if logits is None:
                                continue
                            probs = torch.softmax(logits.detach().cpu(), dim=0).tolist()
                            best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                            changes[field_name] = {
                                "selected_index": int(idx),
                                "best_index": best_idx,
                                "probs": [round(p, 4) for p in probs[:5]] + (["..."] if len(probs) > 5 else []),
                            }
                        if changes:
                            print(f"P3O: variable choices (selected_index -> best_index, sample probs): {changes}")
                            print("------------")
                        # Show a sample prompt impacted by choices (first group), using presentation variants
                        prompt_sample = None
                        prompt_list = getattr(beh, "last_prompt_list", None)
                        # Re-render the first group's prompt with active slot choices using stored group index
                        try:
                            group_indices = getattr(beh, 'last_group_indices', None)
                            if isinstance(group_indices, list) and len(group_indices) > 0:
                                agent_ids = group_indices[0]
                                # Pick the first agent id representative for this group
                                agent_id = agent_ids[0] if isinstance(agent_ids, (list, tuple)) and len(agent_ids) > 0 else 0
                                base_text = template.get_base_prompt_manager_template()
                                # Tag learnable fields in-text so _fill_section treats them as learnable
                                try:
                                    import re as _re
                                    for _fname in (slot_choices.keys() if isinstance(slot_choices, dict) else []):
                                        pattern = r"\\{" + _re.escape(str(_fname)) + r"\\}"
                                        repl = "{" + str(_fname) + ", learnable=True}"
                                        base_text = _re.sub(pattern, repl, base_text)
                                except Exception:
                                    pass
                                data0 = template.assemble_data(agent_id=agent_id, population=getattr(beh, 'population', None), mapping={}, config_kwargs={})
                                prompt_sample = template._fill_section(base_text, data0, slot_values=slot_choices)
                        except Exception:
                            # Fallback to previously stored prompt text
                            if isinstance(prompt_list, list) and prompt_list:
                                prompt_sample = prompt_list[0]
                        if prompt_sample is not None:
                            print("P3O: sample modified prompt:\n" + str(prompt_sample))
                            print("------------")
                except Exception:
                    pass
        
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