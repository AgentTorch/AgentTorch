"""
PyTorch-style P3O Optimizer for discrete prompt optimization.

Usage:
    from agent_torch.optim import P3O
    
    opt = P3O(archetype=arch, lr=0.05)
    
    # Typical loop (no separate update needed):
    arch.sample()              # populates last_group_* on behavior
    opt.step()                 # pulls from archetype (by default) and updates
    opt.zero_grad()

    # Advanced: disable auto update and control timing manually
    opt = P3O(archetype=arch, auto_update_from_archetype=False)
    arch.sample()
    opt.update_from_archetype()
    opt.step()
    opt.zero_grad()
"""

from typing import List, Optional, Callable, Any, Dict, Tuple
import os
import json
from datetime import datetime
import numpy as np
import torch


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
        pipeline: Optional[Callable[..., Any]] = None,
        lambda_param: float = 0.5,
        verbose: bool = False,
        auto_update_from_archetype: bool = True,
    ):
        """Initialize P3O optimizer.
        
        Args:
            archetype: Archetype instance to optimize
            lr: Learning rate for parameter updates
            pipeline: Optional reward pipeline (defaults to expt2-style)
            lambda_param: PSPGO shaped reward coefficient
            verbose: Print optimization details
            auto_update_from_archetype: If True, step() pulls latest behavior and updates
        """
        self.archetype = archetype
        params = list(getattr(self.archetype, 'parameters', lambda: [])())
        self.param_groups = [{'params': params}]
        self.state: Dict[str, Any] = {}
        self.lr = lr
        self.pipeline = pipeline
        self.lambda_param = lambda_param
        self.verbose = verbose
        self.auto_update_from_archetype = auto_update_from_archetype
        
        # PSPGO state
        self._baseline = 0.0
        self._y_bar_vec: Optional[np.ndarray] = None
        
        # Tracking state for train method
        self._step_count = 0
        self._best_reward = float('-inf')
        self._last_reward: Optional[float] = None
        self._last_advantage: Optional[float] = None
        self._last_sampled_choices: Dict[str, int] = {}
        self._metrics: List[Dict[str, Any]] = []
        self._metrics_file_path: Optional[str] = None

        # Optional writer and best snapshot holders
        self._writer = None
        self._best_logits: Dict[str, torch.Tensor] = {}

    def _default_expt2_pipeline(self, group_key, group_pred_or_structured, arch_obj):
        """Default expt2-style pipeline."""
        if isinstance(group_pred_or_structured, dict):
            pred_value = sum(group_pred_or_structured.values()) / max(1, len(group_pred_or_structured))
        else:
            pred_value = float(group_pred_or_structured)
            
        y = np.array([pred_value], dtype=float)
        t = 50.0
        
        if isinstance(group_key, str) and group_key.startswith("job_"):
            row_idx = int(group_key.split("_")[1])
            external_df = getattr(arch_obj._prompt, '_external_df', None)
            if external_df is not None and 0 <= row_idx < len(external_df):
                knowledge_cols = [col for col in external_df.columns if col.startswith('NUMERIC_knowledge_')]
                if knowledge_cols:
                    target_val = external_df.iloc[row_idx][knowledge_cols[0]]
                    t = float(target_val)
            
        gt = np.array([t], dtype=float)
        mse = float(np.mean((y - gt) ** 2))
        
        combined_error = mse
        original_f_y = 10000 - combined_error
        f_y = -10 + 20 * (original_f_y / 10000)
        scaling_factor = 20 / 10000
        grad_f_y = -scaling_factor * (2 * (y - gt) / y.shape[0])
        metrics = {'mse': mse}
        return y, float(f_y), grad_f_y.reshape(-1), metrics

    def update_from_archetype(self) -> None:
        """Pull last predictions/keys and apply a REINFORCE-style update on Variable logits.

        Expected behavior state after arch.sample():
          - last_group_keys: list[str]
          - last_group_outputs: list[float]
          - last_group_structured: Optional[list[dict]]
          - last_slot_choices: dict[field_name -> sampled_idx] (global choices used to render)
        """
        beh = getattr(self.archetype, "_behavior", None) or getattr(self.archetype, "_mock_behavior", None)
        if beh is None:
            if self.verbose:
                print("P3O: no behavior bound; call arch.broadcast(...) or arch.sample(batch_size=N) first")
            return

        keys = getattr(beh, "last_group_keys", None)
        preds = getattr(beh, "last_group_outputs", None)
        structured_preds = getattr(beh, "last_group_structured", None)
        slot_choices = getattr(beh, "last_slot_choices", None)

        if not keys or preds is None or len(keys) == 0:
            if self.verbose:
                print("P3O: no group info available; call arch.sample() after broadcast")
            return

        # Resolve effective pipeline (experiment-provided) if none was explicitly set
        effective_pipeline = self.pipeline
        if effective_pipeline is None and self.archetype is not None:
            try:
                get_pipe = getattr(self.archetype, 'get_p3o_pipeline', None)
                if callable(get_pipe):
                    effective_pipeline = get_pipe()
            except Exception:
                effective_pipeline = None
            if effective_pipeline is None:
                pipe_attr = getattr(self.archetype, 'p3o_pipeline', None)
                if callable(pipe_attr):
                    effective_pipeline = pipe_attr
        if effective_pipeline is None:
            effective_pipeline = self._default_expt2_pipeline
            
        rewards_list: List[float] = []
        y_vecs: List[np.ndarray] = []
        structured_preds = structured_preds or [{}] * len(keys)

        for k, p, structured in zip(keys, preds, structured_preds):
            pipeline_input = structured if structured else p
            out = effective_pipeline(k, pipeline_input, self.archetype)
            
            # Accept expt2-style tuple (y, f_y, grad[, metrics]) or a float/dict
            if isinstance(out, tuple) and len(out) >= 3:
                y, f_y, grad_f_y = out[0], out[1], out[2]
                y_vec = np.array(y, dtype=float).reshape(-1)
                grad_vec = np.array(grad_f_y, dtype=float).reshape(-1)
                # Shaped reward uses current y_bar_vec (initialize lazily)
                if self._y_bar_vec is None:
                    self._y_bar_vec = y_vec.copy()
                shaped = float(f_y) + float(self.lambda_param) * float(np.dot(grad_vec, (y_vec - self._y_bar_vec)))
                rewards_list.append(shaped)
                y_vecs.append(y_vec)
            elif isinstance(out, (int, float)):
                rewards_list.append(float(out))
            elif isinstance(out, dict) and 'reward' in out:
                rewards_list.append(float(out['reward']))
            else:
                raise ValueError("P3O: pipeline must return (y,f_y,grad[,...]) or a float reward or {'reward': ...}.")

        # Update moving average of y
        if y_vecs:
            y_mean = np.mean(np.stack(y_vecs, axis=0), axis=0)
            if self._y_bar_vec is None:
                self._y_bar_vec = y_mean.copy()
            else:
                self._y_bar_vec = 0.9 * self._y_bar_vec + 0.1 * y_mean

        # Per-job advantages (raw rewards)
        job_advantages = {group_key: reward for group_key, reward in zip(keys, rewards_list)}

        # Reporting
        R = float(sum(rewards_list) / max(1, len(rewards_list)))
        average_advantage = float(sum(job_advantages.values()) / max(1, len(job_advantages)))
        
        self._last_reward = R
        self._last_advantage = average_advantage
        if R > self._best_reward:
            self._best_reward = R
            
        if self.verbose:
            print(f"\033[93mP3O: reward={R:.4f}, adv={average_advantage:.4f} over {len(rewards_list)} groups\033[0m")

        # Selected indices per learnable variable for visibility (optional)
        if self.verbose and isinstance(slot_choices, dict) and slot_choices:
            print(f"P3O: selected indices = {slot_choices}")
        
        # Store the actual sampled choices for JSON output
        if isinstance(slot_choices, dict):
            self._last_sampled_choices = slot_choices.copy()

        # If there are no learnable variables or no choices, nothing to update
        if not slot_choices:
            if self.verbose:
                print("P3O: no slot choices present; ensure template has learnable Variables")
            return

        # Simple REINFORCE: each slot uses its actual sampled choice with shared reward
        loss: Optional[torch.Tensor] = None
        template = getattr(self.archetype, "_prompt", None)
        if template is None:
            return

        # Average advantage across all jobs for this batch
        avg_advantage = sum(job_advantages.values()) / max(1, len(job_advantages))

        # Standard REINFORCE for each learnable variable
        for field_name, sampled_idx in slot_choices.items():
            var = getattr(template, "_variables", {}).get(field_name)
            if var is None or not getattr(var, 'learnable', False):
                try:
                    var = template.create_slots().get(field_name)
                except Exception:
                    var = None
            if var is None or not getattr(var, 'learnable', False):
                continue
                
            logits = var.get_parameter(template)
            if logits is None or not isinstance(logits, torch.Tensor):
                continue

            # Standard policy gradient: -advantage * log_prob(sampled_action)
            probs = torch.softmax(logits, dim=0)
            log_prob = torch.log(torch.clamp(probs[sampled_idx], min=1e-8))
            var_loss = -avg_advantage * log_prob
            
            if loss is None:
                loss = var_loss
            else:
                loss = loss + var_loss

        if loss is not None:
            loss.backward()
            # Gradient clipping and parameter bounds for numerical stability
            for group in self.param_groups:
                for param in group['params']:
                    if isinstance(param, torch.Tensor) and param.grad is not None:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
                        # Check for NaN in gradients
                        if torch.isnan(param.grad).any():
                            print(f"WARNING: NaN gradient detected, zeroing")
                            param.grad.zero_()
            
            # Manually apply gradients (P3O doesn't inherit from PyTorch optimizer)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.data.add_(p.grad, alpha=-self.lr)
                        # Clamp parameters to prevent extreme values
                        p.data = torch.clamp(p.data, min=-10.0, max=10.0)
            # Snapshot best logits on improvement
            try:
                if self._last_reward is not None and self._last_reward >= self._best_reward:
                    self._best_reward = self._last_reward
                    self._snapshot_best_logits()
            except Exception:
                pass

            # Optional prompt previews
            if self.verbose:
                behavior = getattr(self.archetype, "_behavior", None)
                prompt_list = getattr(behavior, "last_prompt_list", None)
                if isinstance(prompt_list, list) and prompt_list:
                    for i, p in enumerate(prompt_list[:3]):  # show up to 3 prompts
                        snippet = p.strip()
                        if len(snippet) > 400:
                            snippet = snippet[:400] + "..."
                        print(f"P3O: prompt preview [{i+1}/{len(prompt_list)}] =>\n{snippet}")
        
    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        for group in self.param_groups:
            for param in group['params']:
                if isinstance(param, torch.Tensor) and param.grad is not None:
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
        
        if self.auto_update_from_archetype:
                self.update_from_archetype()
        
        for group in self.param_groups:
            for param in group['params']:
                if isinstance(param, torch.Tensor) and param.grad is not None:
                    param.data.add_(param.grad.data, alpha=-self.lr)

        return loss
    
    def state_dict(self) -> dict:
        """Return state of the optimizer as a dict."""
        return {
            'state': self.state,
            'param_groups': self.param_groups,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state."""
        self.state = state_dict.get('state', {})
        self.param_groups = state_dict.get('param_groups', self.param_groups)

    # -----------------------------
    # Training utilities and helpers
    # -----------------------------
    def _log_metrics(self, step: int, temperature: Optional[float] = None) -> None:
        metrics = {
            'step': int(step),
            'reward': float(self._last_reward) if self._last_reward is not None else None,
            'advantage': float(self._last_advantage) if self._last_advantage is not None else None,
            'baseline': float(self._baseline),
            'best_reward': float(self._best_reward),
        }
        if temperature is not None:
            metrics['temperature'] = float(temperature)
        self._metrics.append(metrics)
        if self._metrics_file_path:
            try:
                with open(self._metrics_file_path, 'w', encoding='utf-8') as f:
                    json.dump(self._metrics, f, indent=2)
            except Exception:
                pass

    def _snapshot_best_logits(self) -> None:
        template = getattr(self.archetype, "_prompt", None)
        if template is None or not hasattr(template, "create_slots"):
            return
        try:
            slots = template.create_slots()
        except Exception:
            return
        best: Dict[str, torch.Tensor] = {}
        for name, var in slots.items():
            if getattr(var, 'learnable', False):
                logits = var.get_parameter(template)
                if isinstance(logits, torch.Tensor):
                    best[name] = logits.detach().clone()
        self._best_logits = best

    def switch_to_best_run(self) -> None:
        template = getattr(self.archetype, "_prompt", None)
        if template is None or not hasattr(template, "create_slots"):
            return
        if not self._best_logits:
            return
        try:
            slots = template.create_slots()
        except Exception:
            return
        for name, tensor in self._best_logits.items():
            var = slots.get(name)
            if var is None or not getattr(var, 'learnable', False):
                continue
            logits = var.get_parameter(template)
            if isinstance(logits, torch.Tensor) and logits.shape == tensor.shape:
                logits.data.copy_(tensor.data)

    def get_slot_probabilities(self) -> Dict[str, List[float]]:
        template = getattr(self.archetype, "_prompt", None)
        if template is None or not hasattr(template, "create_slots"):
            return {}
        try:
            slots = template.create_slots()
        except Exception:
            return {}
        out: Dict[str, List[float]] = {}
        for name, var in slots.items():
            if getattr(var, 'learnable', False):
                logits = var.get_parameter(template)
                if isinstance(logits, torch.Tensor):
                    probs = torch.softmax(logits.detach(), dim=0)
                    out[name] = [float(x) for x in probs.tolist()]
        return out

    def get_final_results(self) -> dict:
        return {
            'best_reward': float(self._best_reward),
            'slot_probabilities': self.get_slot_probabilities(),
            'total_steps': int(self._step_count),
        }

    def get_p3o_selections(self) -> Dict[str, int]:
        """Return deterministic selections per learnable variable via argmax over logits."""
        template = getattr(self.archetype, "_prompt", None)
        if template is None or not hasattr(template, "_variables"):
            return {}
        selections: Dict[str, int] = {}
        for name, var in template._variables.items():
            if getattr(var, "learnable", False):
                param = var.get_parameter(template)
                probs = torch.softmax(param, dim=0)
                selections[name] = int(torch.argmax(probs).item())
        return selections

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass

    def train(
        self,
        *,
        steps: int = 100,
        log_interval: int = 20,
        exploration: str = "balanced",
        batch_size: int = 50,
        sample_kwargs: Optional[dict] = None,
        mode: Optional[str] = None,
    ) -> List[dict]:
        """Run a built-in training loop: sample → step → zero_grad."""
        sample_kwargs = sample_kwargs or {}
        # Optional preset modes
        if mode is not None:
            mode_l = str(mode).lower()
            if mode_l in ("quick", "balanced", "aggressive", "long"):
                exploration = mode_l
            else:
                raise ValueError(f"Unknown P3O mode: {mode}. Choose from: quick|balanced|aggressive|long")
        exp_cfg = get_exploration_config(exploration, steps)
        history: List[dict] = []
        for s in range(steps):
            if s < exp_cfg['temperature_decay_steps']:
                temperature = exp_cfg['initial_temperature'] * (
                    exp_cfg['final_temperature'] / exp_cfg['initial_temperature']
                ) ** (s / max(1, exp_cfg['temperature_decay_steps']))
            else:
                temperature = exp_cfg['final_temperature']
            # sample
            try:
                self.archetype.sample(temperature=temperature, batch_size=batch_size, **sample_kwargs)
            except TypeError:
                self.archetype.sample(batch_size=batch_size, **sample_kwargs)
            # optimize
            self.step()
            self.zero_grad()
            self._step_count += 1
            self._log_metrics(self._step_count, temperature)

            if self.verbose and ((s + 1) % max(1, log_interval) == 0):
                last = self._last_reward if self._last_reward is not None else float('nan')
                best = self._best_reward if self._best_reward != float('-inf') else float('nan')
                print(f"Step {s+1:3d}: reward={last:.3f} best={best:.3f} temp={temperature:.3f}")
                step_files = self.save_step_results(s + 1)
                print(f"Step {s+1} results saved: {step_files['results']}")

            history.append({
                'step': int(self._step_count),
                'reward': float(self._last_reward) if self._last_reward is not None else None,
                'best_reward': float(self._best_reward),
                'temperature': float(temperature),
                'logits': self.get_current_logits_json(),
                'probabilities': self.get_current_probabilities_json(),
            })
        return history

    def get_current_logits_json(self) -> str:
        """Return current logit values for all learnable variables as JSON."""
        logits_dict: Dict[str, Any] = {}
        if hasattr(self.archetype, '_prompt') and hasattr(self.archetype._prompt, '_variables'):
            for var_name, variable in self.archetype._prompt._variables.items():
                if getattr(variable, 'learnable', False):
                    param = variable.get_parameter(self.archetype._prompt)
                    if isinstance(param, torch.Tensor):
                        logits_dict[var_name] = param.detach().cpu().tolist()
        return json.dumps(logits_dict, indent=2)

    def get_current_probabilities_json(self) -> str:
        """Return current probability distributions for all learnable variables as JSON."""
        probs_dict: Dict[str, Any] = {}
        if hasattr(self.archetype, '_prompt') and hasattr(self.archetype._prompt, '_variables'):
            for var_name, variable in self.archetype._prompt._variables.items():
                if getattr(variable, 'learnable', False):
                    param = variable.get_parameter(self.archetype._prompt)
                    if isinstance(param, torch.Tensor):
                        probs = torch.softmax(param, dim=0)
                        probs_dict[var_name] = probs.detach().cpu().tolist()
        return json.dumps(probs_dict, indent=2)

    def _ensure_results_dir(self) -> str:
        """Create optim/results directory if it doesn't exist and return the path."""
        results_dir = os.path.join("optim", "results")
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

    def get_current_step_data(self) -> dict:
        """Get comprehensive step data including logits, probabilities, choices, and rendered template."""
        result: Dict[str, Any] = {"variables": {}}
        if hasattr(self.archetype, '_prompt') and hasattr(self.archetype._prompt, '_variables'):
            for var_name, variable in self.archetype._prompt._variables.items():
                if getattr(variable, 'learnable', False):
                    param = variable.get_parameter(self.archetype._prompt)
                    if isinstance(param, torch.Tensor):
                        logits = param.detach().cpu().tolist()
                        probs = torch.softmax(param, dim=0)
                        probs_list = probs.detach().cpu().tolist()
                        argmax_choice = int(torch.argmax(probs).item())
                        argmax_prob = float(probs[argmax_choice].item())
                        
                        # Use actual sampled choice if available, otherwise fall back to argmax
                        sampled_choice = self._last_sampled_choices.get(var_name, argmax_choice)
                        sampled_prob = float(probs[sampled_choice].item())
                        
                        result["variables"][var_name] = {
                            "logits": logits,
                            "probabilities": probs_list,
                            "argmax": argmax_choice,
                            "highest_probability": argmax_prob,
                            "sampled_choice": sampled_choice,
                            "sampled_probability": sampled_prob,
                            "status": "include" if sampled_choice == 1 else "exclude"
                        }
        
        # Add rendered template with current optimized choices
        result["output_template"] = self._get_rendered_template_with_choices()
        
        return result
    
    def _get_rendered_template_with_choices(self) -> str:
        """Render template with current optimized variable choices using real job data."""
        try:
            template = self.archetype._prompt
            if not hasattr(template, '_variables'):
                return "Template has no variables"
            
            # Get current argmax choices for all learnable variables
            current_choices = {}
            for var_name, variable in template._variables.items():
                if getattr(variable, 'learnable', False):
                    param = variable.get_parameter(template)
                    if isinstance(param, torch.Tensor):
                        probs = torch.softmax(param, dim=0)
                        current_choices[var_name] = int(torch.argmax(probs).item())
            
            # Use real job data from external_df if available (configured via archetype.configure())
            external_df = getattr(template, '_external_df', None)
            if external_df is not None and len(external_df) > 0:
                # Use first job as example
                first_job = external_df.iloc[0]
                sample_data = {"soc_code": first_job.get("soc_code", "UNKNOWN")}
                
                # Add real skill values for variables that exist in the data
                for var_name in template._variables.keys():
                    if var_name != "soc_code":
                        choice = current_choices.get(var_name, 1)
                        if choice == 0:
                            sample_data[var_name] = ""  # Skip/empty when choice=0
                        else:
                            # Use the actual skill content from external data (if available)
                            if var_name in first_job.index:
                                sample_data[var_name] = first_job[var_name]  # Keep original type (binary 0/1)
                            else:
                                sample_data[var_name] = 0  # Default to not relevant
            else:
                # Fallback to sample data
                sample_data = {"soc_code": "11-1011.00"}
                for var_name in template._variables.keys():
                    if var_name != "soc_code":
                        choice = current_choices.get(var_name, 1)
                        sample_data[var_name] = "" if choice == 0 else f"Include {var_name.replace('_', ' ')}"
            
            # Render the prompt with current choices
            try:
                rendered = template._fill_section(
                    template_section=template.__prompt__(),
                    data=sample_data,
                    slot_values=current_choices
                )
                # Clean empty lines before returning (important for LLM prompt quality)
                cleaned = "\n".join(line for line in rendered.split("\n") if line.strip())
                return cleaned
            except Exception as render_error:
                # Return debug info if rendering fails
                return f"Template rendering failed:\nError: {str(render_error)}\nData keys: {list(sample_data.keys())}\nTemplate vars: {list(template._variables.keys())}\nChoices: {current_choices}"
            
        except Exception as e:
            return f"Error in template processing: {str(e)}"

    def save_step_results(self, step) -> dict:
        """Save comprehensive step results to a single JSON file."""
        results_dir = self._ensure_results_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"step_{step if isinstance(step, str) else f'{int(step):03d}'}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        step_data = self.get_current_step_data()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(step_data, f, indent=2)
        return {"results": filepath, "logits": filepath, "probabilities": filepath}


def get_exploration_config(exploration_type: str, total_steps: int = 100) -> Dict[str, Any]:
    """Exploration schedule presets for temperature annealing."""
    configs: Dict[str, Dict[str, Any]] = {
        'aggressive': {
            'initial_temperature': 15.0,
            'final_temperature': 0.05,
            'temperature_decay_steps': int(0.9 * total_steps),
            'description': 'Very high exploration for most of training',
            'steps': 40,
        },
        'balanced': {
            'initial_temperature': 8.0,
            'final_temperature': 0.3,
            'temperature_decay_steps': int(0.6 * total_steps),
            'description': 'Moderate exploration, balanced with exploitation',
            'steps': 30,
        },
        'quick': {
            'initial_temperature': 5.0,
            'final_temperature': 0.5,
            'temperature_decay_steps': int(0.2 * total_steps),
            'description': 'Quick exploration, fast exploitation',
            'steps': 5,
        },
        'long': {
            'initial_temperature': 10.0,
            'final_temperature': 0.1,
            'temperature_decay_steps': int(0.8 * total_steps),
            'description': 'Long exploration phase, late exploitation',
            'steps': 50,
        },
    }
    if exploration_type not in configs:
        raise ValueError(f"Unknown exploration type: {exploration_type}. Choose from: {list(configs.keys())}")
    return configs[exploration_type]
