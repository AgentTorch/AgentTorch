import sys
import os
import torch
from typing import Any, Dict, List

from agent_torch.core.llm.agent_memory import DSPYMemoryHandler, LangchainMemoryHandler
from agent_torch.core.llm.template import Template

os.environ["DSP_CACHEBOOL"] = "False"


class Archetype:
    """Unified Archetype facade.

    Usage:
        Archetype(prompt: str|Template, llm, n_arch=1)
        .sample()                   # before broadcast -> (1,)
        .broadcast(population)
        .sample()                   # after broadcast -> (n_agents,)
        .parameters()               # learnable Variable params (Template only)
    """

    def __init__(self, prompt, llm, n_arch: int = 1):
        self.n_arch = n_arch
        self._prompt = prompt
        self._llm = llm
        self._population = None
        self._behavior = None  # internal DataFrameBehavior/Behavior wrapper
        self._p3o = None  # P3O wrapper (Template path only)

        # Build base user prompt string for LLMArchetype
        if isinstance(prompt, Template):
            # Template: build a simplified baseline prompt string for initialization
            base_user_prompt = prompt.get_base_prompt_manager_template()
        else:
            # String prompt
            base_user_prompt = str(prompt)

        # Initialize LLM archetypes (n_arch copies)
        try:
            self._llm.initialize_llm()
        except Exception:
            pass
        self._llm_archetypes: List[LLMArchetype] = [
            LLMArchetype(self._llm, base_user_prompt, n_arch=self.n_arch)
            for _ in range(self.n_arch)
        ]

    # --- Public API ---
    def broadcast(self, population) -> None:
        """Bind a population and create internal behavior based on prompt type."""
        self._population = population
        # Local imports to avoid circular dependency during module import
        from agent_torch.core.llm.behavior import Behavior  # noqa: WPS433
        if isinstance(self._prompt, Template):
            self._behavior = Behavior(
                archetype=self._llm_archetypes,
                region=population,
                template=self._prompt,
                population=population,
                optimization_interval=3,
            )
        else:
            self._behavior = Behavior(
                archetype=self._llm_archetypes,
                region=population,
            )

    def configure(self, *, external_df=None, ground_truth_src: str | None = None, match_on: str | None = None, value_col: str = "willingness", reducer: str = "mean"):
        """Configure archetype-level data and ground truth.

        - external_df: DataFrame of all jobs (923 rows) to drive prompt generation before/after broadcast
        - ground_truth_src: CSV path for targets
        - match_on: key to match external rows and/or ground truth (e.g., 'job_title' or 'soc_code')
        - value_col: target column in ground truth CSV (default: 'willingness')
        - reducer: how to combine duplicate matches ('mean' default)
        """
        if isinstance(self._prompt, Template):
            # attach external_df and matching config to Template
            if external_df is not None:
                setattr(self._prompt, "_external_df", external_df)
            if ground_truth_src is not None:
                import pandas as pd
                try:
                    gt_df = pd.read_csv(ground_truth_src)
                    setattr(self._prompt, "_ground_truth_df", gt_df)
                    setattr(self._prompt, "_gt_value_col", value_col)
                    setattr(self._prompt, "_gt_reducer", reducer)
                    # Also set path for any legacy consumers
                    setattr(self._prompt, "ground_truth_src", ground_truth_src)
                except Exception:
                    pass
            if match_on is not None:
                setattr(self._prompt, "_match_on", match_on)
        return self

    def sample(self, kwargs: Dict[str, Any] | None = None, print_examples: int = 0) -> torch.Tensor:
        """Sample decisions.
        - If broadcast not called: run a single prompt and return (1,)
        - If broadcast called: run group-based prompts over population and return (n_agents,)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if kwargs is None:
            kwargs = {"device": device, "current_memory_dir": ".agent_torch_memory"}
        else:
            kwargs = {"device": device, "current_memory_dir": ".agent_torch_memory", **kwargs}

        # If broadcast not set, do a single-shot sample
        if self._behavior is None or self._population is None:
            # One prompt only
            prompt_list = []
            if isinstance(self._prompt, Template):
                # If an external_df was configured, generate prompts for all rows
                external_df = getattr(self._prompt, "_external_df", None)
                if external_df is not None:
                    prompt_list = []
                    for row_idx in range(len(external_df)):
                        try:
                            render_text = self._prompt.render(
                                agent_id=row_idx,
                                population=self._population,
                                mapping={},
                                config_kwargs={},
                            )
                        except Exception:
                            # Graceful: show as much as possible (external fields) pre-broadcast
                            base_text = self._prompt.get_base_prompt_manager_template()
                            data = self._prompt.assemble_data(
                                agent_id=row_idx,
                                population=self._population,
                                mapping={},
                                config_kwargs={},
                            )
                            # Safe fill: replace placeholders that exist in data; leave others intact
                            import re as _re
                            def _safe_fill(m):
                                key = m.group(1).strip()
                                return str(data[key]) if key in data else m.group(0)
                            render_text = _re.sub(r"\{([^,}]+)\}", _safe_fill, base_text)
                        prompt_list.append(render_text)
                else:
                    # Single render with minimal context: agent_id 0
                    try:
                        render_text = self._prompt.render(
                            agent_id=0,
                            population=self._population,
                            mapping={},
                            config_kwargs={},
                        )
                    except Exception:
                        base_text = self._prompt.get_base_prompt_manager_template()
                        data = self._prompt.assemble_data(
                            agent_id=0,
                            population=self._population,
                            mapping={},
                            config_kwargs={},
                        )
                        import re as _re
                        def _safe_fill(m):
                            key = m.group(1).strip()
                            return str(data[key]) if key in data else m.group(0)
                        render_text = _re.sub(r"\{([^,}]+)\}", _safe_fill, base_text)
                    prompt_list = [render_text]
            else:
                prompt_list = [str(self._prompt)]

            # Query the first LLM archetype
            max_show = max(0, int(print_examples))
            print(f"\n=== Single-shot LLM Call ===")
            print(f"Prompts: {len(prompt_list)} (showing up to {max_show})")
            for i, p in enumerate(prompt_list[:max_show]):
                print(f"Prompt {i+1}:\n{p}")
            outputs = self._llm_archetypes[0](prompt_list, last_k=0)
            print(f"LLM Responses (first {max_show}): {outputs[:max_show]}{'...' if len(outputs) > max_show else ''}")
            value = 0.0
            if outputs:
                out0 = outputs[0]
                try:
                    text_value = out0["text"] if isinstance(out0, dict) and "text" in out0 else out0
                    value = float(text_value)
                    print(f"Parsed value: {value}")
                except Exception:
                    value = 0.0
                    print(f"Failed to parse, using default: {value}")
            print(f"=== End LLM Call ===\n")
            tensor_out = torch.tensor([value], device=kwargs["device"]).float()
            if len(prompt_list) > 1:
                try:
                    vals = []
                    for out in outputs:
                        tv = out["text"] if isinstance(out, dict) and "text" in out else out
                        vals.append(float(tv))
                    tensor_out = torch.tensor(vals, device=kwargs["device"]).float()
                except Exception:
                    pass
            print(f"Single-shot complete: outputs shape={tuple(tensor_out.shape)}, mean={float(tensor_out.mean().item()):.4f}")
            return tensor_out

        # Broadcast path: delegate to behavior (optionally print examples)
        try:
            pe = int(print_examples)
        except Exception:
            pe = 0
        if pe > 0:
            if kwargs is None:
                kwargs = {"device": device, "current_memory_dir": ".agent_torch_memory"}
            kwargs = {**kwargs, "print_examples": pe}
        result = self._behavior.sample(kwargs=kwargs)
        # Flatten (n,1) -> (n,) for user ergonomics
        if result.ndim == 2 and result.shape[1] == 1:
            result = result.view(-1)

        return result

    def parameters(self):
        """Return learnable parameters for optimization (Template only)."""
        if isinstance(self._prompt, Template):
            return list(self._prompt.get_slot_parameters())
        return []

    # --- Legacy surface kept for internal reuse ---
    def llm(self, llm, user_prompt):
        try:
            llm.initialize_llm()
            return [
                LLMArchetype(llm, user_prompt, n_arch=self.n_arch)
                for _ in range(self.n_arch)
            ]
        except Exception:
            return [
                LLMArchetype(llm, user_prompt, n_arch=self.n_arch)
                for _ in range(self.n_arch)
            ]

    def rule_based(self):
        raise NotImplementedError


class LLMArchetype:
    def __init__(self, llm, user_prompt, n_arch=1):
        self.n_arch = n_arch
        self.llm = llm
        # self.predictor = self.llm.initialize_llm()
        self.backend = getattr(llm, "backend", None)
        self.user_prompt = user_prompt

    def __call__(self, prompt_list, last_k):
        last_k = 2 * last_k + 8

        prompt_inputs = self.preprocess_prompts(prompt_list, last_k)
        # Support llm objects that are directly callable or expose .prompt
        if callable(getattr(self.llm, "__call__", None)):
            agent_outputs = self.llm(prompt_inputs)
        else:
            agent_outputs = self.llm.prompt(prompt_inputs)

        # Save conversation history
        if hasattr(self, "memory_handler"):
            for id, (prompt_input, agent_output) in enumerate(
                zip(prompt_inputs, agent_outputs)
            ):
                self.save_memory(prompt_input, agent_output, agent_id=id)

        return agent_outputs

    def initialize_memory(self, num_agents):
        # Lazy import to improve startup performance
        try:
            from langchain.memory import ConversationBufferMemory  # type: ignore
        except Exception:
            ConversationBufferMemory = None  # type: ignore

        self.num_agents = num_agents  # Number of agents
        if ConversationBufferMemory is not None:
            self.agent_memory = [
                ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                for _ in range(num_agents)
            ]
            # Optional memory handlers depending on available backends
            if self.backend == "dspy":
                self.memory_handler = DSPYMemoryHandler(agent_memory=self.agent_memory, llm=self.llm)
            elif self.backend in ["langchain", "claude"]:
                self.memory_handler = LangchainMemoryHandler(agent_memory=self.agent_memory)
            # Else: no memory handler; proceed without persistent history

    def preprocess_prompts(self, prompt_list, last_k):
        prompt_inputs = []
        for agent_id, prompt in enumerate(prompt_list):
            if hasattr(self, "memory_handler"):
                history = self.get_memory(last_k, agent_id=agent_id)["chat_history"]
                prompt_inputs.append({"agent_query": prompt, "chat_history": history})
            else:
                prompt_inputs.append(prompt)
        return prompt_inputs

    def reflect(self, reflection_prompt, agent_id, last_k=3):
        last_k = 2 * last_k  # get last 6 messages for each AI and Human
        return self.__call__(prompt_list=[reflection_prompt], last_k=last_k)

    def save_memory(self, query, output, agent_id):
        if hasattr(self, "memory_handler"):
            self.memory_handler.save_memory(query, output, agent_id)

    def export_memory_to_file(self, file_dir, last_k):
        if hasattr(self, "memory_handler"):
            self.memory_handler.export_memory_to_file(file_dir, last_k)

    def get_memory(self, last_k, agent_id):
        if hasattr(self, "memory_handler"):
            return self.memory_handler.get_memory(last_k=last_k, agent_id=agent_id)
        return {"chat_history": []}
