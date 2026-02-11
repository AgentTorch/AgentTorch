"""
Wiki hybrid experiment (DSPy → Template → P3O), using your local JSONL Wikipedia dump.

Pipeline:
- Stream a small sample of JSONL docs from WIKI_DIR
- Build a simple DSPy classifier ("Determine the category of this article") with few-shot demos
- Compile with MIPROv2 (light) on a tiny split
- Convert to AgentTorch Template via from_predict with a marker-based attributes block
- Run a short P3O optimization (mode="quick") over the sample

Env knobs:
- WIKI_DIR: path to the folder containing enwiki_namespace_0_*.jsonl
- MAX_WIKI_DOCS: cap number of documents to sample (default: 200)
- DEMO_COUNT: number of few-shot demos used in scaffold (default: 6)
- RR_MARKER: marker string for attributes injection (default: "<<ATTRIBUTES>>")
"""

from __future__ import annotations

import os
import sys
import json
import glob
from typing import List, Dict, Any, Tuple, Optional
import argparse
from dataclasses import dataclass, field

import pandas as pd

# Ensure repo root is importable when running directly
_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

# --- USER CONFIG (fill in if you prefer not to use env vars) ---
# If GOOGLE_API_KEY/GEMINI_API_KEY are not set, we will try this placeholder.
# Replace the value below with your real Gemini API key (keep quotes):
# Leave empty to avoid accidental usage when mocking
USER_DSPY_GEMINI_KEY = ""
# --------------------------------------------------------------

from agent_torch.integrations.dspy_to_template import (
    from_predict,
)
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.mock_llm import MockLLM
from agent_torch.optim.p3o import P3O, get_exploration_config
from datetime import datetime


def _get_env_any(names: List[str]) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v and isinstance(v, str) and v.strip():
            return v
    lower_map = {k.lower(): v for k, v in os.environ.items()}
    for n in names:
        if n.lower() in lower_map and str(lower_map[n.lower()]).strip():
            return lower_map[n.lower()]
    return None


def _extract_id_title_text_url(obj: Dict[str, Any]) -> Dict[str, str]:
    """Precise extractor for common wiki JSONL variants observed in schema report.
    Maps:
      - id: identifier | id
      - title: name | title | page_title | meta.title | document.title
      - url: url | main_entity.url | is_part_of.url
      - text: description | abstract | sections[].has_parts[].text (joined) | fallback string-leaf join
    """
    # id
    art_id = (
        str(obj.get("identifier") or obj.get("id") or "").strip()
    )
    # title
    title = (
        obj.get("name")
        or obj.get("title")
        or obj.get("page_title")
        or obj.get("meta", {}).get("title")
        or obj.get("document", {}).get("title")
        or ""
    )
    # url
    url = (
        obj.get("url")
        or (obj.get("main_entity", {}) or {}).get("url")
        or (obj.get("is_part_of", {}) or {}).get("url")
        or ""
    )
    # text: prefer concise fields first
    text = (
        obj.get("description")
        or obj.get("abstract")
    )
    # sections join if needed
    if not (isinstance(text, str) and text.strip()):
        try:
            sections = obj.get("sections") or []
            parts: List[str] = []
            if isinstance(sections, list):
                for s in sections:
                    if not isinstance(s, dict):
                        continue
                    hp = s.get("has_parts") or []
                    if not isinstance(hp, list):
                        continue
                    for p in hp:
                        if isinstance(p, dict):
                            t = p.get("text")
                            if isinstance(t, str) and t.strip():
                                parts.append(t)
            if parts:
                text = "\n\n".join(parts)
        except Exception:
            text = None
    # fallback: concatenate all string-like leaves
    if not (isinstance(text, str) and text.strip()):
        def _gather_strings(x: Any, acc: List[str]) -> None:
            if isinstance(x, str):
                if x.strip():
                    acc.append(x)
            elif isinstance(x, list):
                for it in x[:20]:
                    _gather_strings(it, acc)
            elif isinstance(x, dict):
                for i, (_k, _v) in enumerate(list(x.items())[:40]):
                    _gather_strings(_v, acc)
        acc2: List[str] = []
        _gather_strings(obj, acc2)
        text = "\n\n".join(acc2)
    return {
        "id": str(art_id or "").strip(),
        "title": str(title or "").strip(),
        "text": str(text or "").strip(),
        "url": str(url or "").strip(),
    }


def _stream_jsonl_sample(dir_path: str, limit: int = 200) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(dir_path, "enwiki_namespace_0_*.jsonl")))
    rows: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if len(rows) >= limit:
                        return rows
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    rec = _extract_id_title_text_url(obj)
                    title = rec.get("title", "")
                    text = rec.get("text", "")
                    if not (isinstance(text, str) and text.strip()):
                        continue
                    rows.append({
                        "id": rec.get("id", ""),
                        "url": rec.get("url", ""),
                        "title": str(title),
                        "text": str(text),
                    })
        except Exception:
            continue
    return rows


def _truncate(s: str, n: int = 600) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


WIKI_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("Science", ["physics", "chemistry", "biology", "research", "scientific", "astronomy", "genetics"]),
    ("Sports", ["football", "soccer", "basketball", "olympic", "league", "match", "player", "coach"]),
    ("Arts", ["art", "painting", "literature", "novel", "film", "music", "album", "poetry", "theatre"]),
    ("History", ["war", "empire", "dynasty", "century", "king", "queen", "ancient", "medieval", "revolution"]),
    ("Geography", ["city", "country", "river", "mountain", "population", "province", "region", "capital"]),
    ("Technology", ["computer", "software", "internet", "technology", "engineering", "algorithm", "programming"]),
]

# Categories we predict (and emit in structured mock responses)
WIKI_CATEGORIES: List[str] = [
    "Science", "Sports", "Arts", "History", "Geography", "Technology"
]


class WikiCategoryMockLLM(MockLLM):
    def prompt(self, prompt_list):
        vals = []
        for _ in prompt_list:
            vals.append({
                "response": {k: float(self._rng.uniform(self.low, self.high)) for k in WIKI_CATEGORIES}
            })
        return vals


class GeminiCategoryLLM:
    """Gemini-backed LLM that returns {'response': {category: score, ...}} and tracks tokens."""
    def __init__(self, api_key: str, model: str | None = None, categories: List[str] | None = None):
        import google.generativeai as genai  # type: ignore
        self._genai = genai
        self._api_key = api_key
        self._model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._model = genai.GenerativeModel(self._model_name)
        self._categories = categories or WIKI_CATEGORIES
        self._prompt_tokens = 0
        self._output_tokens = 0

    def initialize_llm(self):
        # configure each time in case env changed
        try:
            self._genai.configure(api_key=self._api_key)
        except Exception:
            pass
        return self

    def prompt(self, prompt_list: List[Any]):
        outs = []
        for p in prompt_list:
            text = p if isinstance(p, str) else p.get("agent_query", "")
            resp = self._model.generate_content(text)
            um = getattr(resp, "usage_metadata", None)
            if um is not None:
                self._prompt_tokens += int(getattr(um, "prompt_token_count", 0) or 0)
                self._output_tokens += int(getattr(um, "candidates_token_count", 0) or 0)
            # Extract text robustly from response
            cand_text = getattr(resp, "text", None)
            if not cand_text:
                parts_text: List[str] = []
                for c in getattr(resp, "candidates", []) or []:
                    content = getattr(c, "content", None)
                    for pt in (getattr(content, "parts", []) or []):
                        t = getattr(pt, "text", None)
                        if t:
                            parts_text.append(t)
                cand_text = "\n".join(parts_text)
            if not cand_text or not str(cand_text).strip():
                raise ValueError("Empty Gemini response text")
            # Parse first JSON object found in text
            import re, json as _json
            s = str(cand_text).strip()
            m = re.search(r"\{[\s\S]*\}", s)
            if not m:
                raise ValueError("No JSON object found in Gemini response")
            parsed_json = _json.loads(m.group(0))
            if not isinstance(parsed_json, dict):
                raise ValueError("Gemini response is not JSON dict")
            parsed: Dict[str, float] = {}
            for k in self._categories:
                parsed[k] = float(parsed_json.get(k, 0.0))
            outs.append({"response": parsed})
        return outs

    def __call__(self, prompt_inputs: List[Any]):
        return self.prompt(prompt_inputs)

    def total_tokens(self) -> int:
        return int(self._prompt_tokens + self._output_tokens)


def _heuristic_category(text: str) -> str:
    t = (text or "").lower()
    best = "General"
    best_hits = 0
    for label, kw in WIKI_KEYWORDS:
        hits = sum(1 for k in kw if k in t)
        if hits > best_hits:
            best, best_hits = label, hits
    return best


def _build_fewshot_demos(df: pd.DataFrame, k: int) -> List[str]:
    demos: List[str] = []
    n = min(k, len(df))
    for i in range(n):
        row = df.iloc[i]
        content = _truncate(str(row.get("text", "")), 400)
        label = _heuristic_category(content)
        demos.append(f"Content:\n{content}\n\nCategory: {label}")
    return demos


def _build_dspy_module(demos: List[str]):
    import dspy  # type: ignore

    class WikiCategorySignature(dspy.Signature):
        content: str = dspy.InputField(desc="Wikipedia article content")
        category: str = dspy.OutputField(desc="Coarse topic label for the article")

    class WikiClassifier(dspy.Module):
        def __init__(self, instruction: str, demo_texts: List[str]):
            super().__init__()
            self.predictor = dspy.Predict(WikiCategorySignature)
            # DSPy-friendly scaffold attributes
            self.instruction = instruction
            self.demos = demo_texts

        def forward(self, content: str) -> Any:
            return self.predictor(content=content)

    instruction = (
        "Determine the high-level category of the following Wikipedia article. "
        "Choose a concise label like Science, Sports, Arts, History, Geography, or Technology."
    )
    return WikiClassifier(instruction, demos)


def _build_train_val_examples(df: pd.DataFrame, max_train: int = 60) -> Tuple[List[Any], List[Any]]:
    import dspy  # type: ignore
    exs: List[Any] = []
    n = min(len(df), max_train)
    for i in range(n):
        text = str(df.iloc[i].get("text", ""))
        label = _heuristic_category(text)
        ex = dspy.Example(content=_truncate(text, 800), category=label).with_inputs("content")
        exs.append(ex)
    # 70/30 split (small)
    split = max(1, int(0.7 * len(exs)))
    return exs[:split], exs[split: max(split + 10, len(exs))]


def _extract_optimized_instruction(owner: Any, fallback: str = "") -> str:
    try:
        # 1) direct
        val = getattr(owner, "instruction", None)
        if isinstance(val, (list, tuple)):
            val = "\n".join(str(x) for x in val)
        if isinstance(val, str) and val.strip():
            return val
        # 2) prompt_model.kwargs
        pm = getattr(owner, "prompt_model", None)
        kw = getattr(pm, "kwargs", {}) if pm is not None else {}
        instr = kw.get("instruction")
        if isinstance(instr, str) and instr.strip():
            return instr
        # 3) lm.kwargs
        lm = getattr(owner, "lm", None)
        kw2 = getattr(lm, "kwargs", {}) if lm is not None else {}
        instr2 = kw2.get("instruction")
        if isinstance(instr2, str) and instr2.strip():
            return instr2
    except Exception:
        pass
    return fallback


def _compile_with_mipro(module, df: pd.DataFrame) -> Any:
    import dspy  # type: ignore
    from dspy.teleprompt import MIPROv2  # type: ignore
    def accuracy_metric(example, pred, trace=None) -> float:
        try:
            p = str(getattr(pred, "category", "")).strip().lower()
            g = str(getattr(example, "category", "")).strip().lower()
            return 1.0 if p and g and p == g else 0.0
        except Exception:
            return 0.0

    trainset, valset = _build_train_val_examples(df, max_train=80)
    
    optimizer = MIPROv2(metric=accuracy_metric, auto="light", num_threads=2)
    try:
        return optimizer.compile(
            student=module,
            trainset=trainset,
            valset=valset or trainset[:1],
            requires_permission_to_run=False,
        )
    except Exception as e:
        # No LM configured or proposer errors → proceed with the unoptimized module
        print(f"DSPy compile unavailable ({e}). Using base module without DSPy optimization.")
        return module


def _evaluate_dspy_module(module: Any, examples: List[Any], metric_fn) -> Tuple[float, int]:
    """Return (average_metric, token_estimate) over examples using the module's forward."""
    owner = getattr(module, "predictor", module)
    total = 0.0
    n = 0
    token_est = 0
    # Extract scaffold for token estimate
    instr = getattr(owner, "instruction", "") or ""
    demos = getattr(owner, "demos", None) or getattr(owner, "fewshot", None) or []
    demo_text = "\n".join(str(d) for d in demos) if isinstance(demos, (list, tuple)) else ""
    for ex in examples:
        try:
            pred = owner(content=getattr(ex, "content"))
            total += float(metric_fn(ex, pred))
            n += 1
            content = str(getattr(ex, "content", ""))
            approx_text = f"{instr}\n{demo_text}\n{content}"
            token_est += max(1, len(approx_text) // 4)
        except Exception:
            pass
    avg = (total / n) if n else 0.0
    return avg, token_est


    # removed: inlined into main() to reduce verbosity and keep a single flow


def _estimate_dspy_compile_tokens(df: pd.DataFrame, system_prompt: str, auto_mode: str = "medium") -> int:
    """Heuristically estimate DSPy compile token usage based on mode and dataset size.

    Assumptions (override via env):
      - trials per mode: light=10, medium=20, heavy=30
      - fewshot candidates: light=6, medium=10, heavy=12
      - instruction candidates: light=3, medium=5, heavy=6
      - each evaluation uses valset_size prompts
      - tokens ≈ characters/4
    """
    trials_map = {"light": 10, "medium": 20, "heavy": 30}
    fewshot_map = {"light": 6, "medium": 10, "heavy": 12}
    instruct_map = {"light": 3, "medium": 5, "heavy": 6}
    mode = str(auto_mode or "medium").lower()
    trials = int(os.getenv("MIPRO_TRIALS", trials_map.get(mode, 20)))
    few_cands = int(os.getenv("MIPRO_FEWSHOT_CANDS", fewshot_map.get(mode, 10)))
    instr_cands = int(os.getenv("MIPRO_INSTRUCT_CANDS", instruct_map.get(mode, 5)))

    # Build small valset as DSPy does (we reuse our helper for an estimate)
    try:
        _, valset = _build_train_val_examples(df, max_train=80)
        valset_size = max(1, len(valset) or 24)
    except Exception:
        valset_size = 24

    # Estimate tokens per prompt: system + average content
    try:
        avg_n = min(30, len(df))
        contents = (df.get("content") if "content" in df.columns else df.get("text", "")).astype(str).head(avg_n).tolist()
    except Exception:
        contents = []
    avg_content_chars = sum(len(c) for c in contents) / max(1, len(contents)) if contents else 400.0
    sys_chars = len(system_prompt or "")
    est_chars_per_prompt = sys_chars + avg_content_chars
    est_tokens_per_prompt = max(1, int(est_chars_per_prompt // 4))

    # Total eval prompts across trials and candidate combinations
    eval_calls = trials * few_cands * instr_cands * valset_size
    # Add bootstrap overhead ~25%
    total_tokens = int(est_tokens_per_prompt * eval_calls * 1.25)
    return total_tokens


@dataclass
class WikiRunConfig:
    wiki_dir: str
    max_docs: int = 200
    demo_count: int = 6
    marker: str = "<<ATTRIBUTES>>"
    gemini_model: str = "gemini-2.0-flash"
    eval_llm_key: Optional[str] = None
    p3o_llm_key: Optional[str] = None
    exploration: str = "balanced"
    total_steps: int = 100
    batch_cap: int = 50
    slots: List[str] = field(default_factory=lambda: [
        "UseDefinitions",
        "IncludeDates",
        "HighlightNamedEntities",
        "IncludeLocations",
        "SummarizeKeyEvents",
        "MentionNotablePeople",
        "ListRelatedConcepts",
    ])
    categories: List[str] = field(default_factory=lambda: WIKI_CATEGORIES.copy())


def _build_config_from_env() -> WikiRunConfig:
    wiki_dir = os.getenv("WIKI_DIR", r"c:\\Users\\ayanp\\Downloads\\enwiki_namespace_0")
    max_docs = int(os.getenv("MAX_WIKI_DOCS", "200"))
    demo_count = int(os.getenv("DEMO_COUNT", "6"))
    marker = os.getenv("RR_MARKER", "<<ATTRIBUTES>>")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    eval_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    p3o_key = eval_key
    exploration = "balanced"
    total_steps = 100
    batch_cap = 50
    return WikiRunConfig(
        wiki_dir=wiki_dir,
        max_docs=max_docs,
        demo_count=demo_count,
        marker=marker,
        gemini_model=gemini_model,
        eval_llm_key=eval_key,
        p3o_llm_key=p3o_key,
        exploration=exploration,
        total_steps=total_steps,
        batch_cap=batch_cap,
    )


def _select_eval_llm(cfg: WikiRunConfig):
    if cfg.eval_llm_key:
        try:
            return GeminiCategoryLLM(api_key=cfg.eval_llm_key, model=cfg.gemini_model, categories=cfg.categories).initialize_llm()
        except Exception:
            return WikiCategoryMockLLM(low=0, high=100, seed=0)
    return WikiCategoryMockLLM(low=0, high=100, seed=0)


def _select_p3o_llm(cfg: WikiRunConfig) -> Tuple[Any, bool]:
    if cfg.p3o_llm_key:
        try:
            return GeminiCategoryLLM(api_key=cfg.p3o_llm_key, model=cfg.gemini_model, categories=cfg.categories).initialize_llm(), False
        except Exception:
            return WikiCategoryMockLLM(low=0, high=100, seed=0), True
    return WikiCategoryMockLLM(low=0, high=100, seed=0), True


def _build_full_prompt(instruction: str, demos: List[str], attributes: List[str], content: str, categories: List[str]) -> str:
    fewshot_txt = ("Few-shot examples:\n" + "\n".join(str(d) for d in demos) + "\n\n") if demos else ""
    full_attr_block = "Attributes:\n" + "\n".join(f"- {s}" for s in attributes)
    json_keys = "\n".join([f'  "{c}": <YOUR_NUMERIC_PREDICTION>{"," if i < len(categories)-1 else ""}' for i, c in enumerate(categories)])
    out_json_txt = "ONLY OUTPUT THIS JSON. DO NOT OUTPUT ANYTHING ELSE!!!!\n\n{\n" + json_keys + "\n}"
    return f"{instruction}\n\nInputs: content\n\n{fewshot_txt}{full_attr_block}\n\n{content}\n\n{out_json_txt}"


def _build_baseline_prompts(owner: Any, slots: List[str], df: pd.DataFrame, max_rows: int, categories: List[str]) -> Tuple[List[str], List[str], str]:
    instr_full = str(getattr(owner, "instruction", ""))
    demos_full = getattr(owner, "demos", None) or getattr(owner, "fewshot", None) or []
    prompts: List[str] = []
    group_keys: List[str] = []
    for i in range(max_rows):
        content_i = str(df.iloc[i]["content"])
        prompts.append(_build_full_prompt(instr_full, list(demos_full) if isinstance(demos_full, (list, tuple)) else [], slots, content_i, categories))
        group_keys.append(f"job_{i}")
    first = prompts[0] if prompts else ""
    return prompts, group_keys, first


def _build_hybrid_prompts(owner: Any, selected_labels: List[str], df: pd.DataFrame, max_rows: int, categories: List[str]) -> List[str]:
    instr_full = str(getattr(owner, "instruction", ""))
    demos_full = getattr(owner, "demos", None) or getattr(owner, "fewshot", None) or []
    prompts: List[str] = []
    for i in range(max_rows):
        content_i = str(df.iloc[i]["content"])
        prompts.append(_build_full_prompt(instr_full, list(demos_full) if isinstance(demos_full, (list, tuple)) else [], selected_labels, content_i, categories))
    return prompts


def _load_wiki_df(cfg: WikiRunConfig) -> pd.DataFrame:
    if not os.path.isdir(cfg.wiki_dir):
        raise FileNotFoundError(f"WIKI_DIR not found: {cfg.wiki_dir}")
    print(f"Loading Wikipedia sample from: {cfg.wiki_dir}")
    rows = _stream_jsonl_sample(cfg.wiki_dir, limit=cfg.max_docs)
    if not rows:
        raise RuntimeError("No documents loaded from WIKI_DIR")
    df = pd.DataFrame(rows)
    def _row_to_content(row: pd.Series) -> str:
        parts = []
        for col in row.index:
            try:
                val = row[col]
                if isinstance(val, (list, dict)):
                    val = json.dumps(val)
                parts.append(f"{col}: {val}")
            except Exception:
                continue
        return "\n".join(parts)
    df = df.copy()
    df["content"] = df.apply(_row_to_content, axis=1)
    print(f"Loaded {len(df)} docs. Building DSPy module…")
    return df


def _configure_dspy(cfg: WikiRunConfig) -> None:
    try:
        import dspy  # type: ignore
        force_mock = str(os.getenv("DSPY_MOCK", "0")).strip().lower() in ("1", "true", "yes")
        if force_mock:
            try:
                dspy.configure(lm=dspy.MockLM())
            except Exception:
                pass
        elif cfg.eval_llm_key:
            model = cfg.gemini_model
            try:
                dspy.configure(lm=dspy.LM(f"gemini/{model}", api_key=cfg.eval_llm_key))
            except Exception:
                try:
                    from dspy import LM as _LM  # type: ignore
                    dspy.configure(lm=_LM(f"gemini/{model}", api_key=cfg.eval_llm_key))
                except Exception:
                    pass
    except Exception:
        pass


def _compute_dspy_baseline(optimized_module: Any, df: pd.DataFrame, cfg: WikiRunConfig):
    eval_llm = _select_eval_llm(cfg)
    eval_template = from_predict(
        optimized_module,
        slots=cfg.slots,
        title="Wiki Category Classification (Eval)",
        categories=cfg.categories,
        marker=cfg.marker,
        include_attributes_block=True,
        block_header="Attributes:",
    )
    eval_template.configure(external_df=df)

    max_rows = min(cfg.batch_cap, len(df))
    owner_for_full = getattr(optimized_module, "predictor", optimized_module)
    dspy_full_prompts, group_keys, dspy_full_prompt_preview = _build_baseline_prompts(owner_for_full, cfg.slots, df, max_rows, cfg.categories)

    outputs = eval_llm(dspy_full_prompts)
    arch_eval = Archetype(prompt=eval_template, llm=eval_llm, n_arch=1)
    arch_eval.configure(external_df=df)
    opt_eval = P3O(archetype=arch_eval, verbose=False)
    rewards_accum: List[float] = []
    dspy_full_predicted_categories: List[str] = []
    for k, out in zip(group_keys, outputs):
        if not isinstance(out, dict) or "response" not in out:
            raise ValueError("LLM output missing 'response' dict")
        structured = out["response"]
        try:
            predicted_label = max(cfg.categories, key=lambda c: float(structured.get(c, 0.0)))
        except Exception:
            predicted_label = ""
        dspy_full_predicted_categories.append(predicted_label)
        _, reward, _, *_ = opt_eval._default_expt2_pipeline(k, structured, arch_eval)
        rewards_accum.append(float(reward))
    dspy_full_reward = float(sum(rewards_accum) / max(1, len(rewards_accum)))
    dspy_full_tokens = int(getattr(eval_llm, "total_tokens", lambda: 0)())
    try:
        sys_prompt_for_est = str(eval_template.__system_prompt__()) if hasattr(eval_template, "__system_prompt__") else ""
    except Exception:
        sys_prompt_for_est = ""
    dspy_compile_token_estimate = _estimate_dspy_compile_tokens(df, sys_prompt_for_est, auto_mode="light")
    dspy_total_tokens_estimate = int(dspy_compile_token_estimate + dspy_full_tokens)

    return {
        "eval_llm": eval_llm,
        "eval_template": eval_template,
        "max_rows": max_rows,
        "group_keys": group_keys,
        "dspy_full_reward": dspy_full_reward,
        "dspy_full_tokens": dspy_full_tokens,
        "dspy_full_prompt_preview": dspy_full_prompt_preview,
        "dspy_compile_token_estimate": dspy_compile_token_estimate,
        "dspy_total_tokens_estimate": dspy_total_tokens_estimate,
        "dspy_full_predicted_categories": dspy_full_predicted_categories,
    }


def _build_template_for_optimization(optimized_module: Any, df: pd.DataFrame, cfg: WikiRunConfig):
    template = from_predict(
        optimized_module,
        slots=cfg.slots,
        title="Wiki Category Classification",
        categories=cfg.categories,
        marker=cfg.marker,
        include_attributes_block=True,
        block_header="Attributes:",
    )
    template.configure(external_df=df)
    return template


def _run_p3o(template, df: pd.DataFrame, cfg: WikiRunConfig, args) -> Tuple[P3O, List[Dict[str, Any]], Any, bool, int, int, Dict[str, Any]]:
    if args.p3o_mock:
        llm = WikiCategoryMockLLM(low=0, high=100, seed=0)
        is_p3o_mock = True
    else:
        llm, is_p3o_mock = _select_p3o_llm(cfg)
    arch = Archetype(prompt=template, llm=llm, n_arch=2)
    arch.configure(external_df=df)
    opt = P3O(archetype=arch, verbose=True)
    batch_size = min(cfg.batch_cap, len(df))
    print("\nRunning P3O (mode=quick)…")
    planned_steps = int(get_exploration_config(cfg.exploration, total_steps=cfg.total_steps).get("steps", 30))
    history = opt.train(mode=cfg.exploration, steps=planned_steps, log_interval=1, batch_size=batch_size)
    print("Selections:", opt.get_p3o_selections())
    saved_paths: Dict[str, Any] = {}
    try:
        saved_paths = opt.save_step_results("final")
    except Exception:
        saved_paths = {}
    if not is_p3o_mock and isinstance(llm, GeminiCategoryLLM):
        hybrid_tokens_est_total = int(llm.total_tokens())
    else:
        behavior = getattr(arch, "_behavior", None) or getattr(arch, "_mock_behavior", None)
        prompts = getattr(behavior, "last_prompt_list", []) or []
        if not prompts:
            try:
                sample_txt = template.render(agent_id=0, population=None, mapping={}, config_kwargs={})
                prompts = [sample_txt]
            except Exception:
                prompts = []
        tokens_last_step = sum(max(1, len(str(p)) // 4) for p in prompts)
        steps_taken = max(1, len(history))
        hybrid_tokens_est_total = int(tokens_last_step * steps_taken)
    return opt, history, llm, is_p3o_mock, hybrid_tokens_est_total, planned_steps, saved_paths


def _evaluate_hybrid(optimized_module: Any, opt: P3O, df: pd.DataFrame, cfg: WikiRunConfig, eval_llm, eval_template, group_keys: List[str], max_rows: int):
    final_choices = opt.get_p3o_selections()
    selected_labels = [lbl for lbl in cfg.slots if final_choices.get(lbl.lower().replace(" ", "_"), 0) == 1]
    if not selected_labels:
        try:
            from agent_torch.integrations.dspy_to_template import _clean_skill_name as _clean
            selected_labels = [lbl for lbl in cfg.slots if final_choices.get(_clean(lbl), 0) == 1]
        except Exception:
            selected_labels = []
    owner_for_full = getattr(optimized_module, "predictor", optimized_module)
    hybrid_prompts: List[str] = _build_hybrid_prompts(owner_for_full, selected_labels, df, max_rows, cfg.categories)
    get_tokens = getattr(eval_llm, "total_tokens", None)
    before_tokens = int(get_tokens()) if callable(get_tokens) else 0
    hybrid_outputs = eval_llm(hybrid_prompts)
    arch_h_eval = Archetype(prompt=eval_template, llm=eval_llm, n_arch=1)
    arch_h_eval.configure(external_df=df)
    opt_h_eval = P3O(archetype=arch_h_eval, verbose=False)
    hybrid_rewards_accum: List[float] = []
    hybrid_predicted_categories: List[str] = []
    for k, out in zip(group_keys, hybrid_outputs):
        if not isinstance(out, dict) or "response" not in out:
            raise ValueError("LLM output missing 'response' dict")
        structured = out["response"]
        try:
            predicted_label = max(cfg.categories, key=lambda c: float(structured.get(c, 0.0)))
        except Exception:
            predicted_label = ""
        hybrid_predicted_categories.append(predicted_label)
        _, r, _, *_ = opt_h_eval._default_expt2_pipeline(k, structured, arch_h_eval)
        hybrid_rewards_accum.append(float(r))
    hybrid_reward_raw = float(sum(hybrid_rewards_accum) / max(1, len(hybrid_rewards_accum)))
    after_tokens = int(get_tokens()) if callable(get_tokens) else 0
    hybrid_eval_tokens = (after_tokens - before_tokens) if callable(get_tokens) else int(sum(max(1, len(p) // 4) for p in hybrid_prompts))
    return selected_labels, hybrid_reward_raw, hybrid_eval_tokens, hybrid_predicted_categories


def _write_summary(cfg: WikiRunConfig,
                   dspy_module: Any,
                   demos: List[str],
                   optimized_module: Any,
                   template: Any,
                   opt: P3O,
                   history: List[Dict[str, Any]],
                   dspy_full_prompt_preview: str,
                   dspy_full_reward: float,
                   dspy_full_tokens: int,
                   dspy_compile_token_estimate: int,
                   dspy_total_tokens_estimate: int,
                   exploration: str,
                   planned_steps: int,
                   hybrid_reward: float,
                   hybrid_tokens_est_total: int,
                   hybrid_reward_raw: float,
                   hybrid_eval_tokens: int,
                   saved_paths: Dict[str, Any]) -> None:
    try:
        summary_dir = os.path.join("experiments", "wiki_exp")
        os.makedirs(summary_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(summary_dir, f"run_summary_{stamp}.json")
        try:
            step_data = opt.get_current_step_data()
            p3o_output_template = step_data.get("output_template", "")
        except Exception:
            p3o_output_template = ""
        try:
            final_choices = opt.get_p3o_selections()
            selected_labels = [lbl for lbl in cfg.slots if final_choices.get(lbl.lower().replace(" ", "_"), 0) == 1]
            if not selected_labels:
                from agent_torch.integrations.dspy_to_template import _clean_skill_name as _clean
                selected_labels = [lbl for lbl in cfg.slots if final_choices.get(_clean(lbl), 0) == 1]
            lines = ["Attributes:"] + [f"- {lbl}" for lbl in selected_labels]
            explicit_block = "\n".join(lines)
            sys_txt = str(template.__system_prompt__()) if hasattr(template, '__system_prompt__') else ""
            out_txt = str(template.__output__()) if hasattr(template, '__output__') else ""
            rendered_prompt = " ".join([p for p in (sys_txt, explicit_block, out_txt) if p]).strip()
        except Exception:
            rendered_prompt = ""
        try:
            _owner = getattr(optimized_module, "predictor", optimized_module)
            _instr_raw = getattr(_owner, "instruction", None)
            if isinstance(_instr_raw, (list, tuple)):
                dspy_instruction_optimized = "\n".join(str(x) for x in _instr_raw)
            else:
                dspy_instruction_optimized = str(_instr_raw or "")
            _demos = getattr(_owner, "demos", None)
            if _demos is None:
                _demos = getattr(_owner, "fewshot", None)
            dspy_demos_optimized_full = [str(d) for d in _demos] if isinstance(_demos, (list, tuple)) else []
            dspy_num_demos_optimized = len(dspy_demos_optimized_full)
            dspy_demos_optimized = dspy_demos_optimized_full[:6]
        except Exception:
            dspy_instruction_optimized = ""
            dspy_demos_optimized = []
            dspy_num_demos_optimized = 0
        try:
            template_system_prompt = str(template.__system_prompt__()) if hasattr(template, "__system_prompt__") else ""
        except Exception:
            template_system_prompt = ""
        summary = {
            "dspy_meta": {
                "instruction": getattr(dspy_module, "instruction", ""),
                "num_demos": len(demos),
                "full_prompt_preview": _truncate(dspy_full_prompt_preview, 2000),
            },
            "dspy_instruction": getattr(dspy_module, "instruction", ""),
            "dspy_num_demos": len(demos),
            "dspy_instruction_optimized": dspy_instruction_optimized,
            "dspy_num_demos_optimized": dspy_num_demos_optimized,
            "dspy_demos_optimized": dspy_demos_optimized,
            "template_system_prompt": _truncate(template_system_prompt, 2000),
            "template_base_preview": _truncate(template.get_base_prompt_manager_template(), 2000),
            "p3o_history_len": len(history),
            "p3o_last_step": history[-1] if history else {},
            "p3o_selections": opt.get_p3o_selections(),
            "prompt_with_selections": rendered_prompt,
            "p3o_output_template": p3o_output_template,
            "optimizer_results_file": saved_paths.get("results"),
            "dspy_full_reward": float(dspy_full_reward),
            "dspy_full_tokens": int(dspy_full_tokens),
            "dspy_reward_p3o": float(dspy_full_reward),
            "dspy_token_estimate_p3o": int(dspy_full_tokens),
            "dspy_compile_token_estimate": int(dspy_compile_token_estimate),
            "dspy_total_token_estimate": int(dspy_total_tokens_estimate),
            "p3o_mode_used": exploration,
            "p3o_steps": int(planned_steps),
            "hybrid_reward": hybrid_reward,
            "hybrid_token_estimate": int(hybrid_tokens_est_total),
            "hybrid_reward_raw": float(hybrid_reward_raw),
            "hybrid_eval_tokens": int(hybrid_eval_tokens),
            "hybrid_cost_estimate": None,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(summary, f, indent=2)
        print(f"Summary saved: {summary_path}")
    except Exception:
        pass

#--------------------------------Main Function--------------------------------
def main() -> None:

    cfg = _build_config_from_env()

    df = _load_wiki_df(cfg)
    # Quick verification that extraction produced expected fields
    try:
        n_rows = len(df)
        id_ok = int(df["id"].astype(str).str.strip().ne("").sum()) if "id" in df.columns else 0
        title_ok = int(df["title"].astype(str).str.strip().ne("").sum()) if "title" in df.columns else 0
        text_ok = int(df["text"].astype(str).str.strip().ne("").sum()) if "text" in df.columns else 0
        url_ok = int(df["url"].astype(str).str.strip().ne("").sum()) if "url" in df.columns else 0
        print(f"Extraction check → rows={n_rows}, id={id_ok}/{n_rows}, title={title_ok}/{n_rows}, text={text_ok}/{n_rows}, url={url_ok}/{n_rows}")
        if n_rows > 0:
            sample = df.iloc[0]
            preview = str(sample.get("text", ""))[:160]
            print(f"Sample row → id='{sample.get('id','')}', title='{sample.get('title','')}', url='{sample.get('url','')}', text='{preview}'")
    except Exception:
        pass

    # Few-shot scaffold and DSPy module
    demos = _build_fewshot_demos(df, k=cfg.demo_count)
    dspy_module = _build_dspy_module(demos)

    _configure_dspy(cfg)

    optimized_module = _compile_with_mipro(dspy_module, df)

    # Log what DSPy produces (for debugging)
    try:
        _owner = getattr(optimized_module, "predictor", optimized_module)
        # Promote optimized instruction if available but not copied to predictor
        baseline_instr = getattr(dspy_module, "instruction", "")
        chosen_instr = _extract_optimized_instruction(_owner, fallback=str(baseline_instr or ""))
        if isinstance(chosen_instr, str) and chosen_instr.strip():
            try:
                setattr(_owner, "instruction", chosen_instr)
            except Exception:
                pass
        print("DSPy optimized instruction:", getattr(_owner, "instruction", ""))
        _odemos = getattr(_owner, "demos", None) or getattr(_owner, "fewshot", None) or []
        print("DSPy optimized demos:", len(_odemos) if isinstance(_odemos, (list, tuple)) else 0)
    except Exception:
        pass

    baseline = _compute_dspy_baseline(optimized_module, df, cfg)

    eval_llm = baseline["eval_llm"]
    eval_template = baseline["eval_template"]
    max_rows = baseline["max_rows"]
    group_keys = baseline["group_keys"]
    dspy_full_reward = baseline["dspy_full_reward"]
    dspy_full_tokens = baseline["dspy_full_tokens"]
    dspy_full_prompt_preview = baseline["dspy_full_prompt_preview"]
    dspy_compile_token_estimate = baseline["dspy_compile_token_estimate"]
    dspy_total_tokens_estimate = baseline["dspy_total_tokens_estimate"]
    dspy_full_predicted_categories = baseline["dspy_full_predicted_categories"]
    dspy_reward_p3o = dspy_full_reward
    dspy_token_estimate_p3o = dspy_full_tokens

    # ---------------------------------------------------------------------------------------------------------------
    # Build Template directly from the DSPy module 
    template = from_predict(
        optimized_module,
        slots=cfg.slots,
        title="Wiki Category Classification",
        categories=cfg.categories,
        marker=cfg.marker,
        include_attributes_block=True,
        block_header="Attributes:",
    )
    template.configure(external_df=df)

    # CLI: allow forcing mock only for P3O; DSPy eval still auto-detects
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--p3o-mock", action="store_true", help="Use mock LLM for P3O irrespective of API key")
    args, _ = parser.parse_known_args()

    # Create the P3O optimizer and run the hybrid experiment (explicit in main)
    if args.p3o_mock:
        llm = WikiCategoryMockLLM(low=0, high=100, seed=0)
        is_p3o_mock = True
    else:
        llm, is_p3o_mock = _select_p3o_llm(cfg)

    # Create the Archetype and P3O optimizer
    arch = Archetype(prompt=template, llm=llm, n_arch=2)
    arch.configure(external_df=df)
    opt = P3O(archetype=arch, verbose=True)

    #Run the P3O optimizer
    batch_size = min(cfg.batch_cap, len(df))
    print("\nRunning P3O (mode=quick)…")
    planned_steps = int(get_exploration_config(cfg.exploration, total_steps=cfg.total_steps).get("steps", 30))
    history = opt.train(mode=cfg.exploration, steps=planned_steps, log_interval=1, batch_size=batch_size)
    
    print("Selections:", opt.get_p3o_selections())
    saved_paths = {}
    try:
        saved_paths = opt.save_step_results("final")
    except Exception:
        saved_paths = {}
    # Token estimate for hybrid phase
    if not is_p3o_mock and isinstance(llm, GeminiCategoryLLM):
        hybrid_tokens_est_total = int(llm.total_tokens())
    else:
        behavior = getattr(arch, "_behavior", None) or getattr(arch, "_mock_behavior", None)
        prompts = getattr(behavior, "last_prompt_list", []) or []
        if not prompts:
            try:
                sample_txt = template.render(agent_id=0, population=None, mapping={}, config_kwargs={})
                prompts = [sample_txt]
            except Exception:
                prompts = []
        tokens_last_step = sum(max(1, len(str(p)) // 4) for p in prompts)
        steps_taken = max(1, len(history))
        hybrid_tokens_est_total = int(tokens_last_step * steps_taken)

    hybrid_reward = float(history[-1].get("reward", 0.0)) if history else 0.0
    cost_per_1k = float(os.getenv("GEMINI_COST_PER_1K", "0"))
    hybrid_cost_estimate = (hybrid_tokens_est_total / 1000.0) * cost_per_1k if cost_per_1k > 0 else None

    selected_labels, hybrid_reward_raw, hybrid_eval_tokens, hybrid_predicted_categories = _evaluate_hybrid(
        optimized_module, opt, df, cfg, eval_llm, eval_template, group_keys, max_rows
    )

    _write_summary(
        cfg=cfg,
        dspy_module=dspy_module,
        demos=demos,
        optimized_module=optimized_module,
        template=template,
        opt=opt,
        history=history,
        dspy_full_prompt_preview=dspy_full_prompt_preview,
        dspy_full_reward=dspy_full_reward,
        dspy_full_tokens=dspy_full_tokens,
        dspy_compile_token_estimate=dspy_compile_token_estimate,
        dspy_total_tokens_estimate=dspy_total_tokens_estimate,
        exploration=cfg.exploration,
        planned_steps=planned_steps,
        hybrid_reward=hybrid_reward,
        hybrid_tokens_est_total=hybrid_tokens_est_total,
        hybrid_reward_raw=hybrid_reward_raw,
        hybrid_eval_tokens=hybrid_eval_tokens,
        saved_paths=saved_paths,
    )

    print("Done. Use analyze_results.py to generate comparison plots.")


if __name__ == "__main__":
    main()


