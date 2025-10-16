# Running an Experiment (Base and Hybrid)

This guide walks you through running the two sample experiments included in the repo:
- Base experiment: `experiments/base/expt_api.py`
- Hybrid baseline: `experiments/hybrid/hybrid_expt_baseline_api.py`

Both experiments ultimately optimize a `Template`’s learnable Variables using P3O. The hybrid experiment additionally compiles a DSPy program first, converts it to a `Template`, and then optimizes with P3O.

---

## Prerequisites

- Python 3.10+
- Dependencies installed (see your project’s setup instructions).
- From the repo root, use module-style execution for reliable imports:

```bash
python -m experiments.base.expt_api
python -m experiments.hybrid.hybrid_expt_baseline_api
```

---

## Data Layout

The experiments expect a small set of inputs. The hybrid baseline uses a local `expt2_data` folder; the base experiment uses code-local references (adjust paths if needed).

### Hybrid experiment data (recommended layout)
Place files under `experiments/hybrid/expt2_data/`:

- `job_data_processed.csv` (numeric job dataset)
  - Required columns:
    - `soc_code`, `job_name`
    - All `NUMERIC_knowledge_<Category>` columns (targets)
    - exp1 text fields: `Tasks`, `TechnologySkills`, `WorkActivities`, `DetailedWorkActivities`, `WorkContext`, `Skills`, `Knowledge`, `Abilities`, `Interests`, `WorkValues`, `WorkStyles`, `RelatedOccupations`, `ProfessionalAssociations`

- `skill_dimensions_updated/all_skills.csv`
  - Required column: `primary_skill_name` (the slot universe)

- `skill_dimensions_updated/updated_job_vectors.json`
  - Array of items with keys: `onet_code`, `job_title`, `skill_vector` (map of skill → 0/1)

### Base experiment data (default references)
The base script references the “prompt‑opt” dataset example in comments. If you’re not using that, adjust the paths in `experiments/base/expt_api.py` to the data you have. The core flow remains the same: build a slot universe, create a Template with dynamic slots, bind data, and run P3O.

---

## 1) Base Experiment (`experiments/base/expt_api.py`)

The base experiment demonstrates:
- Building a Template with dynamic variable creation via `add_slots(...)`.
- Binding to an LLM (mock by default; Gemini optional).
- Training with P3O using built‑in exploration modes.

### Run
From the repo root:
```bash
python -m experiments.base.expt_api
```

### What it does
- Loads a slot universe, creates a `Template` with `add_slots(slots=[...], presentations=["", "- {value}"])`.
- Binds a mock LLM by default; if you set `GOOGLE_API_KEY`, a Gemini LLM can be used.
- Calls `opt.train(...)` with a short schedule to optimize slot inclusion.

### Customize
- LLM:
  - Default: `JobKnowledgeMockLLM(low=0, high=100, seed=0)`
  - Optional: `GOOGLE_API_KEY` environment variable to use Gemini (requires `google-generativeai`)
- P3O training:
  - Use `opt.train(mode="quick"|"balanced"|"aggressive"|"long", batch_size=...)`
  - Or explicitly set `steps` and `exploration` (`quick|balanced|aggressive|long`)
- Batch size, logging:
  - `batch_size` controls groups sampled per step
  - `log_interval` controls print cadence

---

## 2) Hybrid Baseline (`experiments/hybrid/hybrid_expt_baseline_api.py`)

The hybrid baseline mirrors the “DSPy‑first → P3O” flow:
1) Load exp2‑style data (skills universe + job vectors + numeric dataset)
2) Compile a DSPy program (light) using the local module
3) Convert the optimized DSPy `Predict` program to a `Template`
4) Optimize the Template with P3O

### Run
From the repo root:
```bash
python -m experiments.hybrid.hybrid_expt_baseline_api
```

### What it does
- Ensures import paths for `experiments/hybrid/expt2` and runs the local `mipro_skills.py` compile step (light).
- Converts the compiled DSPy program to a `Template` using `from_predict(...)`, passing the slot universe and optional `categories` for strict JSON output keys.
- Calls `opt.train(mode="long", ...)` to optimize slot inclusion over the fixed scaffold.

### Customize
- Data: ensure files exist under `experiments/hybrid/expt2_data/` as described above
- P3O training:
  - `mode`: `quick | balanced | aggressive | long`
    - quick: small number of steps, fast exploitation
    - balanced: moderate exploration vs exploitation
    - aggressive: high exploration, slow decay (useful early)
    - long: longer schedule, late decay (thorough runs)
  - `steps`, `batch_size`, `log_interval`
- LLM:
  - Default: `JobKnowledgeMockLLM(low=0, high=100, seed=0)`
  - Optional: use your own LLM adapter in the same shape (callable or `.prompt` API returning a structured dict under `"response"`)

---

## DSPy → Template Conversion (for both flows)

You can convert a DSPy `Predict(Signature)` or module directly into a `Template` and optimize it with P3O. The conversion preserves scaffold (instruction, demos) and surfaces signature input/output names.

```python
from agent_torch.integrations.dspy_to_template import from_predict, from_dspy

# Predict-like (enforces IO & optional categories for strict JSON)
template = from_predict(
    module=compiled_program,
    slots=slot_universe,
    title="Job Knowledge (Hybrid)",
    categories=knowledge_categories,    # optional strict JSON keys in output
)

# General DSPy module (scaffold, demos carried over)
template2 = from_dspy(
    module=compiled_program,
    slots=slot_universe,
    title="Any DSPy → Template",
)
```

Bind data and call P3O `train(...)` just like any other `Template`.

---

## Troubleshooting

- Import errors when running a script directly:
  - Prefer module‑style execution from the repo root: `python -m experiments.hybrid.hybrid_expt_baseline_api`
- Missing data files:
  - Hybrid expects `experiments/hybrid/expt2_data/{job_data_processed.csv, skill_dimensions_updated/all_skills.csv, skill_dimensions_updated/updated_job_vectors.json}`
  - Double‑check paths and CSV headers
- LLM shape errors:
  - The mock/LLM should return a list of outputs; each output must be a dict with `{"response": { ... numeric values ... }}` for P3O’s default scoring

---

## Next Steps
- Try different P3O modes (`quick`, `balanced`, `aggressive`, `long`) and compare slot probabilities.
- Swap the LLM to your provider adapter (same call signature).
- Extend the Template’s presentations (or add more learnable variables) to explore richer prompt formats.
