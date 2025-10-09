"""
Baseline hybrid experiment (DSPy-first → P3O), API-style similar to expt_api.py

Flow:
 1) Load exp2-style data (skills universe + sparse job vectors)
 2) Run DSPy light optimization on JobAnalysisModule (prompt-opt)
 3) Convert optimized DSPy module to an AgentTorch Template (scaffold embedded, slots=x-attributes)
 4) Run P3O with long exploration mode to optimize slot inclusion

Usage:
  python hybrid_expt__baseline_api.py

Env:
  - GOOGLE_API_KEY (optional) to use Gemini for P3O; otherwise uses JobKnowledgeMockLLM
  - DSPY_AUTO (defaults to light) for adapter phase if needed
  - P3O_MODE (use quick|balanced|aggressive|long). Defaults to long in this script
"""

from typing import List, Dict, Any, Optional

import os
import sys
import json
import pandas as pd
import torch
import dspy

from agent_torch.integrations.dspy_to_template import from_predict
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.mock_llm import MockLLM
from agent_torch.optim.p3o import P3O


# --- Local experiment paths (experiments/hybrid) ---
HYBRID_ROOT = os.path.abspath(os.path.dirname(__file__))
EXPT2_ROOT = os.path.join(HYBRID_ROOT, "expt2")
if HYBRID_ROOT not in sys.path:
    sys.path.append(HYBRID_ROOT)
if EXPT2_ROOT not in sys.path:
    sys.path.append(EXPT2_ROOT)

# DSPy program will be imported inside main after sys.path is configured


KNOWLEDGE_CATEGORIES = [
    "AdministrationAndManagement", "Administrative", "Biology", "BuildingAndConstruction",
    "Chemistry", "CommunicationsAndMedia", "ComputersAndElectronics", "CustomerAndPersonalService",
    "Design", "EconomicsAndAccounting", "EducationAndTraining", "EngineeringAndTechnology",
    "EnglishLanguage", "FineArts", "FoodProduction", "ForeignLanguage", "Geography",
    "HistoryAndArcheology", "LawAndGovernment", "Mathematics", "Mechanical", "MedicineAndDentistry",
    "PersonnelAndHumanResources", "PhilosophyAndTheology", "Physics", "ProductionAndProcessing",
    "Psychology", "PublicSafetyAndSecurity", "SalesAndMarketing", "SociologyAndAnthropology",
    "Telecommunications", "TherapyAndCounseling", "Transportation"
]


class JobKnowledgeMockLLM(MockLLM):
    def prompt(self, prompt_list):
        vals = []
        for _ in prompt_list:
            vals.append({"response": {k: float(self._rng.uniform(self.low, self.high)) for k in KNOWLEDGE_CATEGORIES}})
        return vals


def _clean_skill_name(skill_name: str) -> str:
    import re
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', skill_name.lower())
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    if cleaned and cleaned[0].isdigit():
        cleaned = 'skill_' + cleaned
    return cleaned or 'unnamed_skill'


def load_data() -> tuple[pd.DataFrame, List[str]]:
    # Local CSV of all skills
    all_skills_csv = os.path.join(HYBRID_ROOT, "expt2_data", "skill_dimensions_updated", "all_skills.csv")
    skills_df = pd.read_csv(all_skills_csv)
    slot_universe: List[str] = skills_df["primary_skill_name"].dropna().astype(str).unique().tolist()

    # Optional local job vectors JSON; build a minimal DF if missing
    job_vectors_path = os.path.join(HYBRID_ROOT, "expt2_data", "skill_dimensions_updated", "updated_job_vectors.json")
    if os.path.exists(job_vectors_path):
        with open(job_vectors_path, 'r', encoding='utf-8') as f:
            job_vectors = json.load(f)
        job_rows: List[Dict[str, Any]] = []
        for job in job_vectors:
            row = {"soc_code": job.get('onet_code', ''), "job_title": job.get('job_title', '')}
            for skill_name, value in job.get('skill_vector', {}).items():
                row[_clean_skill_name(skill_name)] = value
            job_rows.append(row)
        df = pd.DataFrame(job_rows)
    else:
        # Fallback: make a small empty dataset with zeros for all skills
        n_rows = 60
        base = {"soc_code": [""] * n_rows, "job_title": [""] * n_rows}
        for skill_name in slot_universe:
            base[_clean_skill_name(skill_name)] = [0] * n_rows
        df = pd.DataFrame(base)

    # Ensure all skill columns exist and are filled
    for skill_name in slot_universe:
        col = _clean_skill_name(skill_name)
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)
    return df, slot_universe


def build_categories_from_df(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).startswith('NUMERIC_knowledge_')]


def main() -> None:
    print("Starting baseline hybrid API experiment (DSPy → P3O)...")
    print("=" * 60)

    # Ensure import paths for expt2 and its vendored modules (if any)
    EXPT2_VENDOR_ROOT = os.path.join(EXPT2_ROOT, "experiments")
    for p in (HYBRID_ROOT, EXPT2_ROOT, EXPT2_VENDOR_ROOT):
        if p not in sys.path:
            sys.path.append(p)

    # Load data and slots
    ext_df, slot_universe = load_data()
    print(f"Loaded data: {len(ext_df)} rows, slots={len(slot_universe)}")

    # Import DSPy program now that paths are configured
    from expt2.mipro_skills import JobAnalysisModule, run_mipro_optimization  # type: ignore

    # DSPy compile (light) using the local module's own pipeline. Ensure relative
    # paths like ./expt2_data resolve under experiments/hybrid
    _cwd = os.getcwd()
    os.chdir(HYBRID_ROOT)
    try:
        optimized_module, _examples, _summary = run_mipro_optimization(data_type="exp1")
    finally:
        os.chdir(_cwd)
    compiled = optimized_module
    print("DSPy compile complete (light) via local module")

    # Convert compiled DSPy program to Template
    categories = build_categories_from_df(ext_df) or KNOWLEDGE_CATEGORIES
    template = from_predict(compiled, slots=slot_universe, title="Job Knowledge (Hybrid Baseline)", categories=categories)

    # Choose LLM for P3O
    llm = JobKnowledgeMockLLM(low=0, high=100, seed=0)
    print("Using JobKnowledgeMockLLM for P3O")

    # Archetype + P3O
    arch = Archetype(prompt=template, llm=llm, n_arch=7)
    arch.configure(external_df=ext_df)
    opt = P3O(archetype=arch, verbose=True)
    # Run long exploration by default
    opt.train(mode="long", log_interval=1, batch_size=min(50, len(ext_df)))

    # Summary
    print("\nFinal slot probabilities (first 5):")
    shown = 0
    for name, var in list(template._variables.items()):
        if getattr(var, 'learnable', False):
            param = var.get_parameter(template)
            probs = torch.softmax(param, dim=0)
            print(f"  {name}: {probs.detach().tolist()}")
            shown += 1
            if shown >= 5:
                break
    print("Baseline hybrid API experiment complete.")


if __name__ == "__main__":
    main()


