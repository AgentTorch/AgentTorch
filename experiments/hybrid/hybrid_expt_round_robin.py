"""
Round-robin hybrid experiment (DSPy ↔ P3O) using marker-based insertion and closed-loop feedback.

Flow per round:
 1) DSPy compile (fresh instance) with examples augmented by last P3O selections
 2) Convert DSPy → Template with marker so attributes render in-place
 3) P3O train with chosen mode/batch_size
 4) Capture selections and warm-start next round by seeding template choices and DSPy examples
"""

from typing import List, Dict

import os
import sys
import json
import pandas as pd
import torch

from agent_torch.integrations.dspy_to_template import (
    from_predict,
    build_selection_block,
    inject_block_into_examples,
    get_input_field_from_module,
)
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.mock_llm import MockLLM
from agent_torch.optim.p3o import P3O


HYBRID_ROOT = os.path.abspath(os.path.dirname(__file__))
DEPS_ROOT = os.path.abspath(os.path.join(HYBRID_ROOT, "..", "dependencies"))
EXPT2_ROOT = os.path.join(DEPS_ROOT, "expt2")
EXPT2_DATA_ROOT = os.path.join(DEPS_ROOT, "expt2_data")

for p in (HYBRID_ROOT, DEPS_ROOT, EXPT2_ROOT):
    if p not in sys.path:
        sys.path.append(p)


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
    all_skills_csv = os.path.join(EXPT2_DATA_ROOT, "skill_dimensions_updated", "all_skills.csv")
    skills_df = pd.read_csv(all_skills_csv)
    slot_universe: List[str] = skills_df["primary_skill_name"].dropna().astype(str).unique().tolist()

    job_vectors_path = os.path.join(EXPT2_DATA_ROOT, "skill_dimensions_updated", "updated_job_vectors.json")
    if os.path.exists(job_vectors_path):
        with open(job_vectors_path, 'r', encoding='utf-8') as f:
            job_vectors = json.load(f)
        rows: List[Dict[str, any]] = []
        for job in job_vectors:
            row = {"soc_code": job.get('onet_code', ''), "job_title": job.get('job_title', '')}
            for skill_name, value in job.get('skill_vector', {}).items():
                row[_clean_skill_name(skill_name)] = value
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        n_rows = 60
        base = {"soc_code": [""] * n_rows, "job_title": [""] * n_rows}
        for s in slot_universe:
            base[_clean_skill_name(s)] = [0] * n_rows
        df = pd.DataFrame(base)

    for s in slot_universe:
        col = _clean_skill_name(s)
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)
    return df, slot_universe


def build_categories_from_df(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).startswith('NUMERIC_knowledge_')]


def main() -> None:

    # Paths for expt2 imports
    EXPT2_VENDOR_ROOT = os.path.join(EXPT2_ROOT, "experiments")
    for p in (HYBRID_ROOT, DEPS_ROOT, EXPT2_ROOT, EXPT2_VENDOR_ROOT, EXPT2_DATA_ROOT):
        if p not in sys.path:
            sys.path.append(p)

    ext_df, slot_universe = load_data()
    # Minimal log: dataset size
    print(f"Data: rows={len(ext_df)}, slots={len(slot_universe)}")

    # Import DSPy components for MiPRO-like compile
    import dspy  # type: ignore
    from dspy.teleprompt import MIPROv2  # type: ignore
    from expt2.mipro_skills import (
        JobAnalysisModule,
        create_training_examples,
        job_metrics_metric_mipro,
    )  # type: ignore

    rounds = int(os.getenv("RR_ROUNDS", "2"))
    mode_schedule = os.getenv("RR_MODES", "balanced,quick").split(",")
    marker = os.getenv("RR_MARKER", "<<ATTRIBUTES>>")
    batch_size = min(50, len(ext_df))

    last_selections: Dict[str, int] | None = None

    for r in range(rounds):
        mode = mode_schedule[r] if r < len(mode_schedule) else mode_schedule[-1]
        print(f"\nRound {r+1}/{rounds} | mode={mode}")

        # DSPy compile (fresh each round) using injected examples when available
        base_module = JobAnalysisModule()
        input_field = get_input_field_from_module(base_module)
        max_examples = int(os.getenv("RR_MAX_EXAMPLES", "40"))
        base_examples = create_training_examples(max_examples=max_examples, data_type="exp1")

        if last_selections:
            block_text = build_selection_block(slot_universe, last_selections, header="Attributes:")
            train_examples = inject_block_into_examples(base_examples, input_field, block_text)
        else:
            train_examples = base_examples

        train_size = max(1, len(train_examples) // 2)
        trainset = train_examples[:train_size]
        valset = train_examples[train_size:train_size + 2] if len(train_examples) > train_size else train_examples[:1]

        optimizer = MIPROv2(metric=job_metrics_metric_mipro, auto=os.getenv("RR_DSPY_MODE", "light"), num_threads=2)
        optimized_module = optimizer.compile(
            student=base_module,
            trainset=trainset,
            valset=valset,
            requires_permission_to_run=False,
        )

        # Convert with marker so attributes render in-place
        categories = build_categories_from_df(ext_df) or KNOWLEDGE_CATEGORIES
        template = from_predict(
            optimized_module,
            slots=slot_universe,
            title=f"Job Knowledge (RR Round {r+1})",
            categories=categories,
            marker=marker,
            include_attributes_block=True,
            block_header="Attributes:",
        )
        template.configure(external_df=ext_df)

        # Warm-start P3O with last selections if available
        if last_selections:
            template.set_optimized_slots(last_selections)

        llm = JobKnowledgeMockLLM(low=0, high=100, seed=0)
        arch = Archetype(prompt=template, llm=llm, n_arch=3)
        arch.configure(external_df=ext_df)
        opt = P3O(archetype=arch, verbose=False)

        opt.train(mode=mode, log_interval=1, batch_size=batch_size)

        # Freeze selections for next round
        last_selections = opt.get_p3o_selections()
        print(f"Selections: {last_selections}")

    print("\nDone.")


if __name__ == "__main__":
    main()


