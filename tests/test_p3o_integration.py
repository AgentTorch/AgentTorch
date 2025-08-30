#!/usr/bin/env python3
"""
Minimal P3O integration test:
- Builds a template with learnable Variables
- Broadcasts to astoria population (with soc_code)
- Runs a sample to populate behavior state (group keys/outputs and slot choices)
- Runs a P3O step and verifies learnable logits changed
"""

import agent_torch.core.llm.template as lm
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.mock_llm import MockLLM
import agent_torch.populations.astoria as astoria
import pandas as pd
import torch
from agent_torch.optim import P3O


class P3OTemplate(lm.Template):
    system_prompt = "You are evaluating willingness based on job profile and context."
    grouping_logic = "soc_code"

    # Fields
    soc_code = lm.Variable(desc="SOC job code", learnable=False)
    job_title = lm.Variable(desc="Job title", learnable=False)
    Abilities = lm.Variable(desc="Required abilities", learnable=True)
    WorkContext = lm.Variable(desc="Work context", learnable=True)

    def __prompt__(self):
        # Mark learnable placeholders explicitly to enable P3O formatting
        self.prompt_string = (
            "You are evaluating willingness to work during a crisis for:\n"
            "Job: {job_title} (SOC: {soc_code})\n"
            "Required Abilities: {Abilities, learnable=True}\n"
            "Work Context: {WorkContext, learnable=True}\n"
        )

    def __output__(self):
        return "Rate willingness from 0.0 to 1.0 (respond with number only):"


def main():
    # Load jobs table (923 rows)
    jobs_923 = pd.read_csv("tools/jobs_923.csv")

    # Build archetype
    llm = MockLLM()
    arch = Archetype(prompt=P3OTemplate(), llm=llm, n_arch=2)
    arch.configure(
        external_df=jobs_923,
        ground_truth_src="agent_torch/core/llm/data/ground_truth_willingness_all_soc.csv",
        match_on="soc_code",
        value_col="willingness",
    )

    # Single-shot (optional warmup)
    arch.sample(print_examples=0)

    # Broadcast and sample to populate behavior state and slot choices
    arch.broadcast(population=astoria)
    _ = arch.sample(print_examples=0)

    # Capture parameter snapshot
    params = list(arch.parameters())
    if not params:
        print("No learnable parameters found; ensure Variables are learnable=True")
        return
    before = [p.detach().clone() for p in params]

    # Optimize
    opt = P3O(arch.parameters(), archetype=arch, lr=0.1)
    opt.step()
    opt.zero_grad()

    # Verify at least one parameter changed
    after = [p.detach().clone() for p in params]
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    print("P3O parameter update detected:", changed)
    for i, (b, a) in enumerate(zip(before, after)):
        diff = (a - b).abs().max().item()
        print(f" param[{i}] max|Δ|= {diff:.6f}")


if __name__ == "__main__":
    main()


