"""
This file demonstrates the new unified API we are refactoring towards.

- Users define a class-based Template with Variables and hooks
- Users construct an Archetype(prompt=..., llm=..., n_arch=...)
- sample() runs a single-shot; broadcast(...) binds a population and sample() returns (n_agents,) decisions
- Optimizer consumes arch.parameters() to update learnable Variables
"""

import agent_torch.core.llm.template as lm
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.mock_llm import MockLLM
import agent_torch.populations.astoria as astoria
import torch
import torch.nn as nn
import pandas as pd
import os


class MyPromptTemplate(lm.Template):
    system_prompt = "You are evaluating willingness based on job profile and context."

    # Use existing dataset fields
    age = lm.Variable(desc="agent age", learnable=True)
    gender = lm.Variable(desc="agent gender", learnable=False)
    soc_code = lm.Variable(desc="job id", learnable=False)
    abilities = lm.Variable(desc="abilities required", learnable=True)
    work_context = lm.Variable(desc="work context", learnable=True)

    def __prompt__(self):
        self.prompt_string = (
            "You are in your {age}'s and are a {gender}. "
            "As a {soc_code}...you have {abilities} and work in {work_context}."
        )

    def __output__(self):
        return "Rate your willingness to continue normal activities, respond in [0, 1] binary decision only."

    def __data__(self):
        # Point to data source
        base_dir = os.path.dirname(__file__)
        self.src = os.path.join(base_dir, "job_data_clean.pkl")


def fn(api_key=None):

    
    prompt_template = MyPromptTemplate()
    # Ground truth and grouping configured via archetype.configure()
    
    llm = MockLLM()  # or lm.ClaudeHaiku(api_key)
    arch = Archetype(prompt=prompt_template, llm=llm, n_arch=3)

    base_dir = os.path.dirname(__file__)
    all_jobs_df = pd.read_pickle(os.path.join(base_dir, "job_data_clean.pkl"))

    gt_csv_path = os.path.join(base_dir, "agent_torch", "core", "llm", "test_data", "test_data.csv")
    gt_csv = pd.read_csv(gt_csv_path, encoding="utf-8-sig")
    soc_to_val = {r['soc_code']: float(r['willingness']) for _, r in gt_csv.iterrows() if 'soc_code' in r and 'willingness' in r}
    ground_truth_list = [soc_to_val.get(str(row.get('soc_code')), 0.0) for _, row in all_jobs_df.iterrows()]

    # For quick tests, you can limit rows via split for pre-broadcast preview
    arch.configure(external_df=all_jobs_df, split=2)

    arch.sample()  # runs the prompt for base version

    # Use full dataset for broadcast to ensure all soc_codes match
    arch.configure(external_df=all_jobs_df)

    arch.broadcast(population=astoria, match_on="soc_code")

    arch.sample()  # returns (n_agents,) tensor of decisions


    from agent_torch.optim import P3O
    
    params = list(arch.parameters())
    print(f"\nLearnable parameters: {len(params)}")
    
    # Configure archetype with full jobs table and ground truth via Archetype.configure


    # Create P3O with archetype; auto-updates from archetype in step()
    opt = P3O(arch.parameters(), archetype=arch)
    
    for i in range(2):
        # Run population sample; behavior stores group outputs/keys
        arch.sample(print_examples=1)
        # Apply parameter step (auto-pulls group info from archetype)
        opt.step()
        opt.zero_grad()

def simple_fn():
    prompt_template = "You are evaluating willingness based on job profile and context."
    # Configure with ground truth data for optimization    
    llm = MockLLM()  # or lm.ClaudeHaiku(api_key)
    arch = Archetype(prompt=prompt_template, llm=llm, n_arch=3)

    arch.sample(print_examples=1)  # runs the prompt for base version

    arch.broadcast(population=astoria)
    arch.sample(print_examples=1)  # returns (n_agents,) tensor of decisions


if __name__ == "__main__":
    fn()