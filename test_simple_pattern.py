#!/usr/bin/env python3
"""
Simple AgentTorch LLM test with Mock or Real backend.
Usage:
    python test_simple_pattern.py --mock    # Use mock LLM
    python test_simple_pattern.py           # Use real LLM
"""
import argparse
import torch
import os

from agent_torch.core.llm.template import Template
from agent_torch.core.llm.dataframe_archetype import DataFrameArchetype
from agent_torch.core.llm.dataframe_behavior import DataFrameBehavior
from agent_torch.core.dataloader import LoadPopulation
import agent_torch.populations.mock_test_18 as mock_test_18
from agent_torch.core.llm.claude_llm import ClaudeLocal



class MockLLMBackend:
    """Mock LLM returning a fixed score."""
    def __init__(self):
        self.fixed_value = 0.7
        self.call_count = 0
        self.total_prompts = 0
        self.backend = "claude"
    
    def initialize_llm(self):
        return self

    def prompt(self, prompt_list):
        self.call_count += 1
        self.total_prompts += len(prompt_list)
        return [{"text": str(self.fixed_value)} for _ in prompt_list]


def create_template():
    """Build the COVID willingness Template."""
    return Template(
        src="agent_torch/populations/mock_test_18/job_data_clean.pkl",
        ground_truth_src="agent_torch/core/llm/data/ground_truth_willingness.csv",
        archetype_data="I am in my {age, learnable=True}'s. I am a {gender, learnable=False}.",
        external_data=(
            "As a {job_title, learnable=False}, your daily work involves "
            "{primary_tasks, learnable=True} and requires {abilities, learnable=True} in {work_context, learnable=True}."
        ),
        config_data="Factor in COVID-19 cases at {covid_cases, learnable=False} and {unemployment_rate, learnable=False} unemployment.",
        output_format={"type": "float", "range": [0.0, 1.0]},
        output_instruction="Rate your willingness to continue normal activities (0.0-1.0):",
        grouping_logic="job_title"
    )


def setup_behavior(use_mock: bool):

    template = create_template()
    population = LoadPopulation(mock_test_18)

    if use_mock or ClaudeLocal is None:
        llm = MockLLMBackend()
    else:
        llm = ClaudeLocal(
            model_name="claude-3-haiku-20240307",
            system_prompt="You are analyzing work willingness during challenging times.",
            temperature=0.7,
            max_tokens=200
        )

    arche = DataFrameArchetype(template=template, n_arch=3)
    llm_archetypes = arche.llm(llm=llm, user_prompt=template.output_instruction)

    behavior = DataFrameBehavior(
        archetype=llm_archetypes,
        region=mock_test_18,
        template=template,
        population=population,
        optimization_interval=5
    )
    return behavior, template, llm, population


def run_test(use_mock: bool):
    behavior, template, llm, population = setup_behavior(use_mock=use_mock)
    kwargs = {
        "month": "January",
        "year": "2020",
        "covid_cases": 1200,
        "unemployment_rate": 0.05,
        "population_size": population.population_size,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "current_memory_dir": os.getcwd()
    }

    for step in range(10):
        output = behavior.sample(kwargs)
        print("sampled_behavior shape:", tuple(output.shape))
        print("unique values:", torch.unique(output).tolist()[:10])
        print("first 10:", output[:10].flatten().tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AgentTorch LLM test.")
    parser.add_argument("--mock", action="store_true",
                        help="Use Mock LLM instead of real ClaudeLocal.")
    args = parser.parse_args()

    run_test(use_mock=args.mock)
