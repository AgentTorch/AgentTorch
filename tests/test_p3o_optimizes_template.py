import pandas as pd
import torch

import agent_torch.core.llm.template as lm
from agent_torch.core.llm.archetype import Archetype
from agent_torch.optim.p3o import P3O


class TestLLM:
    """LLM stub that returns values based on how 'abilities' is formatted in the prompt.

    Mapping:
      - contains 'The abilities is' -> 0.95 (best)
      - contains 'abilities:'       -> 0.90
      - contains 'with '            -> 0.80
      - contains 'abilities' (direct value only) -> 0.70
      - otherwise (skip)            -> 0.20
    """

    def initialize_llm(self):
        return self

    def prompt(self, prompt_list):
        outs = []
        for p in prompt_list:
            text = p if isinstance(p, str) else p.get("agent_query", "")
            val = 0.2
            if "The abilities is" in text:
                val = 0.95
            elif "abilities:" in text:
                val = 0.90
            elif " with " in text or "with " in text:
                val = 0.80
            elif "abilities" in text:
                val = 0.70
            outs.append({"text": str(val)})
        return outs


class DummyPopulation:
    def __init__(self, n: int, soc_code_value: str):
        self.population_size = n
        self.soc_code = [soc_code_value for _ in range(n)]


class LearnableAbilitiesTemplate(lm.Template):
    system_prompt = "You are evaluating willingness based on job profile and context."

    # Only include external-driven fields in the prompt here
    soc_code = lm.Variable(desc="job id", learnable=False)
    abilities = lm.Variable(desc="abilities required", learnable=True)

    def __prompt__(self):
        # Mark abilities as learnable in the prompt so P3O presentation choices apply
        self.prompt_string = (
            "Job: {soc_code}. Abilities: {abilities, learnable=True}."
        )

    def __output__(self):
        return "Rate willingness from 0.0 to 1.0 (respond with number only):"


def test_p3o_optimizes_learnable_variable():
    # External data with one job row
    df = pd.DataFrame({
        "soc_code": ["11-1011.00"],
        "abilities": ["Leadership, Strategy"],
        "work_context": ["Executive Office"],
    })

    # Ground-truth targets aligned to external rows; set to 1.0 to favor higher LLM predictions
    gt_list = [1.0 for _ in range(len(df))]

    # Archetype with learnable abilities
    template = LearnableAbilitiesTemplate()
    llm = TestLLM()
    arch = Archetype(prompt=template, llm=llm, n_arch=2)

    arch.configure(
        external_df=df,
        ground_truth=gt_list,
        match_on="soc_code",
        # grouping defaults to match_on
    )

    # Bind a tiny population with the same soc_code (one group)
    pop = DummyPopulation(n=5, soc_code_value="11-1011.00")
    arch.broadcast(population=pop)

    # Capture initial probabilities for abilities presentation
    var = template._variables.get("abilities")
    assert var is not None
    init_probs = var.get_probabilities(template).detach().clone()

    # Seed for reproducibility
    torch.manual_seed(0)

    # Optimize with P3O for a number of steps
    opt = P3O(arch.parameters(), archetype=arch, lr=0.1)
    for _ in range(30):
        arch.sample()   # populates last_group_* and last_slot_choices
        opt.step()
        opt.zero_grad()

    final_probs = var.get_probabilities(template).detach()

    # Expect the best formatting to move away from direct/skip toward richer formats (index 3 or 4)
    assert final_probs.numel() == init_probs.numel()
    final_top = int(torch.argmax(final_probs).item())
    assert final_top in (3, 4)
    # Probability of chosen top format should increase vs initial
    assert float(final_probs[final_top]) > float(init_probs[final_top])
    # Direct format (1) should not become more likely
    assert float(final_probs[1]) <= float(init_probs[1])


