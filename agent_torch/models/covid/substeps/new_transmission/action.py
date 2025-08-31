import torch
import numpy as np
import re
import torch.nn.functional as F

from agent_torch.core.helpers import get_by_path
from agent_torch.core.substep import SubstepAction
from agent_torch.core.llm.backend import LangchainLLM
from agent_torch.core.distributions import StraightThroughBernoulli

from agent_torch.core.decorators import with_behavior

from ...calibration.utils.data import get_data, get_labels
from ...calibration.utils.feature import Feature
from ...calibration.utils.llm import AgeGroup, SYSTEM_PROMPT, construct_user_prompt
from ...calibration.utils.misc import week_num_to_epiweek, name_to_neighborhood


@with_behavior
class MakeIsolationDecision(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.mode = self.config["simulation_metadata"]["EXECUTION_MODE"]
        self.num_agents = self.config["simulation_metadata"]["num_agents"]

        self.st_bernoulli = StraightThroughBernoulli.apply

    def string_to_number(self, string):
        if string.lower() == "yes":
            return 1
        else:
            return 0

    def change_text(self, change_amount):
        change_amount = int(change_amount)
        if change_amount >= 1:
            return f"a {change_amount}% increase from last week"
        elif change_amount <= -1:
            return f"a {abs(change_amount)}% decrease from last week"
        else:
            return "the same as last week"

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        timestep_tensor = torch.tensor([timestep])
        one_hot_tensor = F.one_hot(timestep_tensor, num_classes=num_timesteps)

        return one_hot_tensor.to(self.device)

    def forward(self, state, observation):
        # if in heuristic mode, return random values for isolation decision
        if self.mode == "heuristic":
            will_isolate = torch.rand(self.num_agents, 1).to(self.device)
        else:
            if self.behavior is None:
                will_isolate = torch.rand(self.num_agents, 1).to(self.device)
            else:
                will_isolate = self.behavior(observation)

        # Safe handling of output_variables to prevent None access
        if hasattr(self, 'output_variables') and self.output_variables:
            output_key = self.output_variables[0]
        else:
            output_key = "isolation_decision"
            
        return {output_key: will_isolate}