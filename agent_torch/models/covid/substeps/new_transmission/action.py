import torch
import numpy as np
import re
import torch.nn.functional as F

from agent_torch.core.helpers import get_by_path
from agent_torch.core.substep import SubstepAction
from agent_torch.core.llm.backend import LangchainLLM
from agent_torch.core.distributions import StraightThroughBernoulli

from ...calibration.utils.data import get_data, get_labels
from ...calibration.utils.feature import Feature
from ...calibration.utils.llm import AgeGroup, SYSTEM_PROMPT, construct_user_prompt
from ...calibration.utils.misc import week_num_to_epiweek, name_to_neighborhood

from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.handlers.trajectory import LogTrajectory
from chirho.dynamical.ops import simulate


class MakeIsolationDecision(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.mode = self.config["simulation_metadata"]["EXECUTION_MODE"]
        self.num_agents = self.config["simulation_metadata"]["num_agents"]

        self.st_bernoulli = StraightThroughBernoulli.apply

        self.calibration_mode = self.config['simulation_metadata']['calibration']

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

    def my_custom_ode(self, eval_time):

        # A vector of length 3.
        if self.calibration_mode:
            c, k, lam = self.calibrate_odeparams.to(self.device)
        else:
            c, k, lam = self.learnable_args["odeparams"]

        def gamma_like_ode(state):
            y = state['y']
            t = state['t']
            dydt = c * (t ** (k - 1)) * torch.exp(-lam * t) - y * t
            return dict(y=dydt)

        with TorchDiffEq(atol=1e-6, rtol=1e-6):
            end_state = simulate(gamma_like_ode, dict(y=torch.tensor(0.)), torch.tensor(0.), eval_time)

        return end_state['y']

    def forward(self, state, observation):
        # if in debug mode, return random values for isolation
        will_isolate = torch.rand(self.num_agents, 1).to(self.device)

        eval_time = state["current_step"]

        daily_transmission = self.my_custom_ode(eval_time)
        # breakpoint()

        return {self.output_variables[0]: will_isolate,
                self.output_variables[1]: daily_transmission}
