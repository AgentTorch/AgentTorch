import asyncio
import torch.nn as nn
import re
from agent_torch.core.helpers import get_by_path, set_by_path, copy_module
from agent_torch.core.utils import is_async_method


class Controller(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.returns = []

    def observe(self, state, observation_function, agent_type):
        observation = {}
        substep = state["current_substep"]
        try:
            for obs in self.config["substeps"][substep]["observation"][
                agent_type
            ].keys():
                observation = {
                    **observation_function[substep][agent_type][obs](state),
                    **observation,
                }
        except Exception as e:
            observation = None

        return observation

    def act(self, state, observation, policy_function, agent_type):
        action = {}
        substep, step = state["current_substep"], state["current_step"]

        try:
            for policy in self.config["substeps"][substep]["policy"][agent_type].keys():
                action = {
                    **policy_function[substep][agent_type][policy](state, observation),
                    **action,
                }
        except Exception as e:
            action = None

        return action

    def progress(self, state, action, transition_function):
        next_state = copy_module(state)
        del state

        substep = next_state["current_substep"]
        next_substep = (int(substep) + 1) % self.config["simulation_metadata"][
            "num_substeps_per_step"
        ]
        next_state["current_substep"] = str(next_substep)

        for trans_func in self.config["substeps"][substep]["transition"].keys():
            updated_vals = {
                **transition_function[substep][trans_func](
                    state=next_state, action=action
                )
            }
            for var_name in updated_vals:
                assert self.config["substeps"][substep]["transition"][trans_func][
                    "input_variables"
                ][var_name]

                source_path = self.config["substeps"][substep]["transition"][
                    trans_func
                ]["input_variables"][var_name]
                set_by_path(
                    next_state, re.split("/", source_path), updated_vals[var_name]
                )

        return next_state

