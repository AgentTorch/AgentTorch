import torch.nn as nn
from collections import deque

from agent_torch.controller import Controller
from agent_torch.initializer import Initializer
from agent_torch.helpers import to_cpu


class Runner(nn.Module):
    def __init__(self, config, registry) -> None:
        super().__init__()

        self.config = config
        self.registry = registry
        assert self.config["simulation_metadata"]["num_substeps_per_step"] == len(
            list(self.config["substeps"].keys())
        )

        self.initializer = Initializer(self.config, self.registry)
        self.controller = Controller(self.config)

        self.state = None

    def init(self):
        r"""
        initialize the state of the simulation
        """
        self.initializer.initialize()
        self.state = self.initializer.state

        self.state_trajectory = []
        self.state_trajectory.append(
            [to_cpu(self.state)]
        )  # move state to cpu and save in trajectory

    def reset(self):
        r"""
        reinitialize the simulator at the beginning of an episode
        """
        self.init()

    def reset_state_before_episode(self):
        r"""
        reinitialize the state trajectory of the simulator at the beginning of an episode
        """
        self.state_trajectory = []
        self.state_trajectory.append([to_cpu(self.state)])

    def step(self, num_steps=None):
        r"""
        Execute a single episode of the simulation
        """

        assert self.state is not None
        self.reset_state_before_episode()

        if not num_steps:
            num_steps = self.config["simulation_metadata"]["num_steps_per_episode"]

        for time_step in range(num_steps):
            self.state["current_step"] = time_step

            for substep in self.config["substeps"].keys():
                observation_profile, action_profile = {}, {}

                for agent_type in self.config["substeps"][substep]["active_agents"]:
                    assert substep == self.state["current_substep"]
                    assert time_step == self.state["current_step"]

                    observation_profile[agent_type] = self.controller.observe(
                        self.state, self.initializer.observation_function, agent_type
                    )
                    action_profile[agent_type] = self.controller.act(
                        self.state,
                        observation_profile[agent_type],
                        self.initializer.policy_function,
                        agent_type,
                    )

                next_state = self.controller.progress(
                    self.state, action_profile, self.initializer.transition_function
                )
                self.state = next_state

                self.state_trajectory[-1].append(
                    to_cpu(self.state)
                )  # move state in state trajectory to cpu

    def _set_parameters(self, params):
        print("_set_parameters CURRENTLY BREAKS GRADIENT! PLEASE DON'T USE")
        for param in params:
            mode, param_name = param.split("/")[0], param.split("/")[1]
            self.state[mode][param_name].data.copy_(params[param])
            self.state[mode][param_name].requires_grad = True

    def step_from_params(self, num_steps=None, params=None):
        r"""
        execute simulation episode with custom parameters
        """
        self._set_parameters(params)
        self.step(num_steps)

    def forward(self):
        r"""
        Run all episodes of a simulation as defined in config.
        """
        for episode in range(self.config["simulation_metadata"]["num_episodes"]):
            num_steps_per_episode = self.config["simulation_metadata"][
                "num_steps_per_episode"
            ]
            self.step(num_steps_per_episode)
