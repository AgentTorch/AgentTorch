import torch.nn as nn
from collections import deque

from agent_torch.core.controller import Controller
from agent_torch.core.initializer import Initializer
from agent_torch.core.helpers import to_cpu


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
        # self.reset_state_before_episode()

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

    def _set_parameters(self, params_dict):
        print(":: calling _set_parameters_")
        for param_name in params_dict:
            tensor_func = self._map_and_replace_tensor(param_name)
            param_value = params_dict[param_name]
            new_tensor = tensor_func(self, param_value)
            print("new_tensor: ", new_tensor)

    def _map_and_replace_tensor(self, input_string):
        # Split the input string into its components
        parts = input_string.split('.')
        
        # Extract the relevant parts
        function = parts[1]
        index = parts[2]
        sub_func = parts[3]
        arg_type = parts[4]
        var_name = parts[5]
        
        def getter_and_setter(runner, new_value=None):
            current = runner

            substep_type = getattr(runner.initializer, function)
            substep_function = getattr(substep_type[str(index)], sub_func)
            current_tensor = getattr(substep_function, 'calibrate_' + var_name)

            print("current tensor: ", current_tensor)
            
            if new_value is not None:
                assert new_value.requires_grad == current_tensor.requires_grad
                setvar_name = 'calibrate_' + var_name
                setattr(substep_function, setvar_name, new_value)
                current_tensor = getattr(substep_function, 'calibrate_' + var_name)
                return current_tensor
            else:
                return current_tensor

        return getter_and_setter

    def step_from_params(self, num_steps=None, params=None):
        r"""
        execute simulation episode with custom parameters
        """
        if params is None:
            print(" missing parameters!!! ")
            return
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
