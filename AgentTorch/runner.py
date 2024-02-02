from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
import argparse
import re

from abc import ABC, abstractmethod

from AgentTorch.controller import Controller
from AgentTorch.registry import Registry
from AgentTorch.initializer import Initializer

from AgentTorch.helpers import set_by_path

class Runner(nn.Module):
    def __init__(self, config, registry) -> None:
        super().__init__()

        self.config = config
        self.registry = registry
        assert self.config["simulation_metadata"]["num_substeps_per_step"] == len(list(self.config['substeps'].keys()))
        
        self.initializer = Initializer(self.config, self.registry)
        self.controller = Controller(self.config)

        self.state = None
                
        self.trajectory = { 'states': deque(), 'observations': deque(), 'actions': deque(),'rewards': deque() }

    def init(self):
        r"""
            initialize the state of the simulation
        """
        self.initializer.initialize()
        self.state = self.initializer.state

        self.state_trajectory = []
        self.state_trajectory.append([self.state])
        for traj_var in self.trajectory.keys():
            self.trajectory[traj_var].append(deque())


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
        self.state_trajectory.append([self.state])
        self.trajectory = { 'states': deque(), 'observations': deque(), 'actions': deque(),'rewards': deque() }
    
    def step(self, num_steps=None):
        r"""
            Execute a single episode of the simulation
        """

        assert self.state is not None
        self.reset_state_before_episode()
        for traj_var in self.trajectory.keys():
            self.trajectory[traj_var].append(deque())

        if not num_steps:
            num_steps = self.config["simulation_metadata"]["num_steps_per_episode"]
        
        for time_step in range(num_steps):
            self.state['current_step'] = time_step
            for traj_var in self.trajectory.keys():
                self.trajectory[traj_var][-1].append(deque())

            for substep in self.config['substeps'].keys():
                self.trajectory["states"][-1][-1].append(self.state)
                observation_profile, action_profile = {}, {}

                for agent_type in self.config['substeps'][substep]['active_agents']:
                    assert substep == self.state['current_substep']
                    assert time_step == self.state['current_step']
                    
                    observation_profile[agent_type] = self.controller.observe(self.state, self.initializer.observation_function, agent_type)
                    action_profile[agent_type] = self.controller.act(self.state, observation_profile[agent_type], self.initializer.policy_function, agent_type)
                                            
                self.trajectory["observations"][-1][-1].append(observation_profile)
                self.trajectory["actions"][-1][-1].append(action_profile)

                next_state = self.controller.progress(self.state, action_profile, self.initializer.transition_function)
                self.state = next_state

                self.state_trajectory[-1].append(self.state)

    def _set_parameters(self, params):
        for param in params:
            set_by_path(root=self.state, items=re.split('.', param), value=params[param])
        
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
        for episode in range(self.config['simulation_metadata']['num_episodes']):
            num_steps_per_episode = self.config["simulation_metadata"]["num_steps_per_episode"]
            self.step(num_steps_per_episode)
