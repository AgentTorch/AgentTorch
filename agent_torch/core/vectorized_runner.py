"""
Vectorized runner for AgentTorch that efficiently processes batched operations.
"""

import torch
import re
from torch.func import vmap
from agent_torch.core.runner import Runner
from agent_torch.core.helpers.general import (
    get_by_path,
    set_by_path,
    copy_module,
    to_cpu,
)
from agent_torch.core.vectorization import is_vectorized


class VectorizedRunner(Runner):
    """
    A runner that leverages vectorized operations for improved performance.

    This runner enhances the standard Runner by detecting and utilizing vectorized
    implementations of observation, policy, and transition functions. For components
    that don't have vectorized implementations, it falls back to the standard behavior.
    """

    def __init__(self, config, registry) -> None:
        super().__init__(config, registry)
        # Flag to indicate this is a vectorized runner
        self.is_vectorized = True

        # Add metadata about which functions are vectorized
        self._update_vectorized_functions()

    def _update_vectorized_functions(self):
        """Scan all registered functions and track which ones are vectorized."""
        self.vectorized_functions = {"observation": {}, "policy": {}, "transition": {}}

        # Check for vectorized observation functions
        for substep, agent_types in self.initializer.observation_function.items():
            self.vectorized_functions["observation"][substep] = {}
            for agent_type, obs_funcs in agent_types.items():
                self.vectorized_functions["observation"][substep][agent_type] = {}
                for func_name, func in obs_funcs.items():
                    self.vectorized_functions["observation"][substep][agent_type][
                        func_name
                    ] = is_vectorized(func)

        # Check for vectorized policy functions
        for substep, agent_types in self.initializer.policy_function.items():
            self.vectorized_functions["policy"][substep] = {}
            for agent_type, pol_funcs in agent_types.items():
                self.vectorized_functions["policy"][substep][agent_type] = {}
                for func_name, func in pol_funcs.items():
                    self.vectorized_functions["policy"][substep][agent_type][
                        func_name
                    ] = is_vectorized(func)

        # Check for vectorized transition functions
        for substep, trans_funcs in self.initializer.transition_function.items():
            self.vectorized_functions["transition"][substep] = {}
            for func_name, func in trans_funcs.items():
                self.vectorized_functions["transition"][substep][func_name] = (
                    is_vectorized(func)
                )

    def observe(self, state, observation_function, agent_type):
        """
        Override to use vectorized observation functions when available.
        """
        observation = {}
        substep = state["current_substep"]
        try:
            for obs in self.config["substeps"][substep]["observation"][
                agent_type
            ].keys():
                obs_func = observation_function[substep][agent_type][obs]

                # Check if the observation function is marked as vectorized
                if is_vectorized(obs_func):
                    # Use vectorized implementation
                    observation = {**obs_func(state), **observation}
                else:
                    # Fall back to original sequential processing
                    observation = {
                        **observation_function[substep][agent_type][obs](state),
                        **observation,
                    }
        except Exception as e:
            observation = None

        return observation

    def act(self, state, observation, policy_function, agent_type):
        """
        Override to use vectorized policy functions when available.
        """
        action = {}
        substep, step = state["current_substep"], state["current_step"]

        try:
            for policy in self.config["substeps"][substep]["policy"][agent_type].keys():
                policy_func = policy_function[substep][agent_type][policy]

                # Check if the policy function is marked as vectorized
                if is_vectorized(policy_func):
                    # Use vectorized implementation
                    action = {**policy_func(state, observation), **action}
                else:
                    # Fall back to original sequential processing
                    action = {
                        **policy_function[substep][agent_type][policy](
                            state, observation
                        ),
                        **action,
                    }
        except Exception as e:
            action = None

        return action

    def progress(self, state, action, transition_function):
        """
        Override to use vectorized transition functions when available.
        """
        next_state = copy_module(state)
        del state

        substep = next_state["current_substep"]
        next_substep = (int(substep) + 1) % self.config["simulation_metadata"][
            "num_substeps_per_step"
        ]
        next_state["current_substep"] = str(next_substep)

        for trans_func_name in self.config["substeps"][substep]["transition"].keys():
            trans_func = transition_function[substep][trans_func_name]

            # Check if the transition function is marked as vectorized
            if is_vectorized(trans_func):
                # Use vectorized implementation
                updated_vals = {**trans_func(state=next_state, action=action)}
            else:
                # Fall back to original sequential processing
                updated_vals = {
                    **transition_function[substep][trans_func_name](
                        state=next_state, action=action
                    )
                }

            for var_name in updated_vals:
                if (
                    var_name
                    in self.config["substeps"][substep]["transition"][trans_func_name][
                        "output_variables"
                    ]
                ):
                    source_path = self.config["substeps"][substep]["transition"][
                        trans_func_name
                    ]["input_variables"][var_name]
                    set_by_path(
                        next_state, re.split("/", source_path), updated_vals[var_name]
                    )

        return next_state

    def get_vectorized_stats(self):
        """
        Get statistics about vectorized functions in the model.

        Returns:
            dict: Dictionary with statistics about vectorized functions
        """
        stats = {
            "observation": {"vectorized": 0, "total": 0},
            "policy": {"vectorized": 0, "total": 0},
            "transition": {"vectorized": 0, "total": 0},
        }

        # Count vectorized observation functions
        for substep, agent_types in self.vectorized_functions["observation"].items():
            for agent_type, obs_funcs in agent_types.items():
                for func_name, is_vec in obs_funcs.items():
                    stats["observation"]["total"] += 1
                    if is_vec:
                        stats["observation"]["vectorized"] += 1

        # Count vectorized policy functions
        for substep, agent_types in self.vectorized_functions["policy"].items():
            for agent_type, pol_funcs in agent_types.items():
                for func_name, is_vec in pol_funcs.items():
                    stats["policy"]["total"] += 1
                    if is_vec:
                        stats["policy"]["vectorized"] += 1

        # Count vectorized transition functions
        for substep, trans_funcs in self.vectorized_functions["transition"].items():
            for func_name, is_vec in trans_funcs.items():
                stats["transition"]["total"] += 1
                if is_vec:
                    stats["transition"]["vectorized"] += 1

        # Calculate total
        total_vectorized = sum(s["vectorized"] for s in stats.values())
        total_functions = sum(s["total"] for s in stats.values())
        stats["total"] = {
            "vectorized": total_vectorized,
            "total": total_functions,
            "percentage": (
                100 * total_vectorized / total_functions if total_functions > 0 else 0
            ),
        }

        return stats
