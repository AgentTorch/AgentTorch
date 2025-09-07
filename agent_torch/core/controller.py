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
        # Caches to reduce Python overhead in hot path
        self._obs_keys_cache = {}
        self._policy_keys_cache = {}

    def observe(self, state, observation_function, agent_type):
        substep = state["current_substep"]
        # Guard: substep or observation block may be None
        obs_block = self.config["substeps"].get(substep, {}).get("observation")
        if not isinstance(obs_block, dict):
            return None
        # Cache keys for this substep/agent_type
        if substep not in self._obs_keys_cache:
            self._obs_keys_cache[substep] = {}
        if agent_type not in self._obs_keys_cache[substep]:
            agent_map = obs_block.get(agent_type) or {}
            if not isinstance(agent_map, dict):
                self._obs_keys_cache[substep][agent_type] = []
            else:
                self._obs_keys_cache[substep][agent_type] = list(agent_map.keys())
        keys = self._obs_keys_cache[substep][agent_type]
        if not keys:
            return None
        result = {}
        funcs = observation_function[substep][agent_type]
        for obs_key in keys:
            result.update(funcs[obs_key](state))
        return result

    def act(self, state, observation, policy_function, agent_type):
        substep = state["current_substep"]
        pol_block = self.config["substeps"].get(substep, {}).get("policy")
        if not isinstance(pol_block, dict):
            return None
        if substep not in self._policy_keys_cache:
            self._policy_keys_cache[substep] = {}
        if agent_type not in self._policy_keys_cache[substep]:
            agent_map = pol_block.get(agent_type) or {}
            if not isinstance(agent_map, dict):
                self._policy_keys_cache[substep][agent_type] = []
            else:
                self._policy_keys_cache[substep][agent_type] = list(agent_map.keys())
        keys = self._policy_keys_cache[substep][agent_type]
        if not keys:
            return None
        result = {}
        funcs = policy_function[substep][agent_type]
        for pol_key in keys:
            result.update(funcs[pol_key](state, observation))
        return result

    def progress(self, state, action, transition_function):
        next_state = copy_module(state)
        del state

        substep = next_state["current_substep"]
        next_substep = (int(substep) + 1) % self.config["simulation_metadata"][
            "num_substeps_per_step"
        ]
        next_state["current_substep"] = str(next_substep)

        for trans_func in self.config["substeps"][substep]["transition"].keys():
            updated_vals = transition_function[substep][trans_func](
                state=next_state, action=action
            )
            for var_name, value in updated_vals.items():
                source_path = self.config["substeps"][substep]["transition"][
                    trans_func
                ]["input_variables"][var_name]
                set_by_path(next_state, re.split("/", source_path), value)

        return next_state

    def progress_inplace(self, state, action, transition_function):
        """In-place state progression to avoid full copy when safe."""
        substep = state["current_substep"]
        next_substep = (int(substep) + 1) % self.config["simulation_metadata"][
            "num_substeps_per_step"
        ]
        state["current_substep"] = str(next_substep)

        for trans_func in self.config["substeps"][substep]["transition"].keys():
            updated_vals = transition_function[substep][trans_func](
                state=state, action=action
            )
            for var_name, value in updated_vals.items():
                source_path = self.config["substeps"][substep]["transition"][
                    trans_func
                ]["input_variables"][var_name]
                set_by_path(state, re.split("/", source_path), value)

        return state

    def learn_after_episode(self, episode_traj, initializer, optimizer):
        optimizer.zero_grad()
        ret_episode_all = sum(
            [i[0]["agents"]["consumers"]["Q_exp"] for i in episode_traj["states"]]
        )
        ret_episode_0 = ret_episode_all[0]
        ret_episode = ret_episode_all.sum()
        self.returns.append(ret_episode)
        loss = -1e6 * ret_episode
        loss.backward()
        F_t_param = initializer.policy_function["0"]["consumers"][
            "purchase_product"
        ].learnable_args["F_t_params"]
        print(
            f"return is {ret_episode}, return for agent 0 is {ret_episode_0} and the F_t_param for agent 0 is {F_t_param[0]}"
        )
        optimizer.step()
