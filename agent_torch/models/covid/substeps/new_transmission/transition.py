import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import re

from agent_torch.core.substep import SubstepTransitionMessagePassing
from agent_torch.core.helpers import get_by_path
from agent_torch.core.distributions import StraightThroughBernoulli


class NewTransmission(SubstepTransitionMessagePassing):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.SUSCEPTIBLE_VAR = self.config["simulation_metadata"]["SUSCEPTIBLE_VAR"]
        self.EXPOSED_VAR = self.config["simulation_metadata"]["EXPOSED_VAR"]
        self.RECOVERED_VAR = self.config["simulation_metadata"]["RECOVERED_VAR"]

        self.num_timesteps = self.config["simulation_metadata"]["num_steps_per_episode"]
        self.num_weeks = self.config["simulation_metadata"]["NUM_WEEKS"]

        self.STAGE_UPDATE_VAR = 1
        self.INFINITY_TIME = self.config["simulation_metadata"]["INFINITY_TIME"]
        self.EXPOSED_TO_INFECTED_TIME = self.config["simulation_metadata"][
            "EXPOSED_TO_INFECTED_TIME"
        ]

        self.mode = self.config["simulation_metadata"]["EXECUTION_MODE"]
        self.st_bernoulli = StraightThroughBernoulli.apply

        self.calibration_mode = self.config["simulation_metadata"]["calibration"]

        # Precompute one-hot matrices and buffers for reuse
        self._time_one_hot = torch.eye(self.num_timesteps, device=self.device)
        self._week_one_hot = torch.eye(self.num_weeks, device=self.device)
        self._buffer_cache = {}

    def _get_buffer(self, name, like_tensor):
        key = (name, tuple(like_tensor.shape), like_tensor.dtype, like_tensor.device)
        buf = self._buffer_cache.get(key)
        if (
            buf is None
            or buf.shape != like_tensor.shape
            or buf.dtype != like_tensor.dtype
            or buf.device != like_tensor.device
        ):
            buf = torch.empty_like(like_tensor)
            self._buffer_cache[key] = buf
        return buf

    def _lam(
        self,
        x_i,
        x_j,
        edge_attr,
        t,
        R,
        SFSusceptibility,
        SFInfector,
        lam_gamma_integrals,
    ):
        S_A_s = SFSusceptibility[x_i[:, 0].long()]
        A_s_i = SFInfector[x_j[:, 1].long()]
        B_n = edge_attr[1, :]
        integrals = torch.zeros_like(B_n)
        infected_idx = x_j[:, 2].bool()
        infected_times = t - x_j[infected_idx, 3] - 1

        integrals[infected_idx] = lam_gamma_integrals[infected_times.long()]
        edge_network_numbers = edge_attr[0, :]

        I_bar = torch.gather(x_i[:, 4], 0, edge_network_numbers.long()).view(-1)

        will_isolate = x_i[:, 6]  # is the susceptible agent isolating? check x_i vs x_j
        not_isolated = 1 - will_isolate

        if self.mode == "llm":
            res = (
                R * S_A_s * A_s_i * B_n * integrals / I_bar
            )  # not_isolated*R*S_A_s*A_s_i*B_n*integrals/I_bar * 1/2
        else:
            res = R * S_A_s * A_s_i * B_n * integrals / I_bar

        return res.view(-1, 1)

    def message(
        self,
        x_i,
        x_j,
        edge_attr,
        t,
        R,
        SFSusceptibility,
        SFInfector,
        lam_gamma_integrals,
    ):
        return self._lam(
            x_i, x_j, edge_attr, t, R, SFSusceptibility, SFInfector, lam_gamma_integrals
        )

    def update_stages(self, current_stages, newly_exposed_today):
        new_stages = current_stages + newly_exposed_today * self.STAGE_UPDATE_VAR
        return new_stages

    def update_transition_times(self, t, current_transition_times, newly_exposed_today):
        out = self._get_buffer("next_stage_time", current_transition_times)
        out.copy_(current_transition_times)
        mask = newly_exposed_today.bool().squeeze()
        if mask.numel() > 0:
            out[mask] = t + 1 + self.EXPOSED_TO_INFECTED_TIME
        return out

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        if num_timesteps == self.num_timesteps:
            return self._time_one_hot[timestep]
        if num_timesteps == self.num_weeks:
            return self._week_one_hot[timestep]
        timestep_tensor = torch.tensor([timestep], device=self.device)
        one_hot_tensor = F.one_hot(timestep_tensor, num_classes=num_timesteps)
        return one_hot_tensor.to(self.device)

    def update_infected_times(self, t, agents_infected_times, newly_exposed_today):
        out = self._get_buffer("infected_time", agents_infected_times)
        out.copy_(agents_infected_times)
        mask = newly_exposed_today.bool().squeeze()
        if mask.numel() > 0:
            out[mask] = t
        return out

    def forward(self, state, action=None):
        input_variables = self.input_variables
        t = int(state["current_step"])
        time_step_one_hot = self._generate_one_hot_tensor(t, self.num_timesteps)

        week_id = int(t / 7)
        week_one_hot = self._generate_one_hot_tensor(week_id, self.num_weeks)

        if self.calibration_mode:
            R_tensor = self.calibrate_R2
        else:
            R_tensor = self.learnable_args["R2"]  # tensor of size NUM_WEEK
        R = (R_tensor * week_one_hot).sum()

        SFSusceptibility = get_by_path(
            state, re.split("/", input_variables["SFSusceptibility"])
        )
        SFInfector = get_by_path(state, re.split("/", input_variables["SFInfector"]))
        all_lam_gamma = get_by_path(
            state, re.split("/", input_variables["lam_gamma_integrals"]) 
        )

        agents_infected_time = get_by_path(
            state, re.split("/", input_variables["infected_time"]) 
        )
        agents_mean_interactions_split = get_by_path(
            state, re.split("/", input_variables["mean_interactions"]) 
        )
        agents_ages = get_by_path(state, re.split("/", input_variables["age"]))
        current_stages = get_by_path(
            state, re.split("/", input_variables["disease_stage"]) 
        )
        current_transition_times = get_by_path(
            state, re.split("/", input_variables["next_stage_time"]) 
        )

        all_edgelist, all_edgeattr = get_by_path(
            state, re.split("/", input_variables["adjacency_matrix"]) 
        )

        daily_infected = get_by_path(
            state, re.split("/", input_variables["daily_infected"]) 
        )

        agents_infected_index = torch.logical_and(
            current_stages > self.SUSCEPTIBLE_VAR, current_stages < self.RECOVERED_VAR
        )

        will_isolate = action["citizens"]["isolation_decision"]

        all_node_attr = (
            torch.stack(
                (
                    agents_ages,  # 0
                    current_stages.detach(),  # 1
                    agents_infected_index,  # 2
                    agents_infected_time,  # 3
                    agents_mean_interactions_split,  # 4 *agents_mean_interactions_split,
                    torch.unsqueeze(
                        torch.arange(self.config["simulation_metadata"]["num_agents"]),
                        1,
                    ).to(
                        self.device
                    ),  # 5
                    will_isolate,
                )
            )
            .transpose(0, 1)
            .squeeze()
        )  # .t() # 6

        num_nodes = all_node_attr.size(0) if hasattr(all_node_attr, 'size') else None
        agents_data = Data(
            all_node_attr, edge_index=all_edgelist, edge_attr=all_edgeattr, t=t
        )
        if num_nodes is not None:
            agents_data.num_nodes = int(num_nodes)

        new_transmission = self.propagate(
            agents_data.edge_index,
            x=agents_data.x,
            edge_attr=agents_data.edge_attr,
            t=agents_data.t,
            R=R,
            SFSusceptibility=SFSusceptibility,
            SFInfector=SFInfector,
            lam_gamma_integrals=all_lam_gamma.squeeze(),
        )

        prob_not_infected = torch.exp(-1 * new_transmission)
        probs = torch.hstack((1 - prob_not_infected, prob_not_infected))

        potentially_exposed_today = self.st_bernoulli(probs)[:, 0].to(
            self.device
        )  # using straight-through bernoulli
        potentially_exposed_today = potentially_exposed_today * (
            1.0 - will_isolate.squeeze()
        )

        newly_exposed_today = (
            current_stages == self.SUSCEPTIBLE_VAR
        ).squeeze() * potentially_exposed_today

        # In-place update to avoid tensor reallocation
        daily_infected.add_(newly_exposed_today.sum() * time_step_one_hot)

        newly_exposed_today = newly_exposed_today.unsqueeze(1)

        updated_stages = self.update_stages(current_stages, newly_exposed_today)
        updated_next_stage_times = self.update_transition_times(
            t, current_transition_times, newly_exposed_today
        )
        updated_infected_times = self.update_infected_times(
            t, agents_infected_time, newly_exposed_today
        )

        return {
            self.output_variables[0]: updated_stages,
            self.output_variables[1]: updated_next_stage_times,
            self.output_variables[2]: updated_infected_times,
            self.output_variables[3]: daily_infected,
        }
