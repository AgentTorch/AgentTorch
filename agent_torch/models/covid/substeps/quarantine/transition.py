import torch.nn as nn
import re

from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_by_path, logical_not


class UpdateQuarantineStatus(SubstepTransition):
    """Logic: exposed or infected agents can start quarantine"""

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = self.config["simulation_metadata"]["device"]
        self.num_agents = self.config["simulation_metadata"]["num_agents"]
        self.quarantine_days = self.config["simulation_metadata"]["quarantine_days"]
        self.num_steps = self.config["simulation_metadata"]["num_steps_per_episode"]

        self.SUSCEPTIBLE_VAR = self.config["simulation_metadata"]["SUSCEPTIBLE_VAR"]
        self.EXPOSED_VAR = self.config["simulation_metadata"]["EXPOSED_VAR"]
        self.INFECTED_VAR = self.config["simulation_metadata"]["INFECTED_VAR"]
        self.RECOVERED_VAR = self.config["simulation_metadata"]["RECOVERED_VAR"]

        self.INFINITY_TIME = self.config["simulation_metadata"]["INFINITY_TIME"]

        self.END_QUARANTINE_VAR = -1
        self.START_QUARANTINE_VAR = 1
        self.BREAK_QUARANTINE_VAR = -1

    def _end_quarantine(self, t, is_quarantined, quarantine_start_date):
        agents_quarantine_end_date = quarantine_start_date + self.quarantine_days
        agent_quarantine_ends = (t >= agents_quarantine_end_date).long()

        is_quarantined = (
            is_quarantined + agent_quarantine_ends * self.END_QUARANTINE_VAR
        )
        quarantine_start_date = (
            quarantine_start_date * (1 - agent_quarantine_ends)
            + (self.INFINITY_TIME) * agent_quarantine_ends
        )

        return is_quarantined, quarantine_start_date

    def _start_quarantine(
        self, t, is_quarantined, quarantine_start_date, agent_quarantine_start_action
    ):
        agents_quarantine_start = agent_quarantine_start_action.long()

        is_quarantined = (
            is_quarantined + agents_quarantine_start * self.START_QUARANTINE_VAR
        )
        quarantine_start_date = (
            quarantine_start_date * logical_not(agents_quarantine_start)
            + agents_quarantine_start * t
        )
        #         quarantine_start_date = quarantine_start_date*(1 - agents_quarantine_start) + (agents_quarantine_start)*t

        return is_quarantined, quarantine_start_date

    def _break_quarantine(
        self, t, is_quarantined, quarantine_start_date, agent_quarantine_break_action
    ):
        agents_quarantine_break = agent_quarantine_break_action.long()

        is_quarantined = (
            is_quarantined + agents_quarantine_break * self.BREAK_QUARANTINE_VAR
        )
        quarantine_start_date = (
            quarantine_start_date * logical_not(agents_quarantine_break)
            + agents_quarantine_break * self.INFINITY_TIME
        )
        #         quarantine_start_date = quarantine_start_date*(1 - agents_quarantine_break) + (self.INFINITY_TIME)*agents_quarantine_break

        return is_quarantined, quarantine_start_date

    def update_quarantine_status(
        self,
        t,
        is_quarantined,
        quarantine_start_date,
        agent_quarantine_start_action,
        agent_quarantine_break_action,
    ):
        is_quarantined, quarantine_start_date = self._end_quarantine(
            t, is_quarantined, quarantine_start_date
        )
        is_quarantined, quarantine_start_date = self._start_quarantine(
            t, is_quarantined, quarantine_start_date, agent_quarantine_start_action
        )
        is_quarantined, quarantine_start_date = self._break_quarantine(
            t, is_quarantined, quarantine_start_date, agent_quarantine_break_action
        )

        return is_quarantined, quarantine_start_date

    def forward(self, state, action):
        input_variables = self.input_variables
        t = state["current_step"]
        print("Substep: Quarantine")

        is_quarantined = get_by_path(
            state, re.split("/", input_variables["is_quarantined"])
        )
        quarantine_start_date = get_by_path(
            state, re.split("/", input_variables["quarantine_start_date"])
        )

        agent_quarantine_start_action = action["citizens"]["start_compliance_action"]
        agent_quarantine_break_action = action["citizens"]["break_compliance_action"]

        new_is_quarantined, new_quarantine_start_date = self.update_quarantine_status(
            t,
            is_quarantined,
            quarantine_start_date,
            agent_quarantine_start_action,
            agent_quarantine_break_action,
        )

        return {
            self.output_variables[0]: new_is_quarantined,
            self.output_variables[1]: new_quarantine_start_date,
        }
