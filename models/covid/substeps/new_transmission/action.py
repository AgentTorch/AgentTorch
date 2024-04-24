# AGENT_TORCH_PATH = '/u/ayushc/projects/GradABM/MacroEcon/AgentTorch'
# MODEL_PATH = '/u/ayushc/projects/GradABM/MacroEcon/models'

import torch
import numpy as np
import re
import torch.nn.functional as F

# sys.path.append(MODEL_PATH)
# sys.path.insert(0, AGENT_TORCH_PATH)

from AgentTorch.helpers import get_by_path
from AgentTorch.substep import SubstepAction
from AgentTorch.LLM.llm_agent import LLMAgent

from utils.data import get_data, get_labels
from utils.feature import Feature
from utils.llm import AgeGroup, SYSTEM_PROMPT, construct_user_prompt
from utils.misc import week_num_to_epiweek, name_to_neighborhood
from AgentTorch.helpers.distributions import StraightThroughBernoulli


class MakeIsolationDecision(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set values from config
        OPENAI_API_KEY = self.config["simulation_metadata"]["OPENAI_API_KEY"]
        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.mode = self.config["simulation_metadata"]["EXECUTION_MODE"]
        self.num_agents = self.config["simulation_metadata"]["num_agents"]
        self.align_llm = self.config['simulation_metadata']['ALIGN_LLM']
        self.epiweek_start = week_num_to_epiweek(
            self.config["simulation_metadata"]["START_WEEK"]
        )
        self.num_weeks = self.config["simulation_metadata"]["NUM_WEEKS"]
        self.neighborhood = name_to_neighborhood(
            self.config["simulation_metadata"]["NEIGHBORHOOD"]
        )
        self.include_week_count = self.config["simulation_metadata"][
            "INCLUDE_WEEK_COUNT"
        ]
        self.use_ground_truth_case_numbers = self.config["simulation_metadata"][
            "USE_GROUND_TRUTH_CASE_NUMBERS"
        ]
        self.use_ground_truth_4_week_avg = self.config["simulation_metadata"][
            "USE_GROUND_TRUTH_4WK_AVG"
        ]

        # set up llm agent
        self.agent = LLMAgent(agent_profile=SYSTEM_PROMPT, openai_api_key=OPENAI_API_KEY)

        # retrieve case numbers
        if self.use_ground_truth_case_numbers:
            # get the full range of case numbers
            self.cases_week = get_labels(
                self.neighborhood, self.epiweek_start - 1, self.num_weeks, Feature.CASES
            )
        else:
            # get only the starting case numbers for the prompt
            self.cases_week = list(
                get_labels(
                    self.neighborhood, self.epiweek_start - 1, 1, Feature.CASES
                ).to(self.device)
            )

        # retrieve 4 week case averages
        if self.use_ground_truth_4_week_avg:
            # get the full range of 4 week averages
            self.cases_4_week_avg = get_labels(
                self.neighborhood,
                self.epiweek_start - 1,
                self.num_weeks,
                Feature.CASES_4WK_AVG,
            )
        else:
            # this is a bug. we are using the ground truth data for the first 3 weeks of our
            # simulation still, and then moving on to the case numbers.
            self.cases_4_week_avg = list(
                get_labels(
                    self.neighborhood,
                    self.epiweek_start - 1,
                    3,
                    Feature.CASES_4WK_AVG,
                )
            )

        self.external_align_vector = torch.tensor(self.learnable_args['align_vector'], requires_grad=True)
        self.external_align_adjustment_vector = torch.tensor(self.learnable_args['align_adjustment_vector'], requires_grad=True)
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

    async def forward(self, state, observation):
        """
        LLMAgent class has three functions: a) mask sub-groups, b) format_prompt, c) invoke LLM, d) aggregate response
        """
        # if in debug mode, return random values for isolation
        if self.mode == "debug":
            will_isolate = torch.rand(self.num_agents, 1).to(self.device)
            return {self.output_variables[0]: will_isolate}

        # figure out time step
        time_step = int(state["current_step"])
        week_index = time_step // 7
        align_vector = self.external_align_vector

        # prompts are segregated based on agent age
        masks = []
        agent_age = get_by_path(state, re.split("/", self.input_variables["age"]))
        for age_group in AgeGroup:
            # agent subgroups for each prompt
            age_mask = agent_age == age_group.value
            masks.append(age_mask.float())

        if self.align_llm:
            all_align_mask = torch.zeros((self.num_agents, 1)).to(self.device)
            for en in range(len(masks)):
                all_align_mask = all_align_mask + align_vector[en]*masks[en]
            all_adjust_mask = torch.zeros((self.num_agents, 1)).to(self.device)
            for en in range(len(masks)):
                all_adjust_mask = (
                    all_adjust_mask
                    + self.external_align_adjustment_vector[en] * masks[en]
                )

        # if beginning of the week, recalculate isolation probabilities
        if time_step % 7 == 0:
            # figure out week index
            week_index = time_step // 7
            print(f"\nstarting week {week_index}...", end=" ")

            # update case numbers
            if week_index > 0:
                cases = sum(
                    state["environment"]["daily_infected"][time_step - 7 : time_step]
                )
                print(f"incoming #cases {cases}...", end=" ")
                if not self.use_ground_truth_case_numbers:
                    self.cases_week.append(cases)
                if not self.use_ground_truth_4_week_avg and week_index > 2:
                    self.cases_4_week_avg.append(sum(self.cases_week[-4:])/4)

            print(
                f"#cases, #cases_4wk for prompt {int(self.cases_week[week_index])}, "
                + f"{int(self.cases_4_week_avg[week_index])}... sampling isolation probabilities"
            )

            # # prompts are segregated based on agent age
            # masks = []
            # agent_age = get_by_path(state, re.split("/", self.input_variables["age"]))
            # for age_group in AgeGroup:
            #     # agent subgroups for each prompt
            #     age_mask = agent_age == age_group.value
            #     masks.append(age_mask.float())

            # generate the prompt list, prompt 7 times for each age group for each week to get
            # probabilities
            prompt_list = [
                {
                    "age": age_group.text,
                    "location": self.neighborhood.text,
                    "user_prompt": construct_user_prompt(
                        self.include_week_count,
                        self.epiweek_start,
                        week_index,
                        # this is a bug. case data is in float format in csv. should get it in int
                        # to avoid further confusion.
                        int(self.cases_week[week_index]),
                        int(self.cases_4_week_avg[week_index]),
                    ),
                }
                for age_group in AgeGroup
            ] * 7

            # time.sleep(1)
            # execute prompts from LLMAgent and compile response
            agent_output = await self.agent(prompt_list)

            # this is a bug. this probably belongs in the init function, but I don't know if this
            # class gets initialized at each episode. if not, this would cause the week 0 running
            # average to include the last episode's last week.
            if time_step == 0:
                self.isolation_probabilities = (
                    torch.ones((self.num_agents, 1)).to(self.device) / 2
                )

            # assign prompt response to agents
            self.last_isolation_probabilities = self.isolation_probabilities
            self.isolation_probabilities = torch.zeros((self.num_agents, 1)).to(self.device)
            for en, output_value in enumerate(agent_output):
                output_response = output_value["text"]
                decision = output_response.split(".")[0]
                # reasoning to be saved for RAG later
                reasoning = output_response.split(".")[1]
                isolation_response = self.string_to_number(decision)
                # this is a bug. it is a bit hacky, currently prompt_list[0], prompt_list[6], 12
                # etc. belong to the age_group 0, while 1, 7 etc. belong to the age_group 1, and
                # there are 7 responses for each group so we add each one of them with 1/7 weight

                self.isolation_probabilities = self.isolation_probabilities + (
                    masks[en % len(AgeGroup)] * isolation_response * 1 / 7
                )

        # aligned_isolation_probs = all_align_mask*((self.last_isolation_probabilities + self.isolation_probabilities) / 2)
        isolation_probs = (self.last_isolation_probabilities/2 + self.isolation_probabilities/2)
        if self.align_llm:
            isolation_probs = isolation_probs*all_align_mask + all_adjust_mask
        isolation_probs = torch.clamp(isolation_probs, 0, 1)
        will_isolate = self.st_bernoulli(isolation_probs)

        # print("aligned_isolation_probs: ", isolation_probs.shape, "will_isolate: ", will_isolate.shape)

        # # sample isolation decision from probabilities
        # will_isolate = StraightThroughBernoulli.apply(
        #     self.last_isolation_probabilities * 1 / 2
        #     + self.isolation_probabilities * 1 / 2 )

        print(f"day {time_step}, number of isolating agents {sum(will_isolate).item()}")

        return {self.output_variables[0]: will_isolate}
