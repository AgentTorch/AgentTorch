from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from openai import OpenAI

OPENAI_API_KEY = "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"

def _get_column_from_table(table, tag):
    return [row[tag] for row in table]

class County(Enum):
    QUEENS = auto()

    def get_name(self) -> str:
        return {
            County.QUEENS: "Queens County",
        }[self]


class Neighborhood(Enum):
    ASTORIA_SOUTH_LIC_SUNNYSIDE = auto()

    def get_modzcta(self) -> int:
        return {
            Neighborhood.ASTORIA_SOUTH_LIC_SUNNYSIDE: 11101,
        }[self]

    def get_nta_id(self) -> str:
        return {
            Neighborhood.ASTORIA_SOUTH_LIC_SUNNYSIDE: "QN0103",
        }[self]

    def get_population(self) -> int:
        return {
            Neighborhood.ASTORIA_SOUTH_LIC_SUNNYSIDE: 36835,
        }[self]


class Provider:
    def get_dates(self):
        return self.dates[self.start_index : self.end_index]

    def set_start_end_indices(self, start_date, end_date):
        self.start_index, self.end_index = self._parse_start_end_indices(
            start_date, end_date
        )

    def _parse_start_end_indices(self, start_date, end_date):
        return self.dates.index(start_date), self.dates.index(end_date) + 1

    def _set_start_end_caps(self, start_date, end_date):
        self.start_cap, self.end_cap = self._parse_start_end_indices(
            start_date, end_date
        )


class CaseProvider(Provider):
    def __init__(self):
        # read the data file into a table
        with open("./llm_data/caserate-by-modzcta.csv", mode="r", encoding="utf-8") as file:
            table = list(csv.DictReader(file))

        # extract the weeks
        self.dates: list[str] = _get_column_from_table(table, "week_ending")

        # set boundaries
        start_cap = "08/08/2020"
        end_cap = "10/15/2022"
        self._set_start_end_caps(start_cap, end_cap)

        # crop the dates
        self.dates: list[str] = self.dates[self.start_cap : self.end_cap]

        # extract the neighborhood data
        self.case_rates: dict[Neighborhood, list[float]] = {
            neighborhood: [
                float(case_rate_str)
                for case_rate_str in _get_column_from_table(
                    table, f"CASERATE_{neighborhood.get_modzcta()}"
                )
            ][self.start_cap : self.end_cap]
            for neighborhood in Neighborhood
        }

        # hardcoded values for now
        self.neighborhood = Neighborhood.ASTORIA_SOUTH_LIC_SUNNYSIDE
        self.set_start_end_indices("03/06/2021", "11/27/2021")

    def get_case_numbers(self):
        return [
            round(x * self.neighborhood.get_population() / 100000)
            for x in self._get_case_rates()
        ]

    def _get_case_rates(self):
        return self.case_rates[self.neighborhood][self.start_index : self.end_index]


class CensusProvider:
    def __init__(self):
        # read the data file
        data = np.load("llm_data/all_nta_agents.npy", allow_pickle=True).item()

        self.age_distribution = {
            neighborhood: self._parse_age_distribution(data, neighborhood)
            for neighborhood in Neighborhood
        }

        # hardcoded values for now
        self.neighborhood = Neighborhood.ASTORIA_SOUTH_LIC_SUNNYSIDE

    def get_age_distribution(self):
        return self.age_distribution[self.neighborhood]

    def _parse_age_distribution(self, data, neighborhood: Neighborhood) -> list[float]:
        nta_id = neighborhood.get_nta_id()
        return [
            data[nta_id]["age_gender_prob"][i]
            + data[nta_id]["age_gender_prob"][i + 1][0]
            for i in range(0, 12, 2)
        ]


class MobilityProvider(Provider):
    def __init__(self):
        # read the data file into a table
        with open(
            "./llm_data/Global_Mobility_Report.csv", mode="r", encoding="utf-8"
        ) as file:
            table = list(csv.DictReader(file))

        # filters:
        # - filter for new york
        # - crop out the starting saturday (needs to start on a sunday)
        table = [
            row
            for row in table
            if row["country_region_code"] == "US"
            and row["sub_region_1"] == "New York"
            and row["date"] != "2020-02-15"
        ]

        # extract the weeks
        self.dates: list[str] = sorted(set(_get_column_from_table(table, "date")))[6::7]

        # set boundaries
        start_cap = "2020-08-08"
        end_cap = "2022-10-15"
        self._set_start_end_caps(start_cap, end_cap)

        # crop the dates
        self.dates: list[str] = self.dates[self.start_cap : self.end_cap]

        # extract the changes in recreation
        self.recreation_values = {
            county: self._parse_recreation_values(table, county) for county in County
        }

        # hardcoded values for now
        self.county = County.QUEENS
        self.set_start_end_indices("2021-03-06", "2021-11-27")

    def get_dates(self):
        return self.dates[self.start_index : self.end_index]

    def get_recreation_values(self):
        return self.recreation_values[self.county][self.start_index : self.end_index]

    def _parse_recreation_values(self, table, county: County):
        data = [row for row in table if row["sub_region_2"] == county.get_name()]
        daily_changes = [
            int(change)
            for change in _get_column_from_table(
                data, "retail_and_recreation_percent_change_from_baseline"
            )
        ]
        weekly_average_changes = [
            sum(daily_changes[i : i + 7]) / 7 for i in range(0, len(daily_changes), 7)
        ]
        return [round(100 + change) for change in weekly_average_changes][
            self.start_cap : self.end_cap
        ]


class _ResponseType(Enum):
    SENTENCE = auto()
    TRUE_FALSE = auto()

    def get_max_tokens(self) -> int:
        return {
            _ResponseType.SENTENCE: 50,
            _ResponseType.TRUE_FALSE: 1,
        }[self]


class Chat:
    def __init__(self, system_prompt, max_history=100):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.full_history = [{"role": "system", "content": system_prompt}]
        self.max_history = max_history

    def add_prompt(self, user_prompt):
        if len(self.messages) > self.max_history:
            self.messages.pop(1)
            self.messages.pop(1)
        self.messages.append({"role": "user", "content": user_prompt})
        self.full_history.append({"role": "user", "content": user_prompt})

    def add_response(self, response):
        self.messages.append({"role": "assistant", "content": response})
        self.full_history.append({"role": "assistant", "content": response})

    def get_messages(self):
        return self.messages

    def get_full_history(self):
        return self.full_history

    def get_responses(self):
        return [
            message
            for message in self.get_full_history()
            if message["role"] == "assistant"
        ]


@dataclass
class Config:
    model: str
    temperature: float


def _get_content(
    client: OpenAI,
    prompt: Chat,
    config: Config,
    response_type: _ResponseType,
):
    return (
        client.chat.completions.create(
            model=config.model,
            messages=prompt.get_messages(),
            temperature=config.temperature,
            max_tokens=response_type.get_max_tokens(),
        )
        .choices[0]
        .message.content
    )


def get_answer(client: OpenAI, prompt: Chat, config: Config) -> str:
    return _get_content(client, prompt, config, _ResponseType.SENTENCE)


@dataclass
class YesNoAnswersResponse:
    answers: list[bool]
    num_inconclusive_answers: int


def get_yes_no_answers(
    client: OpenAI,
    chat: Chat,
    config: Config,
    num_queries: int,
) -> YesNoAnswersResponse:
    answers = []
    num_inconclusive_answers = 0
    while len(answers) < num_queries:
        answer = _get_content(client, chat, config, _ResponseType.TRUE_FALSE)
        if answer == "True":
            answers.append(True)
        elif answer == "False":
            answers.append(False)
        else:
            num_inconclusive_answers += 1
    return YesNoAnswersResponse(answers, num_inconclusive_answers)