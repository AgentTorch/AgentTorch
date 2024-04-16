from epiweeks import Week

from utils.data import DATA_START_WEEK
from utils.misc import subtract_epiweek

SYSTEM_PROMPT = """Consider a random person with the following attributes:
* age: {age}
* location: {location}
* employment: {employment}

There is a novel disease. It spreads through contact. It is more dangerous to older people.
People have the option to isolate at home or continue their usual recreational activities outside.
Given this scenario, you must estimate the person's actions based on
    1) the information you are given,
    2) what you know about the general population with these attributes.

"There isn't enough information" and "It is unclear" are not acceptable answers.
Give a "Yes" or "No" answer, followed by a period. Give one sentence explaining your choice.
"""


def construct_user_prompt(
    include_week_count: bool,
    epiweek_start: Week,
    week_index: int,
    cases: int,
    cases_4_week_avg: int,
):
    # create the prompt
    user_prompt = ""

    # add the week count
    if include_week_count:
        disease_week_count = subtract_epiweek(
            epiweek_start + week_index, DATA_START_WEEK
        )
        user_prompt += (
            f"It has been {disease_week_count} weeks since the disease started. "
        )

    # add the case news
    user_prompt += (
        f"This week, there are {cases} new cases in the neighborhood, which is "
    )

    # add the change from past month's average
    change = int((cases / cases_4_week_avg - 1) * 100)
    if change >= 1:
        user_prompt += f"a {change}% increase from "
    elif change <= -1:
        user_prompt += f"a {abs(change)}% decrease from "
    else:
        user_prompt += "the same as "

    # finish the prompt
    user_prompt += (
        "the past month's average.\nDoes this person choose to isolate at home?"
    )
    return user_prompt
