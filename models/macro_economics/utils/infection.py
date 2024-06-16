from epiweeks import Week
import numpy as np
import torch
import torch.nn as nn

from macro_economics.utils.data import get_data
from macro_economics.utils.feature import Feature
from macro_economics.utils.misc import epiweek_to_week_num
from macro_economics.utils.neighborhood import Neighborhood

def initial_infection_ratio(neighborhood: Neighborhood, epiweek: Week):
    num_cases = get_data(neighborhood, epiweek - 1, 1, [Feature.CASES])[0, 0]
    return num_cases / neighborhood.population


def create_infection_csv(neighborhood: Neighborhood, epiweek: Week):
    # figure out initial infection ratio
    num_cases = get_data(neighborhood, epiweek - 1, 1, [Feature.CASES])[0, 0]
    initial_infection_ratio = num_cases / neighborhood.population

    correct_num_cases = False
    num_tries = 0
    while not correct_num_cases:
        # copied from disease_stage_file.ipynb
        prob_infected = initial_infection_ratio * torch.ones(
            (neighborhood.population, 1)
        )
        p = torch.hstack((prob_infected, 1 - prob_infected))
        cat_logits = torch.log(p + 1e-9)
        agent_stages = nn.functional.gumbel_softmax(
            logits=cat_logits, tau=1, hard=True, dim=1
        )[:, 0]
        tensor_np = agent_stages.numpy().astype(int)
        tensor_np = np.array(tensor_np, dtype=np.uint8)
        np.savetxt("temp_infections", tensor_np, delimiter='\n')

        # check if the generated file has the correct number of cases
        arr = np.loadtxt(f"temp_infections")
        if arr.sum() == num_cases:
            correct_num_cases = True
            # write the correct file with the line that says stages
            with open(
                f"stage_csvs/{neighborhood.name}_{epiweek_to_week_num(epiweek)}_stages.csv",
                "w",
            ) as f:
                f.write("stages\n")
                np.savetxt(f, tensor_np)

        num_tries += 1
        if num_tries >= 1000:
            raise Exception("failed to create disease stages file")
