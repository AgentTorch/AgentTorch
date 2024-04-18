from epiweeks import Week
import numpy as np
import torch
import torch.nn as nn

from utils.data import get_data
from utils.feature import Feature
from utils.neighborhood import Neighborhood

def initial_infection_ratio(neighborhood: Neighborhood, epiweek: Week):
    num_cases = get_data(neighborhood, epiweek - 1, 1, [Feature.CASES])[0, 0]
    return num_cases / neighborhood.population


def create_infection_csv(neighborhood: Neighborhood, epiweek: Week, save_path):
    """copied from disease_stage_file.ipynb"""
    prob_infected = initial_infection_ratio(neighborhood, epiweek) * torch.ones(
        (neighborhood.population, 1)
    )
    p = torch.hstack((prob_infected, 1 - prob_infected))
    cat_logits = torch.log(p + 1e-9)
    agent_stages = nn.functional.gumbel_softmax(
        logits=cat_logits, tau=1, hard=True, dim=1
    )[:, 0]
    tensor_np = agent_stages.numpy().astype(int)
    tensor_np = np.array(tensor_np, dtype=np.uint8)
    
    np.savetxt(save_path, tensor_np, delimiter='\n')
    # np.savetxt('disease_stages_fictional.csv', tensor_np, delimiter='\n')


if __name__ == '__main__':
    from utils.misc import week_num_to_epiweek, name_to_neighborhood
    neighborhood_name = "Astoria"
    week = 202212
    neighborhood = name_to_neighborhood(neighborhood_name)
    epiweek = week_num_to_epiweek(week)
    save_path = './stage_csvs/{}_{}_stages.csv'.format(neighborhood_name, week)

    create_infection_csv(neighborhood, epiweek, save_path)