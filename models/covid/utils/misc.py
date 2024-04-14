from epiweeks import Week

from utils.data import get_nn_data, Feature
from utils.region import Neighborhood


def get_initial_infection_rate(neighborhood: Neighborhood, epiweek: Week):
    # get number of cases for the previous week
    num_cases = get_nn_data(neighborhood, epiweek - 1, [], Feature.CASES).label
    # divide by population
    return num_cases / neighborhood.population
