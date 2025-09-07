import numpy as np
import pandas as pd
from scipy.stats import gamma
import networkx as nx
import torch.nn.functional as F
import pdb

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


def get_lam_gamma_integrals(shape, params):
    scale, rate = params["scale"], params["rate"]
    t = params["t"]
    b = rate * rate / scale
    a = scale / b

    a, b, t = a.cpu(), b.cpu(), t.cpu()

    res = [
        (gamma.cdf(t_i, a=a, loc=0, scale=b) - gamma.cdf(t_i - 1, a=a, loc=0, scale=b))
        for t_i in range(int(t.item()))
    ]
    res = np.array(res)
    return torch.tensor(res).float()


def get_infected_time(shape, params):
    agents_stages = read_from_file(shape, params)

    agents_infected_time = (500) * torch.ones_like(
        agents_stages
    )  # init all values to infinity time
    agents_infected_time[agents_stages == 1] = (
        -1
    )  # set previously infected agents to -1
    agents_infected_time[agents_stages == 2] = -3  # -1*exposed_to_infected_time

    return agents_infected_time.float()


def get_next_stage_time(
    shape, params, exposed_to_infected_times=3, infected_to_recovered_times=5
):
    agents_stages = read_from_file(shape, params)

    agents_next_stage_time = (500) * torch.ones_like(
        agents_stages
    )  # init all values to infinity time
    agents_next_stage_time[agents_stages == 1] = exposed_to_infected_times
    agents_next_stage_time[agents_stages == 2] = (
        infected_to_recovered_times  # infected_to_recovered time
    )

    return agents_next_stage_time.float()


def initialize_infections(shape, params):
    num_agents = params["num_agents"]
    initial_infections_ratio = params["initial_infected_ratio"]

    prob_infected = initial_infections_ratio * torch.ones((num_agents, 1))
    p = torch.hstack((prob_infected, 1 - prob_infected))
    cat_logits = torch.log(p + 1e-9)
    agents_stages = F.gumbel_softmax(logits=cat_logits, tau=1, hard=True, dim=1)[:, 0]

    return agent_stages.unsqueeze(1)


def read_from_file(shape, params):
    file_path = params["file_path"]
    if file_path.endswith("csv"):
        # Read without header, coerce to numeric
        df = pd.read_csv(file_path, header=None)
        values = pd.to_numeric(df.stack(), errors='coerce').unstack().to_numpy()
    else:
        df = pd.read_pickle(file_path)
        values = df.values
    # Ensure 2D
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    assert values.shape == tuple(shape), f"read_from_file: shape mismatch {values.shape} vs {tuple(shape)} for {file_path}"
    return torch.from_numpy(values)


def get_mean_agent_interactions(shape, params):
    agents_ages = load_population_attribute(shape, params)

    ADULT_LOWER_INDEX, ADULT_UPPER_INDEX = (
        1,
        4,
    )  # ('U19', '20t29', '30t39', '40t49', '50t64', '65A')

    agents_mean_interactions = 0 * torch.ones(size=shape)  # shape: (num_agents)
    mean_int_ran_mu = torch.tensor([2, 3, 4]).float()  # child, adult, elderly

    child_agents = (agents_ages < ADULT_LOWER_INDEX).view(-1)
    adult_agents = torch.logical_and(
        agents_ages >= ADULT_LOWER_INDEX, agents_ages <= ADULT_UPPER_INDEX
    ).view(-1)
    elderly_agents = (agents_ages > ADULT_UPPER_INDEX).view(-1)

    agents_mean_interactions[child_agents.bool(), 0] = mean_int_ran_mu[0]
    agents_mean_interactions[adult_agents.bool(), 0] = mean_int_ran_mu[1]
    agents_mean_interactions[elderly_agents.bool(), 0] = mean_int_ran_mu[2]

    return agents_mean_interactions


def load_population_attribute(shape, params):
    """
    Load population data from a pandas dataframe
    """
    # Load population data
    df = pd.read_pickle(params["file_path"])
    att_tensor = torch.from_numpy(df.values).float()
    return att_tensor.unsqueeze(1)


def initialize_id(shape, params):
    """
    Initialize a unique ID for each agent
    """
    return torch.arange(0, shape[0]).reshape(-1, 1).float()


def network_from_file(params):
    file_path = params["file_path"]
    # Read edge list, tolerate headers; coerce to numeric
    df = pd.read_csv(file_path, header=None)
    # Detect possible header row with non-numeric tokens
    if df.iloc[0].apply(lambda v: isinstance(v, str)).any():
        df = pd.read_csv(file_path, header=0)
    # Use first two columns as source/target
    df = df.iloc[:, :2]
    df = df.apply(pd.to_numeric, errors='coerce').dropna().astype('int64')

    # Build edge_index (2, E) and edge_attr (2, E) without creating dense adjacency
    edge_index = torch.tensor(df.values.T, dtype=torch.long)
    E = edge_index.shape[1]
    edge_type = torch.ones(E, dtype=torch.long)
    edge_weight = torch.ones(E, dtype=torch.float32)
    edge_attr = torch.vstack((edge_type, edge_weight))

    # Return no dense graph; provide edge_index/edge_attr tuple as adjacency_matrix payload
    return None, (edge_index, edge_attr)
