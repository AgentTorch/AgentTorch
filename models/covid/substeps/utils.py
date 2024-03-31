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

# from AgentTorch.helpers import read_from_file

def get_lam_gamma_integrals(shape, params):
    scale, rate = params['scale'], params['rate']
    t = params['t']
    b = rate * rate / scale
    a = scale / b
    
    a, b, t = a.cpu(), b.cpu(), t.cpu()
        
    res = [(gamma.cdf(t_i, a=a, loc=0, scale=b) - gamma.cdf(t_i-1, a=a, loc=0, scale=b)) for t_i in range(int(t.item()))]
    res = np.array(res)
    return torch.tensor(res).float()

def get_infected_time(shape, params):
    agents_stages = read_from_file(shape, params)
    
    agents_infected_time = (500)*torch.ones_like(agents_stages) # init all values to infinity time
    agents_infected_time[agents_stages==1] = -1 # set previously infected agents to -1
    agents_infected_time[agents_stages==2] = -3 # -1*exposed_to_infected_time
    
    return agents_infected_time.float()

def get_next_stage_time(shape, params, exposed_to_infected_times=3, infected_to_recovered_times=5):
    agents_stages = read_from_file(shape, params)
    
    agents_next_stage_time = (500)*torch.ones_like(agents_stages) # init all values to infinity time
    agents_next_stage_time[agents_stages==1] = exposed_to_infected_times
    agents_next_stage_time[agents_stages==2] = infected_to_recovered_times # infected_to_recovered time
    
    return agents_next_stage_time.float()

def initialize_infections(shape, params):
    num_agents = params['num_agents']
    initial_infections_ratio = params['initial_infected_ratio']
    
    prob_infected = initial_infections_ratio * torch.ones((num_agents,1))
    p = torch.hstack((prob_infected,1-prob_infected))
    cat_logits = torch.log(p+1e-9)
    agents_stages = F.gumbel_softmax(logits=cat_logits,tau=1,hard=True,dim=1)[:,0]

    return agent_stages.unsqueeze(1)

def read_from_file(shape, params):    
    file_path = params['file_path']
    
    if file_path[-3:] == 'csv':
        data = pd.read_csv(file_path)
        
    data_values = data.values
    assert data_values.shape == tuple(shape)
    
    data_tensor = torch.from_numpy(data_values)
        
    return data_tensor
    

def get_mean_agent_interactions(shape, params):
    agents_ages = load_population_attribute(shape, params)
    
    ADULT_LOWER_INDEX, ADULT_UPPER_INDEX = 1, 4 # ('U19', '20t29', '30t39', '40t49', '50t64', '65A')
    
    agents_mean_interactions = 0*torch.ones(size=shape) # shape: (num_agents)    
    mean_int_ran_mu = torch.tensor([2, 3, 4]).float() # child, adult, elderly
    
    child_agents = (agents_ages < ADULT_LOWER_INDEX).view(-1)
    adult_agents = torch.logical_and(agents_ages >= ADULT_LOWER_INDEX, agents_ages <= ADULT_UPPER_INDEX).view(-1)
    elderly_agents = (agents_ages > ADULT_UPPER_INDEX).view(-1)

    agents_mean_interactions[child_agents.bool(), 0] = mean_int_ran_mu[0]
    agents_mean_interactions[adult_agents.bool(), 0] = mean_int_ran_mu[1]
    agents_mean_interactions[elderly_agents.bool(), 0] = mean_int_ran_mu[2]
        
    return agents_mean_interactions
    
# def get_mean_agent_interactions(shape, params):
#     pdb.set_trace()
#     agents_ages = torch.from_numpy(pd.read_csv(params['file_path']).values)
#     CHILD_UPPER_INDEX, ADULT_UPPER_INDEX = 1, 5
    
#     agents_mean_interactions = 0*torch.ones(size=shape) # shape: (num_agents)
#     mean_int_ran_mu = torch.tensor([2, 3, 4]).float() # child, adult, elderly
    
#     child_agents = (agents_ages <= CHILD_UPPER_INDEX).view(-1)
#     adult_agents = torch.logical_and(agents_ages > CHILD_UPPER_INDEX, agents_ages <= ADULT_UPPER_INDEX).view(-1)
#     elderly_agents = (agents_ages > ADULT_UPPER_INDEX).view(-1)
    
#     agents_mean_interactions[child_agents.bool(), 0] = mean_int_ran_mu[0] #22
#     agents_mean_interactions[adult_agents.bool(), 0] = mean_int_ran_mu[1] #22
#     agents_mean_interactions[elderly_agents.bool(), 0] = mean_int_ran_mu[2] #22
    
#     pdb.set_trace()
        
#     return agents_mean_interactions

def load_population_attribute(shape, params):
    """
    Load population data from a pandas dataframe
    """
    # Load population data
    df = pd.read_pickle(params['file_path'])
    att_tensor = torch.from_numpy(df.values).float()
    return att_tensor.unsqueeze(1)

def initialize_id(shape, params):
    """
    Initialize a unique ID for each agent
    """
    return torch.arange(0, shape[0]).reshape(-1, 1)
    
def network_from_file(params):
    file_path = params['file_path']

    random_network_edgelist_forward = torch.tensor(pd.read_csv(file_path, header=None).to_numpy()).t().long()        
    random_network_edgelist_backward = torch.vstack((random_network_edgelist_forward[1,:],
                                                     random_network_edgelist_forward[0,:]))
    random_network_edgelist = torch.hstack((random_network_edgelist_forward, 
                                            random_network_edgelist_backward))
    random_network_edgeattr_type = torch.ones(random_network_edgelist.shape[1]).long()
    random_network_edgeattr_B_n = torch.ones(random_network_edgelist.shape[1]).float()
    random_network_edgeattr = torch.vstack((random_network_edgeattr_type, random_network_edgeattr_B_n))

    all_edgelist = torch.hstack((random_network_edgelist,))
    all_edgeattr = torch.hstack((random_network_edgeattr,))

    agents_data = Data(edge_index=all_edgelist, edge_attr=all_edgeattr)

    G = to_networkx(agents_data)
    A = torch.tensor(nx.adjacency_matrix(G).todense())

    return G, (all_edgelist, all_edgeattr)