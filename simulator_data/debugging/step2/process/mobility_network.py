import os
import pandas as pd
from scipy.stats import nbinom
import ray
import random
import networkx as nx

import base64
import uuid

def _get_agent_interactions(age, interaction_dict):
    mean, sd = interaction_dict[age]['mu'], interaction_dict[age]['sigma']
    
    p = mean / (sd*sd)
    n = mean * mean / (sd * sd - mean)
    num_interactions = nbinom.rvs(n, p)
    
    return num_interactions

def mobility_network_wrapper(age_df, num_steps, interaction_dict, age_category_dict, save_path=None):
    
    if save_path is None:
        unique_id = str(uuid.uuid4())
        encoded_id = base64.urlsafe_b64encode(unique_id.encode()).decode()
        save_path = '/tmp/random_mobility_networks_{}'.format(encoded_id)

    num_agents = age_df.shape[0]
    
    agents_random_interactions = [_get_agent_interactions(age_category_dict[age], interaction_dict) for age in age_df]
    
    interactions_list = []
    mobility_networks_list = []
    
    for t in range(num_steps):
        for agent_id in range(num_agents):
            interactions_list.extend([agent_id]*agents_random_interactions[agent_id])
        random.shuffle(interactions_list)
        
        edges_list = [(interactions_list[i], interactions_list[i+1]) for i in range(len(interactions_list)-1)]
        G = nx.Graph()
        G.add_edges_from(edges_list)
        
        outfile = os.path.join(save_path, '{}.csv'.format(t))
        nx.write_edgelist(G, outfile, delimiter = ',', data = False)
        mobility_networks_list.append(outfile)
    
    return mobility_networks_list
            