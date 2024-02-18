import pandas as pd
import torch
import torch.nn as nn
import json

class Registry(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.initialization_helpers = {}
        self.observation_helpers = {}
        self.policy_helpers = {}
        self.transition_helpers = {}
        self.network_helpers = {}
        
        self.helpers = {}
        self.helpers["transition"] = self.transition_helpers
        self.helpers["observation"] = self.observation_helpers
        self.helpers["policy"] = self.policy_helpers
        self.helpers["initialization"] = self.initialization_helpers
        self.helpers["network"] = self.network_helpers
        
    def register(self, obj_source, name, key):
        '''Inserts a new function into the registry'''
        self.helpers[key][name] = obj_source
    
    def view(self):
        '''Pretty prints the entire registry as a JSON object'''
        return json.dumps(self.helpers, indent=2)
    
    def forward(self):
        print("Invoke registry.register(class_obj, key)")

if __name__ == '__main__':
    reg = Registry()
    
    print("Register utilities for Opinion Dynamics model.")

    # transition
    from utils.transition.opinion import NewQExp, NewPurchasedBefore
    reg.register(NewQExp, "new_q_exp", key="transition")
    reg.register(NewPurchasedBefore, "new_purchased_before", key="transition")
        
    # policy
    from utils.policy.opinion import PurchaseProduct
    reg.register(PurchaseProduct, "purchase_product", key="policy")
    
    # observation
    from utils.observation.opinion import GetFromState, GetNeighborsSum, GetNeighborsSumReduced
    reg.register(GetFromState, "get_from_state", key="observation")
    reg.register(GetNeighborsSum, "get_neighbors_sum", key="observation")
    reg.register(GetNeighborsSumReduced, "get_neighbors_sum_reduced", key="observation")
    
    # initialization and network
    from utils.initialization.opinion import zeros, random_normal, constant, random_normal_col_by_col, grid_network
    reg.register(zeros, "zeros", key="initialization")
    reg.register(random_normal, "random_normal", key="initialization")
    reg.register(constant, "constant", key="initialization")
    reg.register(random_normal_col_by_col, "random_normal_col_by_col", key="initialization")
    
    reg.register(grid_network, "grid", key="network")

    import pdb; pdb.set_trace()
