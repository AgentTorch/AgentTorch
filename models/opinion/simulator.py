import pandas as pd
import numpy as np 
import torch
import jax
import torch.optim as optim

from AgentTorch import Runner, Registry

def opdyn_registry():
    reg = Registry()

    from substeps.purchase_product.transition import NewQExp, NewPurchasedBefore
    reg.register(NewQExp, "new_Q_exp", key="transition")
    reg.register(NewPurchasedBefore, "new_purchased_before", key="transition")

    from substeps.purchase_product.action import PurchaseProduct
    reg.register(PurchaseProduct, "purchase_product", key="policy")

    from substeps.purchase_product.observation import GetFromState, GetNeighborsSum, GetNeighborsSumReduced
    reg.register(GetFromState, "get_from_state", key="observation")
    reg.register(GetNeighborsSum, "get_neighbors_sum", key="observation")
    reg.register(GetNeighborsSumReduced, "get_neighbors_sum_reduced", key="observation")

    from AgentTorch.helpers import zeros, random_normal, constant, grid_network
    reg.register(zeros, "zeros", key="initialization")
    reg.register(random_normal, "random_normal", key="initialization")
    reg.register(constant, "constant", key="initialization")
    reg.register(grid_network, "grid", key="network")

    from substeps.utils import random_normal_col_by_col
    reg.register(random_normal_col_by_col, "random_normal_col_by_col", key="initialization")

    return reg

class OpDynRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.optimizer = optim.Adam(self.parameters(), 
        #         lr=self.config['simulation_metadata']['learning_params']['lr'], 
        #         betas=self.config['simulation_metadata']['learning_params']['betas'])

    def forward(self):
        for episode in range(self.config['simulation_metadata']['num_episodes']):
            num_steps_per_episode = self.config["simulation_metadata"]["num_steps_per_episode"]
            self.reset()
            self.step(num_steps_per_episode)

            #self.controller.learn_after_episode(jax.tree_map(lambda x: x[-1], self.trajectory), self.initializer, self.optimizer)

    def execute(self):
        self.forward()
