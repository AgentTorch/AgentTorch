import torch
import torch.nn as nn
import numpy as np
import re
from AgentTorch.substep import SubstepObservation

class GetFromState(SubstepObservation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state):
        input_variables = self.input_variables
                
        return {ix: get_by_path(state, re.split("/", input_variables[ix])) for ix in input_variables.keys()}


class GetNeighborsSum(SubstepObservation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                
    def forward(self, state):
        input_variables = self.input_variables
        
        adjacency_matrix = get_by_path(state, re.split("/", input_variables["adjacency_matrix"]))
        query_feature = get_by_path(state, re.split("/", input_variables["query_feature"]))
                
        neighborhood_hops = int(self.args["neighborhood"])

        neighbors_sum = torch.matmul((adjacency_matrix**neighborhood_hops), query_feature.long())
        
        return {self.output_variables[0]: neighbors_sum}
    
    
class GetNeighborsSumReduced(SubstepObservation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
    def forward(self, state):
        input_variables = self.input_variables

        adjacency_matrix = get_by_path(state, re.split("/", input_variables["adjacency_matrix"]))
        query_feature = get_by_path(state, re.split("/", input_variables["query_feature"]))

        neighborhood_hops = int(self.args["neighborhood"])
        neighbors_sum = torch.matmul((adjacency_matrix**neighborhood_hops), query_feature.long())

        return {self.output_variables[0] : neighbors_sum.sum(axis=1)}
