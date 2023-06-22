import torch
import torch.nn as nn
import numpy as np
import re

from utils.general import *

class GetFromState(nn.Module):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables
        
        learnable_args, fixed_args = arguments['learnable'], arguments['fixed']
        self.learnable_args, self.fixed_args = learnable_args, fixed_args

        if learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

    def forward(self, state):
        input_variables = self.input_variables
                
        return {ix: get_by_path(state, re.split("/", input_variables[ix])) for ix in input_variables.keys()}


class GetNeighborsSum(nn.Module):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables
        
        learnable_args, fixed_args = arguments['learnable'], arguments['fixed']
        self.learnable_args, self.fixed_args = learnable_args, fixed_args

        if learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)
            
        self.args = {**self.fixed_args, **self.learnable_args}
                
    def forward(self, state):
        input_variables = self.input_variables
        
        adjacency_matrix = get_by_path(state, re.split("/", input_variables["adjacency_matrix"]))
        query_feature = get_by_path(state, re.split("/", input_variables["query_feature"]))
                
        neighborhood_hops = int(self.args["neighborhood"])

        neighbors_sum = torch.matmul((adjacency_matrix**neighborhood_hops), query_feature.long())
        
        return {self.output_variables[0]: neighbors_sum}
    
    
class GetNeighborsSumReduced(nn.Module):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables # ['N_p']
        
        learnable_args, fixed_args = arguments['learnable'], arguments['fixed']
        self.learnable_args, self.fixed_args = learnable_args, fixed_args

        if learnable_args:
            self.learnable_args = nn.ParameterDict(self.learnable_args)
            
        self.args = {**self.fixed_args, **self.learnable_args}

    def forward(self, state):
        input_variables = self.input_variables

        adjacency_matrix = get_by_path(state, re.split("/", input_variables["adjacency_matrix"]))
        query_feature = get_by_path(state, re.split("/", input_variables["query_feature"]))

        neighborhood_hops = int(self.args["neighborhood"])
        neighbors_sum = torch.matmul((adjacency_matrix**neighborhood_hops), query_feature.long())

        return {self.output_variables[0] : neighbors_sum.sum(axis=1)}
