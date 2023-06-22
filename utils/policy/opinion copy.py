import torch
import torch.nn as nn
import numpy as np
import re

from utils.general import *


class PurchaseProduct(nn.Module):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__()
        self.config = config
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.learnable_args, self.fixed_args = arguments['learnable'], arguments['fixed']
        if self.learnable_args is not None:
            self.learnable_args = nn.ParameterDict(self.learnable_args)

    def forward(self, state, observation):        
        F_t = observation['F_t'].reshape(-1,1)
        Q_exp = observation['Q_exp']
        Q_des = observation['Q_des'].reshape(-1,1)
        N_i = observation['N_i']
        N_p = observation['N_p'].reshape(-1,1)

        step_id = state["current_step"]
        substep_id = state["current_substep"]
        
        utility = ((1-F_t)*(Q_exp - Q_des) + 
              torch.nan_to_num((step_id > 0)*F_t*N_i/N_p))
        
        argmax_utility = torch.argmax(utility, axis=1) # replace with gumbel softmax trick?
        max_utility_Q_exp = torch.take_along_dim(Q_exp, argmax_utility.reshape(-1,1), dim=1).reshape(-1)
        actions = ((max_utility_Q_exp - Q_des.reshape(-1)) > 0)*(argmax_utility+1) #1 added since 0 implies no purchase
        
        return {self.output_variables[0] : actions}
