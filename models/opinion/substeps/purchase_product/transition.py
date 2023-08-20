import torch
import torch.nn as nn
from torch.distributions import Normal
from AgentTorch.substep import SubstepTransition

class NewPurchasedBefore(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                
    def forward(self, state, action):
        old_purchased_before = get_by_path(state, re.split("/", self.input_variables['Purchased_before']))
        new_purchased_before = old_purchased_before + action['consumers']['purchase_action']
        return {self.output_variables[0]: new_purchased_before}

class NewQExp(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, state, action):
        arguments = self.args
        input_variables = self.input_variables
                
        current_Q_exp = get_by_path(state, re.split("/", input_variables['Q_exp']))
        num_agents, num_products = current_Q_exp.shape[0], current_Q_exp.shape[1]
        
        new_Q_exp = ((1.0 - action['consumers']['purchase_action'].sum(axis=1)).unsqueeze(1))*(current_Q_exp)
        add_Q_exp_list = []
        
        for i in range(num_products):
            r_sample = Normal(loc = arguments[f"distribution_params_{i}_mu"], scale=arguments[f"distribution_params_{i}_sigma"]).rsample()
            add_Q_exp_i = (action['consumers']['purchase_action'][:,i]
                           *(
                    (state['agents']['consumers']['Purchased_before'][:,i]==0)*r_sample)
                    + 
                    (state['agents']['consumers']['Purchased_before'][:,i]!=0)*(current_Q_exp[:,i])
            )
            add_Q_exp_list.append(add_Q_exp_i)
        
        new_Q_exp += torch.stack(add_Q_exp_list, axis=1)

        return {self.output_variables[0]: new_Q_exp}