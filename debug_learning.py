import torch.nn as nn


class MetaClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_dict = nn.M

class RandomLearnableModule(nn.Module):
    def __init__(self, learnable_params, fixed_params):
        super().__init__()
        
        print('parse arguments and input_variables in the initializer.')
        
        if self.learnable_params is not None:
            self.learnable_params = nn.ParameterDict(learnable_params)
        else:
            self.learnable_params = learnable_params
            
        self.fixed_params = fixed_params
                
    def forward(self):
        pass
