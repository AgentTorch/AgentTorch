from omegaconf import OmegaConf
import torch
import torch.nn as nn
import numpy as np

class Configurator(nn.Module):
    '''TO BE DONE'''
    def __init__(self):
        super().__init__()
        
        self.metadata = OmegaConf.create()
        
        self.agents = OmegaConf.create()
        self.environment = OmegaConf.create()
        self.objects = OmegaConf.create()
        self.network = OmegaConf.create()
        
        self.substeps = OmegaConf.create()
        
        self.state = OmegaConf.create({'environment': self.environment, 'agents': self.agents, 'objects': self.objects, 'network': self.network})
        
        self.config = OmegaConf.create({'metadata': self.metadata, 'state': self.state, 'substeps': self.substeps})


    def create_variable(self, key, name, shape, dtype, initialization_function, learnable=False, value=None):
        '''Fundamental unit of an AgentTorch simulator which is learnable or not'''
        variable_dict = OmegaConf.create()
        variable_dict.update({'name': name})
        variable_dict.update({'shape': shape})
        variable_dict.update({'initialization_function': initialization_function})
        variable_dict.update({'learnable': learnable})
        variable_dict.update({'dtype': dtype})
        
        if initialization_function is None:
            variable_dict.update({'value': value})
                                        
        return OmegaConf.create({key: variable_dict})
    
    def create_initializer(self, generator, arguments):
        initializer = OmegaConf.create()
        
        initializer.update({'generator': generator})
        
        arguments_dict = arguments[0]
        for argument in arguments[1:]:
            argument_dict = OmegaConf.merge(argument_dict, argument)
        
        initializer.update({'arguments': arguments_dict})
        
        return initializer
        
    def add_metadata(self, key, value):
        self.config['metadata'].update({key: value})
    
    def get(self, variable_name):
        return OmegaConf.select(self.config, variable_name)
    
    def add_agents(self, key, number, all_properties=None):
        _created_agent = OmegaConf.create()
        if all_properties is None:
            all_properties = OmegaConf.create()
            
        _created_agent.update({'number': number, 'properties': all_properties})
        self.agents.update({key: _created_agent})
        
    def add_property(self, agent_name, key, name, shape, dtype, initialization_function, learnable=False, value=None):
        root_object = conf.get(agent_name)
        property_object = self.create_variable(key=key, name=name, shape=shape, dtype=dtype, initialization_function=initialization_function, learnable=learnable, value=value)
        
        root_object['properties'].update({property_object.key(): property})
        
            
if __name__ == '__main__':
    
    conf = Configurator()
    
    # simulation parameters
    conf.add_metadata('num_episodes', 3)
    conf.add_metadata('num_steps_per_episode', 20)
    conf.add_metadata('num_substeps_per_step', 1)
    conf.add_metadata('h', 72)
    conf.add_metadata('w', 72)
    conf.add_metadata('n_channels', 16)
    conf.add_metadata('batch_size', 8)
    conf.add_metadata('device', 'cpu')
    conf.add_metadata('hidden_size', 128)
    conf.add_metadata('fire_rate', 0.5)
    conf.add_metadata('angle', 0.0)
    conf.add_metadata('learning_params', {'lr': 2e-3, 'betas': [0.5, 0.5], 'lr_gamma': 0.9999, 'model_path': 'saved_model.pth'})
    
    w, h = conf.get('metadata.w'), conf.get('metadata.h')    
    automata_number = h*w
    
    n_channels = conf.get('metadata.n_channels')
    
    # Create Agent
    automata = conf.add_agents(key="automata", number=automata_number)
    
    # Populate agent properties (is a variable): initialization function which has arguments also a variable.
    arguments_list = [conf.create_variable(key='n_channels', name="n_channels", learnable=False, shape=(1,), initialization_function=None, value=n_channels, dtype="int"),
                conf.create_variable(key='batch_size', name="batch_size", learnable=False, shape=(1,), initialization_function=None, value='metadata.batch_size', dtype="int"),
                conf.create_variable(key='device', name="device", learnable=False, shape=(1,), initialization_function=None, value='metadata.device', dtype="str")]

    from utils.initialization.nca import nca_initialize_state
    cell_state_initializer = conf.create_initializer(generator = nca_initialize_state, arguments=arguments_list)
    
    automata_cell_state = conf.add_property(root='agents.automata', key='cell_state', learnable=True, shape=(n_channels,), initialization_function=cell_state_initializer, dtype="float")

        
    