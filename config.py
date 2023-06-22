# Create Config file
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import numpy as np

class CreateConfig(nn.Module):
    def __init__(self):
        super().__init__()

        self.simulation_metadata = OmegaConf.create()
        
        self.agents = OmegaConf.create()
        self.environment = OmegaConf.create()
        self.objects = OmegaConf.create()
        self.network = OmegaConf.create()
        
        self.substeps = OmegaConf.create()
        
        print("!!!IN PROGRESS CODE.. DOES NOT CURRENTLY WORK!!!!")
            
    def create_variable(self, key, name, initialization_function, shape=(1,), dtype, learnable=False, value=None):
        variable_dict = OmegaConf.create()
        
        variable_dict.update({'name': name})
        variable_dict.update({'shape': shape})
        variable_dict.update({'initialization_function': initialization_function})
        variable_dict.update({'learnable': learnable})
        variable_dict.update({'dtype': dtype})
        
        if initialization_function is None:
            variable_dict.update({'value': value})
                                        
        return OmegaConf.create({key: variable_dict})
    
    def create_initializer(self, generator, arguments, dtype):
        initializer = OmegaConf.create()
        
        initializer.update({'generator': generator})
        initializer.update({'arguments': arguments})
        initializer.update({'dtype': dtype})
        
        return initializer
    
    def _create(self, number, all_properties):
        created_element = OmegaConf.create()
        created_element.update({'number': number, 'properties': all_properties})
        
        return created_element
    
    def add_metadata(self, key, value):
        self.simulation_metadata.update({key : value})

    def add_agent(self, agent_name, number, all_properties=None):
        _created_agent = self._create(number, all_properties)
        self.agents.update({agent_name: _created_agent})
    
    def add_object(self, object_name, number, all_properties):
        _created_object = self._create(number, all_properties)
        self.objects.update({object_name: _created_object})
        
    def add_network(self, network_name, network_type, arguments, category='agent_agent'):
        self.networks.update({category: {network_name: {'type': network_type, 'arguments': arguments}}})
        
    def add_environment_property(self, variable, name, initialization_function):
        _created_variable = self.create_property(name, initialization_function)
        self.environment.update({variable : _created_variable})
    
    def execute(self):
        
        self.state = OmegaConf.create({'environment': self.environment, 'agents': self.agents, 'objects': self.objects, 'network': self.network})
        
        self.config = OmegaConf.create({'simulation_metadata': self.simulation_metadata, 'state': self.state, 'substeps': self.substeps})
        
        # self.state = OmegaConf.merge(self.environment, self.agents, self.objects, self.network)
        # self.config = OmegaConf.merge(self.simulation_metadata, self.state, self.substeps)
    
    def forward(self):
        pass

if __name__ == '__main__':
    print("create config python API")
    
    conf = CreateConfig()
    
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
    n_channels = conf.get('metadata.n_channels')
    automata_number = h*w
    
    # Create an Agent and assign it properties
    automata = conf.add_agent(key="automata", number=automata_number)

    arguments = {'n_channels': conf.create_variable('n_channels', learnable=False, shape=(1,), initialization_function=None, value=n_channels, dtype="int"),
                'batch_size': conf.create_variable('batch_size', learnable=False, shape=(1,), initialization_function=None, value=${'metadata.batch_size'}, dtype="int"),
                'device': conf.create_variable('device', learnable=False, shape=(1,), initialization_function=None, value=${'metadata.device'}, dtype="str")}
    
    from utils.initialization.nca import nca_initialize_state
    cell_state_initializer = conf.create_initializer(generator = nca_initialize_state, arguments=arguments)
    
    automata_cell_state = conf.add_property('agents.automata', key='cell_state', learnable=True, shape=(n_channels,), initialization_function=cell_state_initializer, dtype="float")

    # Create network
    evolution_network = conf.add_network(key="evolution_network", network_type="grid", arguments={'shape'=[w, h]}, category="agent_agent")
    

    # Define substeps
    from utils.transitions.nca import NCAEvolve
    evolve_transition = conf.create_function(key="NCAEvolve", generator=NCAEvolve, arguments=None, input_variables={"cell_state": "agents.automata.cell_state"}, output_variables=["cell_state"])
    substep_0 = conf.create_substep('0', name="Evolution", description="All automata cell states evolve by one step", active_agents=[automata], observation=None, policy=None, transition=[evolve_transition], reward=None)