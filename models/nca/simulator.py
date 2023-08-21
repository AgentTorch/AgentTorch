import torch
import numpy as np
import sys
sys.path.insert(0, '../../')
from AgentTorch import Configurator, Runner
from AgentTorch.helpers import read_config

def configure_nca(config_path):
    conf = Configurator()

    # add metadata
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

    # create agent
    w, h = conf.get('simulation_metadata.w'), conf.get('simulation_metadata.h')    
    automata_number = h*w
    automata = conf.add_agents(key="automata", number=automata_number)

    # add agent properties
    n_channels = conf.get('simulation_metadata.n_channels')
    batch_size = conf.get('simulation_metadata.batch_size')
    device = conf.get('simulation_metadata.device')

    from substeps.utils import nca_initialize_state

    arguments_list = [conf.create_variable(key='n_channels', name="n_channels", learnable=False, shape=(1,), initialization_function=None, value=n_channels, dtype="int"),
                    conf.create_variable(key='batch_size', name="batch_size", learnable=False, shape=(1,), initialization_function=None, value=batch_size, dtype="int"),
                    conf.create_variable(key='device', name="device", learnable=False, shape=(1,), initialization_function=None, value=device, dtype="str")]

    cell_state_initializer = conf.create_initializer(generator = nca_initialize_state, arguments=arguments_list)

    conf.add_property(root='state.agents.automata', key='cell_state', name="cell_state", learnable=True, shape=(n_channels,), initialization_function=cell_state_initializer, dtype="float")

    # add environment network
    from AgentTorch.helpers import grid_network
    conf.add_network('evolution_network', grid_network, arguments={'shape': [w, h]})

    # add substep
    from substeps.evolve_cell.transition import NCAEvolve
    evolve_transition = conf.create_function(NCAEvolve, input_variables={'cell_state':'agents/automata/cell_state'}, output_variables=['cell_state'], fn_type="transition")
    conf.add_substep(name="Evolution", active_agents=["automata"], transition_fn=[evolve_transition])

    conf.render(config_path)

    return read_config(config_path), conf.reg

class NCARunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _nca_initialize_state(self, shape, params):
        device = torch.device(params['device'])
        batch_size = params['batch_size']
        n_channels = int(params['n_channels'].item())
        processed_shape = shape #[process_shape_omega(s) for s in shape]
        grid_shape = [np.sqrt(processed_shape[0]).astype(int), np.sqrt(processed_shape[0]).astype(int), processed_shape[1]]
        seed_x = np.zeros(grid_shape, np.float32)
        seed_x[grid_shape[0]//2, grid_shape[1]//2, 3:] = 1.0
        x0 = np.repeat(seed_x[None, ...], batch_size, 0)
        x0 = torch.from_numpy(x0.astype(np.float32)).to(device)
        return x0

    def reset(self):
        shape = [5184, 16]
        params = {'n_channels': torch.tensor([16.]), 'batch_size': torch.tensor([8.]), 'device': 'cpu'}
        x0 = self._nca_initialize_state(shape, params)
        self.state = self.initializer.state
        self.state['agents']['automata']['cell_state'] = x0