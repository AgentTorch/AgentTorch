import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import imageio

from AgentTorch.runner import Runner
from AgentTorch.helpers import *

class NCARunner(Runner):
    def __init__(self, **args, **kwargs):
        super().__init__(**args, **kwargs)

    def _nca_initialize_state(self, shape, params):
        device = torch.device(params['device'])
        batch_size = params['batch_size']
        n_channels = int(params['n_channels'].item())
        processed_shape = [process_shape_omega(s) for s in shape]
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