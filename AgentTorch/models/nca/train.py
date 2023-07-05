import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import imageio

from AgentTorch import Runner, Registry
from AgentTorch.helpers import *

def create_registry():
    reg = Registry()
    # transition
    from substeps.evolve_cell.transition import NCAEvolve
    reg.register(NCAEvolve, "NCAEvolve", key="transition")

    from utils.initialization.nca import nca_initialize_state
    from utils.initialization.opinion import grid_network
    reg.register(nca_initialize_state, "nca_initialize_state", key="initialization")
    
    reg.register(grid_network, "grid", key="network")
    
    return reg

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

# *************************************************************************
# Parsing command line arguments
parser = argparse.ArgumentParser(
    description="AgentTorch: design, simulate and optimize agent-based models"
)
parser.add_argument(
    "-c", "--config", help="Name of the yaml config file with the parameters."
)
# *************************************************************************
args = parser.parse_args()
config_file = args.config

config = read_config(config_file)
registry = create_registry()

runner = NCARunner(config, registry)
device = torch.device(runner.config['simulation_metadata']['device'])
# *************************************************************************
# Generating target
def load_emoji(index, path="data/emoji.png"):
    im = imageio.imread(path)
    emoji = np.array(im[:, index*40:(index+1)*40].astype(np.float32))
    emoji /= 255.0
    return emoji
TARGET_EMOJI = 0
TARGET_PADDING = 16
target_img = load_emoji(TARGET_EMOJI)
p = TARGET_PADDING
pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
h, w = pad_target.shape[:2]
pad_target = np.expand_dims(pad_target, axis=0)
pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(device)
# *************************************************************************

optimizer = optim.Adam(runner.parameters(), 
                lr=runner.config['simulation_metadata']['learning_params']['lr'], 
                betas=runner.config['simulation_metadata']['learning_params']['betas'])
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 
                runner.config['simulation_metadata']['learning_params']['lr_gamma'])
loss_log = []

for ix in range(runner.config['simulation_metadata']['num_episodes']):
    runner.reset()
    optimizer.zero_grad()
    runner.execute()
    output = runner.state_trajectory[-1][-1]
    x = output['agents']['automata']['cell_state']
    loss = F.mse_loss(x[:, :, :, :4], pad_target)
    try:
        loss.backward()
    except:
        import ipdb; ipdb.set_trace()
    optimizer.step()
    scheduler.step()
    loss_log.append(loss.item())    
torch.save(runner.state_dict(), runner.config['simulation_metadata']['learning_params']['model_path'])
