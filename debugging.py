from agent_torch.models.covid.simulator import get_runner, get_registry
from agent_torch.core.helpers import read_config

config_path = 'agent_torch/models/covid/yamls/config.yaml'

config = read_config(config_path)
registry = get_registry()
runner = get_runner(config, registry)

breakpoint()
runner.init()
named_params_learnable = [(name, param) for (name, param) in runner.named_parameters() if param.requires_grad]
print("named learnable_params: ", named_params_learnable)

breakpoint()
runner.step(2)

traj = runner.state_trajectory[-1][-1]
preds = traj['environment']['daily_infected']
loss = preds.sum()
breakpoint()

loss.backward()
print([p.grad for p in runner.parameters()])

breakpoint()
