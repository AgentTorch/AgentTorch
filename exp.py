# re-use existing models and population data easily
from agent_torch.models import covid
from agent_torch.populations import sample

# use the executor to plug-n-play
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation

# agent_"torch" works seamlessly with the pytorch API
from torch.optim import SGD

loader = LoadPopulation(sample)
simulation = Executor(model=covid, pop_loader=loader)

simulation.init()
simulation.execute()