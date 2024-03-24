import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'AgentTorch'))
sys.path.append(os.path.join(os.getcwd(), 'AgentTorch/AgentTorch'))

from AgentTorch.dataloader import DataLoader
from AgentTorch.simulate import Executor

from models import macro_economics
from populations import NZ

num_agents = 16460
agents_sim = Executor(macro_economics, NZ, num_agents)
