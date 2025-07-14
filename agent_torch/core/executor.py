import importlib
import sys
from tqdm import trange
import dask.dataframe as dd
from agent_torch.core.dataloader import DataLoader
from agent_torch.core.runner import Runner


class BaseExecutor:
    def __init__(self, model):
        self.model = model

    def _get_runner(self, config):
        module_name = self.model.__name__
        module = importlib.import_module(module_name)

        registry = module.registry
        print("Registry: ", registry)
        runner = Runner(config, registry)
        return runner


class Executor(BaseExecutor):
    def __init__(self, model, data_loader=None, pop_loader=None) -> None:
        super().__init__(model)
        self.model = model
        if pop_loader:
            self.pop_loader = pop_loader
            self.data_loader = DataLoader(model, self.pop_loader)
        else:
            self.data_loader = data_loader

        self.config = self.data_loader.get_config()
        self.runner = self._get_runner(self.config)

    def init(self):
        self.config = self.data_loader.get_config()
        self.runner = self._get_runner(self.config)
        self.runner.init()
        # self.learnable_params = [
        #     param for param in self.runner.parameters() if param.requires_grad
        # ]
        # self.opt = opt(self.learnable_params)

    def execute(self, key=None):
        num_episodes = self.config["simulation_metadata"]["num_episodes"]
        num_steps_per_episode = self.config["simulation_metadata"][
            "num_steps_per_episode"
        ]

        for episode in trange(num_episodes):
            # self.opt.zero_grad()
            self.runner.reset()
            self.runner.step(num_steps_per_episode)

        if key is not None:
            self.simulation_values = self.runner.get_simulation_values(key)

    def get_simulation_values(self, key, key_type="environment"):
        if isinstance(self.runner.state_trajectory, dd.DataFrame):
            self.runner.state_trajectory = self.runner.state_trajectory.compute()

        self.simulation_values = self.runner.state_trajectory[-1][-1][key_type][key]
        return self.simulation_values
