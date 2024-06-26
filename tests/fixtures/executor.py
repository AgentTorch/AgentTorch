import pytest

from agent_torch.models import covid
from agent_torch.populations import astoria

from agent_torch.core.dataloader import LoadPopulation
from agent_torch.core.executor import Executor


@pytest.fixture
def executor():
    loader = LoadPopulation(astoria)
    executor = Executor(model=covid, pop_loader=loader)

    return executor
