import pytest

from models import covid
from populations import astoria

from agent_torch.dataloader import LoadPopulation
from agent_torch.executor import Executor


@pytest.fixture
def executor():
    loader = LoadPopulation(astoria)
    executor = Executor(model=covid, pop_loader=loader)

    return executor
