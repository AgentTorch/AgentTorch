import pytest

from agent_torch.config import Configurator


@pytest.fixture
def config():
    config = Configurator()
    return config


@pytest.fixture
def agent_count():
    return 10400


@pytest.fixture
def agent_properties():
    type_function = {
        "generator": "read_from_file",
        "arguments": {
            "file_path": {
                "name": "Types File",
                "shape": None,
                "initialization_function": None,
                "learnable": False,
                "value": "data/types.csv",
            }
        },
    }

    return {
        "type": {
            "name": "Type",
            "shape": [10400, 1],
            "dtype": "int",
            "initialization_function": type_function,
            "learnable": False,
        }
    }
