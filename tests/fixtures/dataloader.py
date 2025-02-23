import pytest
import vaex
from pathlib import Path


@pytest.fixture
def sample_parquet_file():
    df = vaex.from_arrays(x=[1, 2, 3], y=[4, 5, 6])
    file_path = "/u/almurph/AgentTorch/tests/fixtures/" + "sample.parquet"
    df.export_parquet(file_path)
    return file_path
