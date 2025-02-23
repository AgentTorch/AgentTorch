import pytest
import vaex
import tempfile
import os
import numpy as np
from pathlib import Path
import pandas as pd
from agent_torch.data.census.dataloader import AgentDataLoader


@pytest.fixture
def sample_age_gender_data():
    data = []
    areas = ["TX0001", "TX0002"]
    genders = ["male", "female"]
    age_groups = [
        "U5",
        "5t9",
        "10t14",
        "20t21",
        "80t84",
        "85plus",
    ]

    for area in areas:
        for gender in genders:
            for age in age_groups:
                data.append(
                    {
                        "area": area,
                        "gender": gender,
                        "age": age,
                        "count": np.random.randint(100, 3000),
                        "region": area[:2],
                    }
                )
    return pd.DataFrame(data)


@pytest.fixture
def sample_parquet_files(sample_age_gender_data, tmp_path):
    age_gender_path = tmp_path / "age_gender.parquet"

    sample_age_gender_data.to_parquet(str(age_gender_path))

    return {"age_gender": age_gender_path}


def test_load_demographic_data(sample_parquet_files):
    loader = AgentDataLoader()

    df_age_gender = loader.load_parquet_agents(str(sample_parquet_files["age_gender"]))
    assert isinstance(df_age_gender, vaex.dataframe.DataFrame)
    assert set(df_age_gender.column_names) == {
        "area",
        "gender",
        "age",
        "count",
        "region",
    }
    assert len(df_age_gender) == 24  # 2 areas * 2 genders * 6 age groups


def test_load_specific_columns(sample_parquet_files):
    loader = AgentDataLoader()

    # test age-gender specific columns
    df = loader.load_parquet_agents(
        str(sample_parquet_files["age_gender"]), columns=["area", "gender", "count"]
    )
    assert set(df.column_names) == {"area", "gender", "count"}


def test_error_handling(sample_parquet_files):
    loader = AgentDataLoader()

    # test invalid columns
    df = loader.load_parquet_agents(
        str(sample_parquet_files["age_gender"]), columns=["invalid_column"]
    )
    assert df is None

    # test invalid file path
    df = loader.load_parquet_agents("invalid_file.parquet")
    assert df is None

    # test missing base directory
    with pytest.raises(ValueError, match="base_dir must be specified"):
        loader.load_agent_files()
