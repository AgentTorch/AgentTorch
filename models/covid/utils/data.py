from __future__ import annotations

from epiweeks import Week
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils.feature import Feature
from utils.misc import epiweek_to_week_num, week_num_to_epiweek
from utils.neighborhood import Neighborhood

DATAPATH = "./data/county_data.csv"
TABLE = pd.read_csv(DATAPATH)
DATA_START_WEEK = week_num_to_epiweek(TABLE["epiweek"].iloc[0])
DATA_END_WEEK = week_num_to_epiweek(TABLE["epiweek"].iloc[-1])
NN_INPUT_WEEKS = 3
DTYPE = torch.float32


class NNDataset(Dataset):
    def __init__(self, metadata, features):
        # shape [n_data, len(Neighborhood)]
        self.metadata: torch.Tensor = metadata
        # shape [n_data, NN_INPUT_WEEKS, len(feature_list)]
        self.features: torch.Tensor = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (
            self.metadata[idx],
            self.features[idx, :, :],
        )


def get_data(
    neighborhood: Neighborhood,
    epiweek_start: Week,
    num_weeks: int,
    feature_list: list[Feature],
):
    # filter by neighborhood
    table = TABLE.query(f"nta_id == '{neighborhood.nta_id}'")

    # get the epiweek index
    week_num = epiweek_to_week_num(epiweek_start)
    week_index = table.query(f"epiweek == {week_num}").index[0]

    # features_vector shape: [num_weeks, len(feature_list)]
    features_table = table[week_index : week_index + num_weeks][
        [feature.column_name for feature in feature_list]
    ]
    features_vector = torch.tensor(features_table.values, dtype=DTYPE)

    # return features
    return features_vector


def get_dataloader(
    neighborhood: Neighborhood,
    epiweek_start: Week,
    num_weeks: int,
    feature_list: list[Feature],
):
    # check that the start and end dates are valid
    if epiweek_to_week_num(epiweek_start) < epiweek_to_week_num(
        DATA_START_WEEK - NN_INPUT_WEEKS
    ) or epiweek_to_week_num(epiweek_start + num_weeks) > epiweek_to_week_num(
        DATA_END_WEEK
    ):
        raise Exception("epiweeks out of bounds")

    # gather data for each week
    features = []
    metadata = []

    for i in range(num_weeks):
        # feature shape: [NN_INPUT_WEEKS, len(feature_list)],
        prediction_week = epiweek_start + i
        features_week = get_data(
            neighborhood, prediction_week - NN_INPUT_WEEKS, NN_INPUT_WEEKS, feature_list
        )
        features.append(features_week)

        # neighborhood vector shape: [len(Neighborhood)]
        neighborhood_vector = torch.eye(len(Neighborhood), dtype=DTYPE)[
            neighborhood.value
        ]
        metadata.append(neighborhood_vector)

    # metadata shape: [num_weeks, len(Neighborhood)]
    metadata = torch.stack(metadata)
    # X shape: [num_weeks, NN_INPUT_WEEKS, len(feature_list)]
    X = torch.stack(features)

    # create data utils
    dataset = NNDataset(metadata, X)
    dataloader = DataLoader(dataset, batch_size=num_weeks)
    return dataloader


def get_labels(
    neighborhood: Neighborhood,
    epiweek_start: Week,
    num_weeks: int,
    label_feature: Feature,
):
    # labels shape: [num_weeks, 1]
    return get_data(neighborhood, epiweek_start, num_weeks, [label_feature])[:, 0]
