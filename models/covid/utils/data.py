from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from epiweeks import Week
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils.region import Neighborhood

DATAPATH = "./data/county_data.csv"
TABLE = pd.read_csv(DATAPATH)
DTYPE = torch.float32
INPUT_WEEKS = 3


def get_epiweek_num(epiweek):
    return int(epiweek.cdcformat())


class Feature(Enum):
    RETAIL_CHANGE = auto()
    GROCERY_CHANGE = auto()
    PARKS_CHANGE = auto()
    TRANSIT_CHANGE = auto()
    WORK_CHANGE = auto()
    RESIDENTIAL_CHANGE = auto()
    CASES = auto()

    @property
    def column_name(self) -> str:
        return {
            Feature.RETAIL_CHANGE: "retail_and_recreation_change_week",
            Feature.GROCERY_CHANGE: "grocery_and_pharmacy_change_week",
            Feature.PARKS_CHANGE: "parks_change_week",
            Feature.TRANSIT_CHANGE: "transit_stations_change_week",
            Feature.WORK_CHANGE: "workplaces_change_week",
            Feature.RESIDENTIAL_CHANGE: "residential_change_week",
            Feature.CASES: "cases_week",
        }[self]


@dataclass
class NNData:
    features: torch.Tensor
    label: torch.Tensor


class RDataset(Dataset):
    def __init__(self, metadata, features, labels):
        # shape [n_data, len(Neighborhood)]
        self.metadata: torch.Tensor = metadata
        # shape [n_data, INPUT_WEEKS, len(feature_list)]
        self.X: torch.Tensor = features
        # shape [n_data, 1]
        self.y: torch.Tensor = labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.metadata[idx],
            self.X[idx, :, :],
            self.y[idx],
        )


def get_nn_data(
    neighborhood: Neighborhood,
    epiweek: Week,
    feature_list: list[Feature],
    label_feature: Feature,
):
    # filter by neighborhood
    table = TABLE.query(f"nta_id == '{neighborhood.nta_id}'")

    # get the epiweek index
    epiweek_num = get_epiweek_num(epiweek)
    epiweek_index = table.query(f"epiweek == {epiweek_num}").index[0]
    if epiweek_index < INPUT_WEEKS:
        raise Exception("epiweek too early")

    # get features of the past INPUT_WEEKS weeks
    features_table = table[epiweek_index - INPUT_WEEKS : epiweek_index][
        [feature.column_name for feature in feature_list]
    ]
    features_vector = torch.tensor(features_table.values, dtype=DTYPE)

    # get labels for the current week
    labels_table = table.iloc[epiweek_index][label_feature.column_name]
    label = torch.tensor(labels_table, dtype=DTYPE)

    # return values
    return NNData(features=features_vector, label=label)


def get_dataloader(
    neighborhood: Neighborhood,
    epiweeks: list[Week],
    feature_list: list[Feature],
    label_feature: Feature,
):
    # gather data for each week
    features = []
    labels = []
    metadata = []

    for epiweek in epiweeks:
        # feature shape: [INPUT_WEEKS, len(feature_list)], label shape: [1]
        nn_data = get_nn_data(neighborhood, epiweek, feature_list, label_feature)
        features.append(nn_data.features)
        labels.append(nn_data.label)
        # neighborhood vector shape: [len(Neighborhood)]
        neighborhood_vector = torch.eye(len(Neighborhood), dtype=DTYPE)[
            neighborhood.value
        ]
        metadata.append(neighborhood_vector)

    # add the extra dimensions to the beginning
    metadata = torch.stack(metadata)
    X = torch.stack(features)
    y = torch.stack(labels)

    # create data utils
    dataset = RDataset(metadata, X, y)
    dataloader = DataLoader(dataset, batch_size=len(epiweeks))

    return dataloader
