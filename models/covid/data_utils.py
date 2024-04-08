import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from epiweeks import Week, Year
import pandas as pd
global N
import os

dtype = torch.float
WEEKS_AHEAD = 4
PAD_VALUE = -999
DAYS_IN_WEEK = 7
NOISE_LEVELS_FLU = [0.15, 0.25, 0.50, 0.75]
NOISE_LEVELS_COVID = [0.5, 1.0, 1.5, 2.0]
# daily
datapath = './Data/Processed/covid_data.csv'
# county_datapath = f'./Data/Processed/county_data.csv'
datapath_flu_hhs = './Data/Processed/flu_region_data.csv'
datapath_flu_state = './Data/Processed/flu_state_data.csv'
population_path = './Data/table_population.csv'
county_datapath = f'./data/MN_county_data.csv'
# EW_START_DATA = '202012'
# EW_START_DATA = '202022'
EW_START_DATA = '202045'
EW_START_DATA_FLU = '201740'  # for convenience

# Select signals COVID
macro_features = [
    'retail_and_recreation_percent_change_from_baseline',
    'grocery_and_pharmacy_percent_change_from_baseline',
    'parks_percent_change_from_baseline',
    'transit_stations_percent_change_from_baseline',
    'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline',
    'apple_mobility',
    'death_jhu_incidence',
    'positiveIncr',
]

# Select signals Flu
include_cols = [
    "symptom:Fever",
    "symptom:Low-grade fever",
    "symptom:Cough",
    "symptom:Sore throat",
    "symptom:Headache",
    "symptom:Fatigue",
    "symptom:Vomiting",
    "symptom:Diarrhea",
    "symptom:Shortness of breath",
    "symptom:Chest pain",
    "symptom:Dizziness",
    "symptom:Confusion",
    "symptom:Generalized tonic–clonic seizure",
    "symptom:Weakness",
]

states = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'ID',
    'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
    'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
    'WI', 'WY', 'X'
]

counties = {
    'MA': [
        '25001', '25003', '25005', '25009', '25011', '25013', '25015', '25021',
        '25023', '25027'
    ],
    'MN': [
        '27001', '27003', '27005', '27007', '27009', '27011', '27013', '27015',
        '27017', '27019', '27021', '27023', '27025', '27027', '27029', '27031',
        '27033', '27035', '27037', '27039', '27041', '27043', '27045', '27047',
        '27049', '27051', '27053', '27055', '27057', '27059', '27061', '27063',
        '27065', '27067', '27069', '27071', '27073', '27075', '27077', '27079',
        '27081', '27083', '27087', '27089', '27091', '27085', '27093', '27095',
        '27097', '27099', '27101', '27103', '27105', '27107', '27109', '27111',
        '27113', '27115', '27117', '27119', '27121', '27123', '27125', '27127',
        '27129', '27131', '27133', '27135', '27139', '27141', '27143', '27137',
        '27145', '27147', '27149', '27151', '27153', '27155', '27157', '27159',
        '27161', '27163', '27165', '27167', '27169', '27171', '27173'
    ],
    'GA': [
        '13001', '13003', '13005', '13007', '13009', '13011', '13013', '13015',
        '13017', '13019', '13021', '13023', '13025', '13027', '13029', '13031',
        '13033', '13035', '13037', '13039', '13041', '13043', '13045', '13047',
        '13049', '13051', '13053', '13055', '13057', '13059', '13061', '13063',
        '13065', '13067', '13069', '13071', '13073', '13075', '13077', '13079',
        '13081', '13083', '13085', '13087', '13089', '13091', '13093', '13095',
        '13097', '13099', '13101', '13103', '13105', '13107', '13109', '13111',
        '13113', '13115', '13117', '13119', '13121', '13123', '13125', '13127',
        '13129', '13131', '13133', '13135', '13137', '13139', '13141', '13143',
        '13145', '13147', '13149', '13151', '13153', '13155', '13157', '13159',
        '13161', '13163', '13165', '13167', '13169', '13171', '13173', '13175',
        '13177', '13179', '13181', '13183', '13185', '13187', '13189', '13191',
        '13193', '13195', '13197', '13199', '13201', '13203', '13205', '13207',
        '13209', '13211', '13213', '13215', '13217', '13219', '13221', '13223',
        '13225', '13227', '13229', '13231', '13233', '13235', '13237', '13239',
        '13241', '13243', '13245', '13247', '13249', '13251', '13253', '13255',
        '13257', '13259', '13261', '13263', '13265', '13267', '13269', '13271',
        '13273', '13275', '13277', '13279', '13281', '13283', '13285', '13287',
        '13289', '13291', '13293', '13295', '13297', '13299', '13301', '13303',
        '13305', '13307', '13309', '13311', '13313', '13315', '13317', '13319',
        '13321', '13510'
    ]
}

########################################################
#           helpers
########################################################


def convert_to_epiweek(x):
    return Week.fromstring(str(x))


def get_epiweeks_list(start_ew, end_ew):
    """
        returns list of epiweeks objects between start_ew and end_ew (inclusive)
        this is useful for iterating through these weeks
    """
    if type(start_ew) == str:
        start_ew = convert_to_epiweek(start_ew)
    if type(end_ew) == str:
        end_ew = convert_to_epiweek(end_ew)
    iter_weeks = list(Year(2017).iterweeks()) + list(Year(2018).iterweeks()) + list(Year(2019).iterweeks()) \
            + list(Year(2020).iterweeks()) + list(Year(2021).iterweeks())
    idx_start = iter_weeks.index(start_ew)
    idx_end = iter_weeks.index(end_ew)
    return iter_weeks[idx_start:idx_end + 1]


def create_window_seqs(
    X: np.array,
    y: np.array,
    min_sequence_length: int,
):
    """
        Creates windows of fixed size with appended zeros
        @param X: features
        @param y: targets, in synchrony with features (i.e. x[t] and y[t] correspond to the same time)
    """
    # convert to small sequences for training, starting with length 10
    seqs = []
    targets = []
    mask_ys = []

    # starts at sequence_length and goes until the end
    # for idx in range(min_sequence_length, X.shape[0]+1, 7): # last in range is step
    for idx in range(min_sequence_length, X.shape[0] + 1, 1):
        # Sequences
        seqs.append(torch.from_numpy(X[:idx, :]))
        # Targets
        y_ = y[:idx]
        mask_y = torch.ones(len(y_))
        targets.append(torch.from_numpy(y_))
        mask_ys.append(mask_y)
    seqs = pad_sequence(seqs, batch_first=True, padding_value=0).type(dtype)
    ys = pad_sequence(targets, batch_first=True,
                      padding_value=PAD_VALUE).type(dtype)
    mask_ys = pad_sequence(mask_ys, batch_first=True,
                           padding_value=0).type(dtype)

    return seqs, ys, mask_ys


########################################################
#           COVID: state/national level data
########################################################


def load_df(region, ew_start_data, ew_end_data):
    """ load and clean data"""
    df = pd.read_csv(datapath, low_memory=False)
    df = df[(df["region"] == region)]
    df['epiweek'] = df.loc[:, 'epiweek'].apply(convert_to_epiweek)
    # subset data using init parameters
    df = df[(df["epiweek"] <= ew_end_data) & (df["epiweek"] >= ew_start_data)]
    df = df.fillna(method="ffill")
    df = df.fillna(method="backfill")
    df = df.fillna(0)
    return df


def get_state_train_data(region, pred_week, ew_start_data=EW_START_DATA):
    """ get processed dataframe of data + target as array """
    # import data
    region = str.upper(region)
    pred_week = convert_to_epiweek(pred_week)
    ew_start_data = convert_to_epiweek(ew_start_data)
    df = load_df(region, ew_start_data, pred_week)
    # select targets
    # targets = df.loc[:,['positiveIncr','death_jhu_incidence']].values
    targets = df.loc[:, ['positiveIncr']].values
    # now subset based on input ew_start_data
    df = df[macro_features]
    return df, targets


def get_state_test_data(region, pred_week):
    """
        @ param pred_week: prediction week
    """
    pred_week = convert_to_epiweek(pred_week)
    # import smoothed dataframe
    df = load_df(region, pred_week + 1, pred_week + 4)
    new_cases = df.loc[:, 'positiveIncr'].values
    new_deaths = df.loc[:, 'death_jhu_incidence'].values
    return new_cases, new_deaths


def get_train_targets_all_regions(pred_week):
    deaths_all_regions = {}
    for region in states:
        _, targets = get_state_train_data(region, pred_week)
        deaths_all_regions[region] = targets[:, 1]  # index 1 is inc deaths
    return deaths_all_regions


def get_train_features_all_regions(pred_week):
    features_all_regions = {}
    for region in states:
        df, _ = get_state_train_data(region, pred_week)
        features_all_regions[region] = df.to_numpy()
    return features_all_regions


########################################################
#           COVID: county level data
# note: to obtain data, use get_features_per_county.ipynb
########################################################


def load_county_df(county, ew_start_data, ew_end_data):
    """ load and clean data"""
    df = pd.read_csv(county_datapath)
    df = df[(df["geo_value"] == int(county))]
    from datetime import datetime
    from datetime import date

    def convert_date_to_epiweek(x):
        date = datetime.strptime(x, '%Y-%m-%d')
        return Week.fromdate(date)

    df['epiweek'] = df.loc[:, 'time_value'].apply(convert_date_to_epiweek)
    # subset data using init parameters
    df = df[(df["epiweek"] <= ew_end_data) & (df["epiweek"] >= ew_start_data)]
    df = df.fillna(0)  # there are zeros at the beginning
    return df


def get_county_train_data(county,
                          pred_week,
                          ew_start_data=EW_START_DATA,
                          noise_level=0):
    # import data
    pred_week = convert_to_epiweek(pred_week)
    ew_start_data = convert_to_epiweek(ew_start_data)
    df = load_county_df(county, ew_start_data, pred_week)
    # select targets
    targets = df.loc[:, ['cases', 'deaths']].values
    if noise_level > 0:
        # noise_level is an index for your list
        noise = NOISE_LEVELS_COVID[noise_level - 1]
        std_vals = np.std(targets, axis=0) * noise
        noise_dist = np.random.normal(scale=std_vals, size=targets.shape)
        noisy_targets = targets + noise_dist
        noisy_targets = noisy_targets.astype('int32')
        targets = np.maximum(noisy_targets, 0)
    df.drop(columns=['epiweek', 'geo_value', 'time_value'], inplace=True)
    return df, targets


def get_county_test_data(county, pred_week):
    """
        @ param pred_week: prediction week
    """
    pred_week = convert_to_epiweek(pred_week)
    # import smoothed dataframe
    df = load_county_df(county, pred_week, pred_week + 4)
    new_cases = df.loc[:, 'cases'].values
    new_deaths = df.loc[:, 'deaths'].values
    return new_cases, new_deaths


########################################################
#           FLU: regional/state/national level data
########################################################


def load_df_flu(region, ew_start_data, ew_end_data, geo):
    """ load and clean data"""
    if geo == 'hhs':
        datapath = datapath_flu_hhs
    elif geo == 'state':
        datapath = datapath_flu_state
    df = pd.read_csv(datapath, low_memory=False)

    df = df[(df["region"] == region)]
    df['epiweek'] = df.loc[:, 'epiweek'].apply(convert_to_epiweek)
    # subset data using init parameters
    df = df[(df["epiweek"] <= ew_end_data) & (df["epiweek"] >= ew_start_data)]
    df = df.fillna(method="ffill")
    df = df.fillna(method="backfill")
    df = df.fillna(0)
    return df


def get_state_train_data_flu(region,
                             pred_week,
                             ew_start_data=EW_START_DATA_FLU,
                             geo='state',
                             noise_level=0):
    """ get processed dataframe of data + target as array """
    # import data
    region = str.upper(region)
    pred_week = convert_to_epiweek(pred_week)
    ew_start_data = convert_to_epiweek(ew_start_data)
    df = load_df_flu(region, ew_start_data, pred_week, geo)
    # select targets
    targets = pd.to_numeric(df['ili']).values.reshape(-1, 1)  # we need this 2d
    if noise_level > 0:
        # noise_level is an index for your list
        noise = NOISE_LEVELS_FLU[noise_level - 1]
        NOISE_STD = targets.std() * noise
        noise_dist = np.random.normal(loc=0,
                                      scale=NOISE_STD,
                                      size=targets.shape)
        noisy_targets = targets + noise_dist
        targets = np.array([max(ix, 0) for ix in noisy_targets])
    # now subset based on input ew_start_data
    df = df[["month"] + include_cols]
    return df, targets


def get_state_test_data_flu(region, pred_week, geo='state'):
    """
        @ param pred_week: prediction week
    """
    pred_week = convert_to_epiweek(pred_week)
    # import smoothed dataframe
    df = load_df_flu(region, pred_week + 1, pred_week + 4, geo)
    ili = df.loc[:, 'ili'].values
    return ili


def get_dir_from_path_list(path):
    outdir = path[0]
    if not (os.path.exists(outdir)):
        os.makedirs(outdir)
    for p in path[1:]:
        outdir = os.path.join(outdir, p)
        if not (os.path.exists(outdir)):
            os.makedirs(outdir)
    return outdir


if __name__ == "__main__":
    pass