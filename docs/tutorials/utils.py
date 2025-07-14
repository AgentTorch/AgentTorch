import pandas as pd
import torch


def get_covid_cases_data(csv_path, county_name):
    cases_data = pd.read_csv(
        "/Users/shashankkumar/Documents/GitHub/MacroEcon/models/covid/data/county_data.csv"
    )
    # print(cases_data['county'].unique())
    cases_data = cases_data[cases_data["county"] == county_name].sort_values("date")
    cases_data["date"] = pd.to_datetime(cases_data["date"])
    cases_data["year"] = cases_data["date"].dt.year
    cases_data["month"] = cases_data["date"].dt.month - 1
    monthly_cases = (
        cases_data.groupby(["year", "month"])["cases_week"].sum().reset_index()
    )
    monthly_cases["year"] = (
        monthly_cases["year"].astype(int) - monthly_cases["year"].astype(int).min() + 2
    )
    monthly_cases["cases_month"] = monthly_cases["cases_week"]
    monthly_cases = monthly_cases.drop(columns=["cases_week"])
    return monthly_cases[-17:]


def get_labor_data(read_path, monthly_cases):
    labor_data = pd.read_csv(read_path)
    labor_data = labor_data.rename(
        columns={
            "Revised 2019-2023 Labor Force Data": "area",
            "Unnamed: 1": "year",
            "Unnamed: 2": "month",
            "Unnamed: 3": "labor_force",
            "Unnamed: 4": "employed",
            "Unnamed: 5": "unemployed",
            "Unnamed: 6": "unemployment_rate",
        }
    )
    labor_data = labor_data.drop(labor_data.index[:2])
    labor_data = labor_data.dropna()
    labor_data.reset_index(drop=True, inplace=True)
    labor_data["labor_force"] = labor_data["labor_force"].str.replace("\t", "")
    labor_data["labor_force"] = labor_data["labor_force"].str.replace(",", "")
    labor_data["labor_force"] = labor_data["labor_force"].astype(int)
    month_to_index = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "Jun": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11,
        "Avg": 12,
    }
    labor_data["month"] = labor_data["month"].map(month_to_index)
    labor_data["year"] = (
        labor_data["year"].astype(int) - labor_data["year"].astype(int).min()
    )
    # labor_data['area'] = labor_data['area'].replace('Kings County', 1)
    labor_data.sort_values(by=["year", "month"], inplace=True)
    labor_data = labor_data[labor_data["year"] != 0]
    merged_data = monthly_cases.merge(labor_data, on=["year", "month"])
    merged_data["labor_force_pct_change"] = (
        merged_data["labor_force"].pct_change() * 100
    )
    return merged_data[-17:]


def normalize_data(data):
    return 2 * ((data - data.min()) / (data.max() - data.min())) - 1


def update_kwargs(monthly_cases, kwargs):
    cases = monthly_cases["cases_month"].values
    month_to_index = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "Jun": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11,
        "Avg": 12,
    }
    index_to_month = {v: k for k, v in month_to_index.items()}
    index_to_year = {1: 2020, 2: 2021, 3: 2022, 4: 2023}

    for en, case in enumerate(cases):
        kwargs["covid_cases"] = case * 100
        kwargs["month"] = index_to_month[monthly_cases["month"].values[en]]
        kwargs["year"] = index_to_year[monthly_cases["year"].values[en]]
        yield kwargs


def get_labor_force_correlation(monthly_cases, earning_behavior, data_path, inp_kwargs):
    labor_force_list = []
    for kwargs in update_kwargs(monthly_cases, inp_kwargs):
        print(
            "Month:",
            kwargs["month"],
            "Year:",
            kwargs["year"],
            "Cases:",
            kwargs["covid_cases"],
        )
        output_behavior = earning_behavior.sample(kwargs)
        labor_force = torch.bernoulli(output_behavior).sum().item()
        labor_force_list.append(labor_force)

    labor_force_list_df = pd.DataFrame(labor_force_list, columns=["Labor Force"])
    labor_force_list_df["Pct Change"] = labor_force_list_df["Labor Force"].pct_change()

    observed_labor_force = get_labor_data(data_path, monthly_cases)
    correlation_value = observed_labor_force["labor_force_pct_change"].corr(
        labor_force_list_df["Pct Change"]
    )
    print("Correlation Value:", correlation_value)
    return labor_force_list_df, observed_labor_force, correlation_value
