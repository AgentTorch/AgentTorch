"""
Source: https://github.com/sneakatyou/Syspop/tree/NYC/syspop/process
"""

import numpy as np
import pandas as pd
from pandas import merge as pandas_merge
from pandas import DataFrame, concat
from numpy.random import randint as numpy_randint
from numpy.random import choice as numpy_choice
from numpy import NaN, isnan
import ray
import logging
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
from agent_torch.core.census.generate.address import add_random_address
import random
import sys

sys.path.append(
    "/Users/shashankkumar/Documents/GitHub/MacroEcon/simulator_data/step2/process/"
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compared_synpop_household_with_census(
    houshold_dataset: DataFrame, pop_input: DataFrame, proc_area: int
) -> dict:
    """Compared simulated household number with census

    Args:
        houshold_dataset (DataFrame): census household data
        pop_input (DataFrame): simulated population data with household information
        proc_area (int): area name

    Returns:
        dict: difference between simulation and census
    """

    def _get_household_children_num(household_data_result: DataFrame) -> dict:
        """Get the number of household against the number of children

        Args:
            household_data_result (DataFrame): Census household data

        Returns:
            dict: Census houshold information
        """
        household_data_result["household"] = household_data_result["household"].fillna(
            "default_9999_9999"
        )
        household_data_result["children_num"] = (
            household_data_result["household"].str.split("_").str[1].astype(int)
        )
        household_data_result["household"] = household_data_result["household"].replace(
            "default_9999_9999", NaN
        )
        household_data_result["children_num"] = household_data_result[
            "children_num"
        ].replace(9999, NaN)

        return household_data_result

    pop_input = _get_household_children_num(pop_input)

    orig_children_num = list(houshold_dataset.columns)
    orig_children_num.remove("area")
    all_possible_children_num = list(
        set(list(pop_input["children_num"].unique()) + orig_children_num)
    )

    truth_all_households = {}
    syspop_all_households = {}
    for pro_children_num in all_possible_children_num:
        if isnan(pro_children_num):
            continue

        try:
            truth_all_households[pro_children_num] = int(
                houshold_dataset[houshold_dataset["area"] == proc_area][
                    pro_children_num
                ].values[0]
            )
        except KeyError:
            truth_all_households[pro_children_num] = 0
        syspop_all_households[pro_children_num] = len(
            pop_input[pop_input["children_num"] == pro_children_num][
                "household"
            ].unique()
        )

    return {"truth": truth_all_households, "synpop": syspop_all_households}


def assign_any_remained_people(
    proc_base_pop: DataFrame, adults: DataFrame, children: DataFrame
) -> DataFrame:
    """Randomly assign remained people to existing household"""

    # Randomly assign remaining adults and children to existing households
    existing_households = proc_base_pop["household"].unique()
    existing_households = [
        x
        for x in existing_households
        if x != "NaN" and not (isinstance(x, float) and np.isnan(x))
    ]

    while len(adults) > 0:
        household_id = numpy_choice(existing_households)
        num_adults_to_add = numpy_randint(0, 3)

        if num_adults_to_add > len(adults):
            num_adults_to_add = len(adults)

        adult_ids = adults.sample(num_adults_to_add).index.tolist()
        proc_base_pop.loc[proc_base_pop.index.isin(adult_ids), "household"] = (
            household_id
        )
        adults = adults.loc[~adults.index.isin(adult_ids)]

    while len(children) > 0:
        household_id = numpy_choice(existing_households)
        num_children_to_add = numpy_randint(0, 3)

        if num_children_to_add > len(children):
            num_children_to_add = len(children)

        children_ids = children.sample(num_children_to_add).index.tolist()
        proc_base_pop.loc[proc_base_pop.index.isin(children_ids), "household"] = (
            household_id
        )
        children = children.loc[~children.index.isin(children_ids)]

    return proc_base_pop


def rename_household_id(df: DataFrame, proc_area: str, adult_list: list) -> DataFrame:
    """Rename household id from {id} to {adult_num}_{children_num}_{id}

    Args:
        df (DataFrame): base popualtion data

    Returns:
        DataFrame: updated population data
    """
    # Compute the number of adults and children in each household

    df["is_adult"] = df["age"].isin(adult_list)
    df["household"] = df["household"].astype(int)

    grouped = (
        df.groupby("household")["is_adult"]
        .agg(num_adults="sum", num_children=lambda x: len(x) - sum(x))
        .reset_index()
    )

    # Merge the counts back into the original DataFrame
    df = pandas_merge(df, grouped, on="household")

    # Create the new household_id column based on the specified format
    df["household_new_id"] = (
        f"{proc_area}_"
        + df["num_adults"].astype(str)
        + "_"
        + df["num_children"].astype(str)
        + "_"
        + df["household"].astype(str)
    )

    # Drop the temporary 'is_adult' column and other intermediate columns if needed
    return df.drop(["is_adult", "num_adults", "num_children"], axis=1)


@ray.remote
def create_household_composition_v3_remote(
    proc_houshold_dataset, proc_base_pop, proc_area, adult_list, children_list
):
    return create_household_composition_v3(
        proc_houshold_dataset, proc_base_pop, proc_area, adult_list, children_list
    )


def create_household_composition_v3(
    proc_houshold_dataset: DataFrame,
    proc_base_pop: DataFrame,
    proc_area: int or str,
    adult_list: list,
    children_list: list,
) -> DataFrame:
    """Create household composition (V3)

    Args:
        proc_houshold_dataset (DataFrame): Household dataset
        proc_base_pop (DataFrame): Base population dataset
        proc_area (intorstr): Area to use

    Returns:
        DataFrame: Updated population dataset
    """
    print("Starting create_household_composition_v3...")

    sorted_proc_houshold_dataset = proc_houshold_dataset.sort_values(
        by="household_num", ascending=False, inplace=False
    )

    household_types = proc_houshold_dataset[
        ["Family_households", "Nonfamily_households"]
    ]
    household_types["Family_households_prob"] = (
        household_types["Family_households"] / proc_houshold_dataset["household_num"]
    )
    household_types["Nonfamily_households_prob"] = (
        household_types["Nonfamily_households"] / proc_houshold_dataset["household_num"]
    )
    household_proportions = household_types[
        ["Family_households_prob", "Nonfamily_households_prob"]
    ]

    # Calculate average number of children per family
    avg_children_per_family = (
        proc_houshold_dataset["children_num"] / proc_houshold_dataset["household_num"]
    )  # Only for NZ data
    num_households = proc_houshold_dataset["household_num"].iloc[0]

    print(f"Number of households: {num_households}")

    unassigned_adults = proc_base_pop[proc_base_pop["age"].isin(adult_list)].copy()
    unassigned_children = proc_base_pop[proc_base_pop["age"].isin(children_list)].copy()
    household_types_choices = ["Family", "Nonfamily"]
    household_id = 0
    for _, row in sorted_proc_houshold_dataset.iterrows():
        for num in tqdm(range(num_households), desc="Processing"):
            household_type = random.choices(
                household_types_choices, weights=household_proportions.values.flatten()
            )[0]

            print(f"Household type: {household_type}")

            # Simulate family composition (if family household)
            if household_type == "Family":
                children_num = (
                    int(random.randint(0, 2) * avg_children_per_family)
                    if proc_houshold_dataset["children_num"].iloc[0] > 0
                    else 0
                )
            else:
                children_num = 0
            # Simulate number of individuals
            total_individuals = int(
                random.randint(0, 2)
                * proc_houshold_dataset["Average_household_size"].iloc[0]
            )
            if (total_individuals - children_num) <= 0:
                adults_num = total_individuals
                children_num = 0
            else:
                adults_num = total_individuals - children_num

            print(f"Number of adults: {adults_num}, Number of children: {children_num}")

            if (
                len(unassigned_adults) < adults_num
                or len(unassigned_children) < children_num
            ):
                print("Not enough adults or children to assign.")
                continue

            adult_ids = unassigned_adults.sample(adults_num)["index"].tolist()

            try:
                adult_majority_ethnicity = (
                    proc_base_pop.loc[proc_base_pop["index"].isin(adult_ids)][
                        "ethnicity"
                    ]
                    .mode()
                    .iloc[0]
                )
                children_ids = (
                    unassigned_children[
                        unassigned_children["ethnicity"] == adult_majority_ethnicity
                    ]
                    .sample(children_num)["index"]
                    .tolist()
                )
            except (
                ValueError,
                IndexError,
            ):
                print("Error occurred while sampling children by ethnicity.")
                # Value Error: not enough children for a particular ethnicity to be sampled from;
                # IndexError: len(adults_id) = 0 so mode() does not work
                children_ids = unassigned_children.sample(children_num)[
                    "index"
                ].tolist()

            print(f"Adult IDs: {adult_ids}")
            print(f"Children IDs: {children_ids}")

            # Update the household_id for the selected adults and children in the proc_base_pop DataFrame
            proc_base_pop.loc[proc_base_pop["index"].isin(adult_ids), "household"] = (
                f"{household_id}"
            )
            proc_base_pop.loc[
                proc_base_pop["index"].isin(children_ids), "household"
            ] = f"{household_id}"

            unassigned_adults = unassigned_adults.loc[
                ~unassigned_adults["index"].isin(adult_ids)
            ]
            unassigned_children = unassigned_children.loc[
                ~unassigned_children["index"].isin(children_ids)
            ]

            household_id += 1

    print("Assigning any remaining people...")
    proc_base_pop = assign_any_remained_people(
        proc_base_pop, unassigned_adults, unassigned_children
    )

    print("Renaming household IDs...")
    proc_base_pop = rename_household_id(proc_base_pop, proc_area, adult_list)

    print("Finished create_household_composition_v3.")

    return proc_base_pop


def household_wrapper(
    houshold_dataset: DataFrame,
    base_pop: DataFrame,
    adult_list: list,
    children_list: list,
    base_address: DataFrame,
    geo_address_data: DataFrame or None = None,
    use_parallel: bool = False,
    n_cpu: int = 8,
) -> DataFrame:
    """Assign people to different households

    Args:
        houshold_dataset (DataFrame): _description_
        base_pop (DataFrame): _description_
    """
    start_time = datetime.utcnow()
    if use_parallel:
        ray.init(num_cpus=n_cpu, ignore_reinit_error=True)

    base_pop["household"] = NaN

    all_areas = list(base_pop["area"].unique())
    total_areas = len(all_areas)

    results = []
    for i, proc_area in enumerate(all_areas):
        logger.info(f"{i}/{total_areas}: Processing {proc_area}")

        proc_base_pop = base_pop[base_pop["area"] == proc_area].reset_index()

        proc_houshold_dataset = household_prep(houshold_dataset, proc_base_pop)

        if len(proc_base_pop) == 0:
            continue

        if use_parallel:
            base_pop_results = ray.get(
                [
                    create_household_composition_v3_remote.remote(
                        proc_houshold_dataset,
                        proc_base_pop,
                        proc_area,
                        adult_list,
                        children_list,
                    )
                    for proc_area in all_areas
                ]
            )
            base_pop = pd.concat(base_pop_results, ignore_index=True)
            ray.shutdown()
            break
        else:
            result = create_household_composition_v3(
                proc_houshold_dataset,
                proc_base_pop,
                proc_area,
                adult_list,
                children_list,
            )

            results.append(result)

            try:
                for result in results:
                    result_index = result["index"]
                    result_content = result.drop("index", axis=1)
                    base_pop.iloc[result_index] = result_content
            except Exception as e:
                print(e)

    end_time = datetime.utcnow()

    total_mins = round((end_time - start_time).total_seconds() / 60.0, 3)
    logger.info(f"Processing time (household): {total_mins}")

    if geo_address_data is not None:
        proc_address_data = add_random_address(
            deepcopy(base_pop),
            geo_address_data,
            "household",
            use_parallel=False,
            n_cpu=n_cpu,
        )
        base_address = concat([base_address, proc_address_data])

    return base_pop, base_address


def obtain_household_adult_num(proc_household_data: DataFrame) -> DataFrame:
    """Obtain household adult number based on total people and children

    Args:
        proc_household_data (DataFrame): Household dataset

    Returns:
        DataFrame: Updated household
    """
    proc_household_data["adult_num"] = (
        proc_household_data["people_num"] - proc_household_data["children_num"]
    )

    proc_household_data["adult_num"] = proc_household_data["adult_num"].clip(lower=0)

    return proc_household_data


def get_household_scaling_factor(
    proc_base_synpop: DataFrame, proc_household_data: DataFrame
) -> float:
    """Get household scaling factor

    Args:
        proc_base_synpop (DataFrame): Base synthetic population
        proc_household_data (DataFrame): Base household dataset

    Returns:
        dict: Scaling factor for both adult and children
    """
    scaling_factor = (
        len(proc_base_synpop)
        / (
            proc_household_data["people_num"] * proc_household_data["household_num"]
        ).sum()
    )

    return scaling_factor


def household_prep(
    household_input: DataFrame, synpop_input: DataFrame, scaling: bool = False
) -> DataFrame:
    """Splitting and child and adults

    Args:
        household_input (DataFrame): Household data
        synpop_input (DataFrame): Synthetic population

    Returns:
        DataFrame: Updated household data
    """
    proc_area = list(synpop_input["area"].unique())[0]

    proc_household_data = household_input[household_input["area"] == proc_area]
    proc_base_synpop = synpop_input[synpop_input["area"] == proc_area]

    proc_household_data = obtain_household_adult_num(proc_household_data)

    if scaling:
        scaling_factor = get_household_scaling_factor(
            proc_base_synpop, proc_household_data
        )

        proc_household_data["household_num"] = (
            proc_household_data["household_num"] * scaling_factor
        )

        proc_household_data["household_num"] = proc_household_data[
            "household_num"
        ].apply(lambda x: max(1, round(x)))

    return proc_household_data

