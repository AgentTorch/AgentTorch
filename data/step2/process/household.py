'''
Source: https://github.com/sneakatyou/Syspop/tree/NYC/syspop/process
'''

from copy import deepcopy
from datetime import datetime
from logging import getLogger
from random import choices as random_choices
from random import sample as random_sample

import ray
from numpy import NaN, isnan
from numpy.random import choice as numpy_choice
from numpy.random import choice as random_choice
from numpy.random import randint
from numpy.random import randint as numpy_randint
from pandas import DataFrame, concat, isna
from pandas import merge as pandas_merge
from process.address import add_random_address

logger = getLogger()

def add_people(
    pop_input: DataFrame,
    total_households: int,
    adult_list: list,
    map_age_to_range,
    proc_num_children: int,
    proc_area: int,
    all_ethnicities: list,
    parents_age_limits={"min": 18, "max": 65},
    single_parent: bool = False,
) -> DataFrame:
    """Add people to household

    Args:
        pop_input (DataFrame): Base population
        total_households (int): Total households to be assigned
        proc_num_children (int): Number of children to be assigned
        proc_area (int): Area name
        all_ethnicities (list): All ethnicities
        parents_age_limits (dict, optional): Parents age range.
            Defaults to {"min": 18, "max": 65}.

    Returns:
        Dataframe: Updated population information
    """

    def _select_female_ethnicity(target_ethnicity: str, input_ethnicity: list) -> str:
        """Select female ethnicity based on male

        Args:
            target_ethnicity (str): The male ethnicity
            input_ethnicity (list): potential ethnicities to be chosen from

        Returns:
            str: randomly selected ethnicity
        """

        input_ethnicity.remove(target_ethnicity)

        individual_percetnage = 0.3 / len(input_ethnicity)

        all_ethnicities = [target_ethnicity] + input_ethnicity
        weights = [0.7] + [individual_percetnage] * len(input_ethnicity)

        return random_choices(all_ethnicities, weights, k=1)[0]

    for proc_household in range(total_households):
        proc_household_id = f"{proc_area}_{proc_num_children}_{proc_household}"

        if single_parent:
            proc_household_id += "_sp"

            selected_single_parent = pop_input[
                (pop_input["age"] in adult_list)
                & isna(pop_input["household"])
            ]

            if len(selected_single_parent) == 0:
                continue

            selected_single_parent = selected_single_parent.sample(n=1)
            selected_single_parent["household"] = proc_household_id
            pop_input.loc[selected_single_parent.index] = selected_single_parent
            selected_children_ethnicity = selected_single_parent["ethnicity"].values[0]
        else:
            selected_parents_male = pop_input[
                (pop_input["gender"] == "male")
                & (pop_input["age"] in adult_list)
                & isna(pop_input["household"])
            ]

            if len(selected_parents_male) == 0:
                continue

            selected_parents_male = selected_parents_male.sample(n=1)
            selected_parents_male["household"] = proc_household_id
            pop_input.loc[selected_parents_male.index] = selected_parents_male

            selected_parents_female_ethnicity = _select_female_ethnicity(
                selected_parents_male["ethnicity"].values[0], deepcopy(all_ethnicities)
            )
            selected_parents_female = pop_input[
                (pop_input["gender"] == "female")
                & (pop_input["ethnicity"] == selected_parents_female_ethnicity)
                & (
                    pop_input["age"]
                    == map_age_to_range(max(
                        0.7 * selected_parents_male["age"].values[0],
                        parents_age_limits["min"],
                    )
                ))
                & (
                    pop_input["age"]
                    == map_age_to_range(min(
                        1.3 * selected_parents_male["age"].values[0],
                        parents_age_limits["max"],
                    ))
                )
                & isna(pop_input["household"])
            ]

            if len(selected_parents_female) == 0:
                continue

            selected_parents_female = selected_parents_female.sample(n=1)
            selected_parents_female["household"] = proc_household_id
            pop_input.loc[selected_parents_female.index] = selected_parents_female

            selected_children_ethnicity = selected_parents_female_ethnicity

        selected_children = pop_input[
            (pop_input["age"] not in adult_list)
            & (pop_input["ethnicity"] == selected_children_ethnicity)
            & isna(pop_input["household"])
        ]

        if len(selected_children) < proc_num_children:
            continue

        selected_children = selected_children.sample(n=proc_num_children)
        selected_children["household"] = proc_household_id
        pop_input.loc[selected_children.index] = selected_children

    return pop_input


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


def randomly_assign_people_to_household_v2(
    proc_base_pop: DataFrame,
    adult_list: list,
    proc_area: str,
    target_household_num: dict = None,
    household_size={"min": 1, "max": 10},
) -> DataFrame:
    # target_household_num = {0: 300, 1: 50, 2: 110, 3: 40, 4: 15}
    unassigned_people = proc_base_pop[proc_base_pop["household"].isna()]

    unassigned_adults = len(unassigned_people[unassigned_people["age"] in adult_list])
    unassigned_children = len(unassigned_people[unassigned_people["age"] not in adult_list])

    target_children_num = 0
    for children_num in target_household_num:
        target_children_num += children_num * target_household_num[children_num]
    adjuste_factor = unassigned_children / target_children_num

    for children_num in target_household_num:
        target_household_num[children_num] *= adjuste_factor

    x = 3


def randomly_assign_people_to_household(
    proc_base_pop: DataFrame,
    proc_area: str,
    adult_list: list,
    household_num: dict = None,
    household_size={"min": 1, "max": 10},
) -> DataFrame:
    """Randomly assign people to household with size defined in household_size

    Args:
        proc_base_pop (DataFrame): base population
        proc_area (str): the area name
        household_size (dict, optional): household size to be used.
            Defaults to {"min": 1, "max": 10}.

    Returns:
        Dataframe: Updated population
    """
    unassigned_people = proc_base_pop[isna(proc_base_pop["household"])]
    index_unassigned = 0
    index_unassigned_noadult = 0
    while len(unassigned_people) > 0:  # up to 10 people in a household
        # Randomly select x rows
        sample_size = randint(household_size["min"], household_size["max"])

        try:
            selected_rows = unassigned_people.sample(n=sample_size)
        except ValueError:
            selected_rows = unassigned_people.sample(n=len(unassigned_people))

        # Check if there is a row with age < 18
        if (selected_rows["age"] not in adult_list).any():
            # If there is, check if there is also a row with age > 18
            if (selected_rows["age"] in adult_list).any():
                # If there is, assign the label to these rows
                mask = selected_rows["age"] not in adult_list
                children_num = len(selected_rows[mask])
                proc_base_pop.loc[
                    selected_rows.index, "household"
                ] = f"{proc_area}_{children_num}_random{index_unassigned}"
                no_adult_family_tries = 0
            else:
                # If there isn't a row with age > 18, put the rows back and try again
                try:
                    no_adult_family_tries += 1
                except NameError:
                    no_adult_family_tries = 0

                if (
                    no_adult_family_tries == 5
                ):  # if we are not able to find adults any more ...
                    children_num = len(selected_rows)
                    proc_base_pop.loc[
                        selected_rows.index, "household"
                    ] = f"{proc_area}_{children_num}_noadult{index_unassigned_noadult}"
                    no_adult_family_tries = 0
                    index_unassigned_noadult += 1
                else:
                    continue
        else:
            # If there isn't a row with age < 18, assign the label to these rows
            proc_base_pop.loc[
                selected_rows.index, "household"
            ] = f"{proc_area}_0_random{index_unassigned}"
            no_adult_family_tries = 0

        unassigned_people = proc_base_pop[isna(proc_base_pop["household"])]
        index_unassigned += 1

    proc_base_pop = send_remained_children_to_household(proc_base_pop)

    return proc_base_pop


def send_remained_children_to_household(proc_base_pop: DataFrame) -> DataFrame:
    """Send remained children (those children in a household without adults)
    to different households

    Args:
        proc_base_pop (DataFrame): Base population

    Returns:
        DataFrame: Updated population
    """

    def _add_one_child_to_name(input_name: str) -> str:
        """Add one child to the household name

        Args:
            input_name (str): Input name such as 110500_3_random132

        Returns:
            str: such as 110500_4_random132
        """
        input_name = input_name.split("_")
        input_name[1] = str(int(input_name[1]) + 1)
        return "_".join(input_name)

    children_only_data = proc_base_pop[
        proc_base_pop["household"].str.contains("_noadult.*$")
    ]
    ramdom_household_data = proc_base_pop[
        proc_base_pop["household"].str.contains("_random.*$", regex=True)
    ]

    # If no random household created
    if len(ramdom_household_data) == 0:
        return proc_base_pop

    for i in range(len(children_only_data)):
        proc_child = children_only_data.iloc[[i]]
        selected_household = ramdom_household_data.sample(n=1)

        proc_base_pop.loc[proc_child.index, "household"] = _add_one_child_to_name(
            selected_household["household"].values[0]
        )

    return proc_base_pop


@ray.remote
def create_household_composition_remote(
    houshold_dataset: DataFrame,
    proc_base_pop: DataFrame,
    adult_list: list,
    map_age_to_range,
    num_children: list,
    all_ethnicities: list,
    proc_area: str,
) -> DataFrame:
    return create_household_composition(
        houshold_dataset, proc_base_pop, adult_list, map_age_to_range, num_children, all_ethnicities, proc_area
    )


def update_household_name(proc_base_pop: DataFrame, adult_list) -> DataFrame:
    """Update household name from {area}_{id} to {area}_{child num}_{id}

    Args:
        proc_base_pop (DataFrame): population to be used

    Returns:
        DataFrame: updated population
    """
    proc_base_pop["is_child"] = proc_base_pop["age"].apply(
        lambda x: 1 if x not in adult_list else 0
    )
    proc_base_pop["num_children"] = (
        proc_base_pop.groupby("household")["is_child"]
        .transform("sum")
        .fillna(0)
        .astype(int)
    )

    # Split the 'household' column
    proc_base_pop[["area_tmp", "id_tmp"]] = proc_base_pop["household"].str.split(
        "_", expand=True
    )

    # Combine 'area', 'num_children', and 'id' to form the new 'household'
    proc_base_pop["household"] = (
        proc_base_pop["area_tmp"]
        + "_"
        + proc_base_pop["num_children"].astype(str)
        + "_"
        + proc_base_pop["id_tmp"]
    )

    return proc_base_pop.drop(
        columns=["is_child", "num_children", "area_tmp", "id_tmp"]
    )


def create_household_composition_v2(
    houshold_dataset: DataFrame,
    proc_base_pop: DataFrame,
    proc_area: str,
    adult_list: list
) -> DataFrame:
    """Create household composition, e.g., based on the number of children

    Args:
        houshold_dataset (DataFrame): Household dataset
        proc_base_pop (DataFrame): Synthetic population
        proc_area (str): Area to be used

    Returns:
        DataFrame: Updated synthetic population
    """
    # ---------------------------------
    # Step 1: all possible children number (e.g., 0, 1, 2, etc)
    # and the total number of househodls
    # ---------------------------------
    all_possible_children_num = [
        col for col in houshold_dataset.columns if col not in ["area"]
    ]

    total_households_num = houshold_dataset[all_possible_children_num].sum().sum()

    # ---------------------------------
    # Step 2: extract adults and children from the population
    # ---------------------------------
    proc_base_pop_adults = proc_base_pop[proc_base_pop["age"] in adult_list]
    proc_base_pop_children = proc_base_pop[proc_base_pop["age"] not in adult_list]

    # ---------------------------------
    # Step 3: assign households ID randomly to adults
    # ---------------------------------
    proc_base_pop_adults["household"] = numpy_randint(
        1, total_households_num + 1, size=len(proc_base_pop_adults)
    )
    proc_base_pop_adults["household"] = f"{proc_area}_" + proc_base_pop_adults[
        "household"
    ].astype(str)

    all_households = list(proc_base_pop_adults["household"].unique())

    # ---------------------------------
    # Step 4: assign children to households
    # ---------------------------------
    for proc_children_num in all_possible_children_num:
        if proc_children_num == 0:
            continue
        # --------------------------------
        # Step 4.1: the children has not been assigned with a household
        # --------------------------------
        households_with_children = proc_base_pop_children[
            proc_base_pop_children["household"].notna()
        ]["household"]

        # --------------------------------
        # Step 4.2: the household IDs has not been assigned
        # --------------------------------
        remained_households = [
            item
            for item in all_households
            if item not in list(households_with_children)
        ]

        # --------------------------------
        # Step 4.3: the household we need for this children number
        # --------------------------------
        proc_household_num = houshold_dataset[proc_children_num].values[0]

        # --------------------------------
        # Step 4.4: randomly select a range of householeds ID from the pool
        # --------------------------------
        selected_households = random_sample(remained_households, proc_household_num)

        for _ in range(proc_children_num):
            subset_rows = random_choice(
                proc_base_pop_children[
                    proc_base_pop_children["household"].isna()
                ].index,
                size=proc_household_num,
                replace=False,
            )

            proc_base_pop_children.loc[subset_rows, "household"] = random_choice(
                selected_households, size=len(subset_rows)
            )

    proc_base_pop.loc[proc_base_pop_adults.index] = proc_base_pop_adults
    proc_base_pop.loc[proc_base_pop_children.index] = proc_base_pop_children

    # ---------------------------------
    # Step 5: assign the remained people
    # ---------------------------------
    remained_people = proc_base_pop[proc_base_pop["household"].isna()]
    for proc_index in remained_people.index:
        remained_people.loc[proc_index, "household"] = random_choice(all_households, 1)[
            0
        ]

    proc_base_pop.loc[remained_people.index] = remained_people

    return update_household_name(proc_base_pop,adult_list)


def create_household_composition(
    houshold_dataset: DataFrame,
    proc_base_pop: DataFrame,
    adult_list: list,
    map_age_to_range,
    num_children: list,
    all_ethnicities: list,
    proc_area: str,
    household_adults_ratio: dict = {2: 0.5, 1: 0.2, "others": 0.3},
) -> DataFrame:
    """Create household composistion using 3 steps:
        - step 1: two parents family (following census)
        - step 2: single parent family (for remaining people)
        - step 3: randomly assign rest people to families (for remaining people)

    Args:
        houshold_dataset (DataFrame): _description_
        num_children (list): _description_
        all_ethnicities (list): _description_
        proc_area (str): _description_

    Returns:
        DataFrame: _description_
    """
    # Step 1: First round assignment (two parents)
    logger.info("Start step 1 ....")
    for proc_num_children in num_children:
        total_households = int(
            int(
                houshold_dataset[houshold_dataset["area"] == proc_area][
                    proc_num_children
                ].values[0]
            )
            * household_adults_ratio[2]
        )

        proc_base_pop = add_people(
            deepcopy(proc_base_pop),
            total_households,
            adult_list,
            map_age_to_range,
            proc_num_children,
            proc_area,
            all_ethnicities,
        )

    # synpop_validation_data_after_step1 = compared_synpop_household_with_census(
    #    houshold_dataset, proc_base_pop, proc_area
    # )

    logger.info("Start step 2 ....")
    # Step 2: Second round assignment (single parent)
    for proc_num_children in num_children:
        # total_households = (
        #    synpop_validation_data_after_step1["truth"][proc_num_children]
        #    - synpop_validation_data_after_step1["synpop"][proc_num_children]
        # )

        # if total_households <= 0:
        #    continue

        total_households = int(
            int(
                houshold_dataset[houshold_dataset["area"] == proc_area][
                    proc_num_children
                ].values[0]
            )
            * household_adults_ratio[1]
        )

        proc_base_pop = add_people(
            proc_base_pop,
            total_households,
            adult_list,
            map_age_to_range,
            proc_num_children,
            proc_area,
            all_ethnicities,
            single_parent=True,
        )

    logger.info("Start step 3 ....")
    # Step 3: randomly assigned the rest people
    synpop_validation_data = compared_synpop_household_with_census(
        houshold_dataset, proc_base_pop, proc_area
    )

    target_household_data = {}

    for children_num in synpop_validation_data["truth"].keys():
        target_household_data[children_num] = (
            synpop_validation_data["truth"][children_num]
            - synpop_validation_data["synpop"][children_num]
        )

    logger.info("Start step 2 ....")
    # Step 3: 3rd round assignment (random number of adults)
    for proc_num_children in num_children:
        # proc_base_pop = randomly_assign_people_to_household(proc_base_pop, proc_area)
        proc_base_pop = randomly_assign_people_to_household_v2(
            proc_base_pop, adult_list,proc_area, target_household_data
        )

    proc_base_pop.drop("children_num", axis=1, inplace=True)

    return proc_base_pop


def assign_any_remained_people(
    proc_base_pop: DataFrame, adults: DataFrame, children: DataFrame
) -> DataFrame:
    """Randomly assign remained people to existing household"""

    # Randomly assign remaining adults and children to existing households
    existing_households = proc_base_pop["household"].unique()
    existing_households = [
        x
        for x in existing_households
        if x != "NaN" and not (isinstance(x, float) and isnan(x))
    ]

    while len(adults) > 0:
        household_id = numpy_choice(existing_households)
        num_adults_to_add = numpy_randint(0, 3)

        if num_adults_to_add > len(adults):
            num_adults_to_add = len(adults)

        adult_ids = adults.sample(num_adults_to_add).index.tolist()
        proc_base_pop.loc[
            proc_base_pop.index.isin(adult_ids), "household"
        ] = household_id
        adults = adults.loc[~adults.index.isin(adult_ids)]

    while len(children) > 0:
        household_id = numpy_choice(existing_households)
        num_children_to_add = numpy_randint(0, 3)

        if num_children_to_add > len(children):
            num_children_to_add = len(children)

        children_ids = children.sample(num_children_to_add).index.tolist()
        proc_base_pop.loc[
            proc_base_pop.index.isin(children_ids), "household"
        ] = household_id
        children = children.loc[~children.index.isin(children_ids)]

    return proc_base_pop


def rename_household_id(df: DataFrame, proc_area: str,adult_list: list) -> DataFrame:
    """Rename household id from {id} to {adult_num}_{children_num}_{id}

    Args:
        df (DataFrame): base popualtion data

    Returns:
        DataFrame: updated population data
    """
    # Compute the number of adults and children in each household

    df["is_adult"] = df["age"] not in adult_list
    df["household"] = df["household"].astype(int)

    grouped = (
        df.groupby("household")["is_adult"]
        .agg(num_adults="sum", num_children=lambda x: len(x) - sum(x))
        .reset_index()
    )

    # Merge the counts back into the original DataFrame
    df = pandas_merge(df, grouped, on="household")

    # Create the new household_id column based on the specified format
    df["household"] = (
        f"{proc_area}_"
        + df["num_adults"].astype(str)
        + "_"
        + df["num_children"].astype(str)
        + "_"
        + df["household"].astype(str)
    )

    # Drop the temporary 'is_adult' column and other intermediate columns if needed
    return df.drop(["is_adult", "num_adults", "num_children"], axis=1)


def create_household_composition_v3(
    proc_houshold_dataset: DataFrame, proc_base_pop: DataFrame, proc_area: int or str, adult_list: list
) -> DataFrame:
    """Create household composition (V3)

    Args:
        proc_houshold_dataset (DataFrame): Household dataset
        proc_base_pop (DataFrame): Base population dataset
        proc_area (intorstr): Area to use

    Returns:
        DataFrame: Updated population dataset
    """
    sorted_proc_houshold_dataset = proc_houshold_dataset.sort_values(
        by="household_num", ascending=False, inplace=False
    )

    unassigned_adults = proc_base_pop[proc_base_pop["age"] in adult_list].copy()
    unassigned_children = proc_base_pop[proc_base_pop["age"] not in adult_list].copy()

    household_id = 0
    for _, row in sorted_proc_houshold_dataset.iterrows():
        for _ in range(row["household_num"]):
            if (
                len(unassigned_adults) < row["adult_num"]
                or len(unassigned_children) < row["children_num"]
            ):
                print("Not enough adults or children to assign.")
                continue

            adult_ids = unassigned_adults.sample(row["adult_num"])["index"].tolist()

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
                    .sample(row["children_num"])["index"]
                    .tolist()
                )
            except (
                ValueError,
                IndexError,
            ):
                # Value Error: not enough children for a particular ethnicity to be sampled from;
                # IndexError: len(adults_id) = 0 so mode() does not work
                children_ids = unassigned_children.sample(row["children_num"])[
                    "index"
                ].tolist()

            # Update the household_id for the selected adults and children in the proc_base_pop DataFrame
            proc_base_pop.loc[
                proc_base_pop["index"].isin(adult_ids), "household"
            ] = f"{household_id}"
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

    proc_base_pop = assign_any_remained_people(
        proc_base_pop, unassigned_adults, unassigned_children
    )

    return rename_household_id(proc_base_pop, proc_area,adult_list)


def household_wrapper(
    houshold_dataset: DataFrame,
    base_pop: DataFrame,
    adult_list: list,
    map_age_to_range,
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

    base_pop["household"] = NaN

    num_children = list(houshold_dataset.columns)
    num_children.remove("area")

    all_areas = list(base_pop["area"].unique())
    total_areas = len(all_areas)
    results = []

    for i, proc_area in enumerate(all_areas):
        logger.info(f"{i}/{total_areas}: Processing {proc_area}")

        proc_base_pop = base_pop[base_pop["area"] == proc_area].reset_index()

        proc_houshold_dataset = household_prep(houshold_dataset, proc_base_pop)

        if len(proc_base_pop) == 0:
            continue

        proc_base_pop = create_household_composition_v3(
            proc_houshold_dataset, proc_base_pop, proc_area,adult_list
        )

        results.append(proc_base_pop)

    for result in results:
        result_index = result["index"]
        result_content = result.drop("index", axis=1)
        base_pop.iloc[result_index] = result_content

    base_pop[["area", "age"]] = base_pop[["area", "age"]].astype(int)
    end_time = datetime.utcnow()

    total_mins = round((end_time - start_time).total_seconds() / 60.0, 3)
    logger.info(f"Processing time (household): {total_mins}")

    if geo_address_data is not None:
        proc_address_data = add_random_address(
            deepcopy(base_pop),
            geo_address_data,
            "household",
            use_parallel=use_parallel,
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

    proc_household_data = proc_household_data[
        ["area", "adult_num", "children_num", "household_num"]
    ]

    return proc_household_data
