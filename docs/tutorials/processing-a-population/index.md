# Tutorial: Generating Base Population and Household Data

This tutorial will guide you through the process of generating base population
and household data for a specified region using census data. We’ll use a
`CensusDataLoader` class to handle the data processing and generation.

## Before Starting

Make sure your `population data` and `household data` are in the prescribed
format. Names of the column need to be same as shown in the excerpts.

Lets see a snapshot of the data

`Population Data` is a dictionary containing two pandas DataFrames:
'`age_gender`' and '`ethnicity`'. Each DataFrame provides demographic
information for different areas and regions.

The `age_gender` DataFrame provides a comprehensive breakdown of population
data, categorized by area, gender, and age group.

#### Columns Description

- `area`: Serves as a unique identifier for each geographical area, represented
  by a string (e.g., `'BK0101'`, `'SI9593'`).
- `gender`: Indicates the gender of the population segment, with possible values
  being `'female'` or `'male'`.
- `age`: Specifies the age group of the population segment, using a string
  format such as `'20t29'` for ages 20 to 29, and `'U19'` for those under 19
  years of age.
- `count`: Represents the total number of individuals within the specified
  gender and age group for a given area.
- `region`: A two-letter code that identifies the broader region encompassing
  the area (e.g., `'BK'` for Brooklyn, `'SI'` for Staten Island).

##### Example Entry

Here is a sample of the data structure within the `age_gender` DataFrame:

| area   | gender | age   | count | region |
| ------ | ------ | ----- | ----- | ------ |
| BK0101 | female | 20t29 | 3396  | BK     |
| BK0101 | male   | 20t29 | 3327  | BK     |

This example entry demonstrates the DataFrame's layout and the type of
demographic data it contains, highlighting its utility for detailed population
studies by age and gender.

The `ethnicity` DataFrame is structured to provide detailed population data,
segmented by both geographical areas and ethnic groups.

##### Columns Description

- `area`: A unique identifier assigned to each area, formatted as a string
  (e.g., `'BK0101'`, `'SI9593'`). This identifier helps in pinpointing specific
  locations within the dataset.
- `ethnicity`: Represents the ethnic group of the population in the specified
  area.
- `count`: Indicates the number of individuals belonging to the specified ethnic
  group within the area. This is an integer value representing the population
  count.
- `region`: A two-letter code that signifies the broader region that the area
  belongs to (e.g., `'BK'` for Brooklyn, `'SI'` for Staten Island).

##### Example Entry

Below is an example of how the data is presented within the DataFrame:

| area   | ethnicity | count | region |
| ------ | --------- | ----- | ------ |
| BK0101 | asian     | 1464  | BK     |
| BK0101 | black     | 937   | BK     |

This example illustrates the structure and type of data contained within the
`ethnicity` DataFrame, showcasing its potential for detailed demographic
studies.

`Household Data` contains the following columns:

- `area`: Represents a unique identifier for each area.
- `people_num`: The total number of people within the area.
- `children_num`: The number of children in the area.
- `household_num`: The total number of households.
- `family_households`: Indicates the number of households identified as family
  households, highlighting family-based living arrangements.
- `nonfamily_households`: Represents the number of households that do not fall
  under the family households category, including single occupancy and unrelated
  individuals living together.
- `average_household_size`: The average number of individuals per household.

Below is a sample excerpt:

| area   | people_num | children_num | household_num | family_households | nonfamily_households | average_household_size |
| ------ | ---------- | ------------ | ------------- | ----------------- | -------------------- | ---------------------- |
| 100100 | 104        | 56           | 418           | 1                 | 0                    | 2.488038               |
| 100200 | 132        | 73           | 549           | 1                 | 0                    | 2.404372               |
| 100300 | 5          | 0            | 10            | 0                 | 1                    | 5.000000               |

Now that we have verified our input, we can proceed to next steps!

## Step 1: Set Up File Paths

First, we need to specify the paths to our data files.

Make sure to replace the placeholder paths with the actual paths to your data
files.

```python
# Path to the population data file. Update with the actual file path.
POPULATION_DATA_PATH = "docs/tutorials/processing-a-population/sample_data/NYC/population.pkl"

# Path to the household data file. Update with the actual file path.
HOUSEHOLD_DATA_PATH = "docs/tutorials/processing-a-population/sample_data/NYC/household.pkl"
```

## Step 2: Define Age Group Mapping

We’ll define a mapping for age groups to categorize adults and children in the
household data:

```python
AGE_GROUP_MAPPING = {
    "adult_list": ["20t29", "30t39", "40t49", "50t64", "65A"],  # Age ranges for adults
    "children_list": ["U19"],  # Age range for children
}
```

## Step 3: Load Data

Now, let’s load the population and household data:

```python
import numpy as np
import pandas as pd

# Load household data
HOUSEHOLD_DATA = pd.read_pickle(HOUSEHOLD_DATA_PATH)

# Load population data
BASE_POPULATION_DATA = pd.read_pickle(POPULATION_DATA_PATH)
```

## Step 4: Set Up Additional Parameters

We’ll set up some additional parameters that might be needed for data
processing. These are not essential for generating population, but still good to
know if you decide to use them in future.

```python
# Placeholder for area selection criteria, if any. Update or use as needed.
# Example: area_selector = ["area1", "area2"]
# This will be used to filter the population data to only include the selected areas.
area_selector = None

# Placeholder for geographic mapping data, if any. Update or use as needed.
geo_mapping = None
```

## Step 5: Initialize the Census Data Loader

Create an instance of the `CensusDataLoader` class:

```python
from agent_torch.data.census.census_loader import CensusDataLoader

census_data_loader = CensusDataLoader(n_cpu=8, use_parallel=True)
```

This initializes the loader with 8 CPUs and enables parallel processing for
faster data generation.

## Step 6: Generate Base Population Data

Generate the base population data for a specified region:

```python
census_data_loader.generate_basepop(
    input_data=BASE_POPULATION_DATA,  # The population data frame
    region="astoria",  # The target region for generating base population
    area_selector=area_selector,  # Area selection criteria, if applicable
)
```

This will create a base population of 100 individuals for the “astoria” region.
The generated data will be exported to a folder named “astoria” under the
“populations” folder.

#### Overview of the Generated Base Population Data

Each row corresponds to attributes of individual residing in the specified
region while generating the population.

| area   | age   | gender | ethnicity | region |
| ------ | ----- | ------ | --------- | ------ |
| BK0101 | 20t29 | female | black     | BK     |
| BK0101 | 20t29 | female | hispanic  | BK     |
| ...    | ...   | ...    | ...       | ...    |
| BK0101 | U19   | male   | asian     | SI     |
| BK0101 | U19   | female | white     | SI     |
| BK0101 | U19   | male   | asian     | SI     |

## Step 7: Generate Household Data

Finally, generate the household data for the specified region:

```python
census_data_loader.generate_household(
    household_data=HOUSEHOLD_DATA,  # The loaded household data
    household_mapping=AGE_GROUP_MAPPING,  # Mapping of age groups for household composition
    region="astoria"  # The target region for generating households
)
```

This will create household data for the “astoria” region based on the previously
generated base population. The generated data will be exported to the same
“astoria” folder under the “populations” folder.

## Bonus: Generate Population Data of Specific Size

For quick experimentation, this may come in handy.

```python
census_data_loader.generate_basepop(
    input_data=BASE_POPULATION_DATA,  # The population data frame
    region="astoria",  # The target region for generating base population
    area_selector=area_selector,  # Area selection criteria, if applicable
    num_individuals = 100 # Saves data for first 100 individuals, from the generated population
)
```

## Bonus: Export Population Data

If you have already generated your synthetic population, you just need to export
it to "populations" folder under the desired "region", in order for you to use
it with AgentTorch.

```python
POPULATION_DATA_PATH = "/population_data.pickle"  # Replace with actual path
census_data_loader.export(population_data_path=POPULATION_DATA_PATH,region="astoria")
```

In case you want to export data for only few individuals

```python
census_data_loader.export(population_data_path=POPULATION_DATA_PATH,region="astoria",num_individuals = 100)
```

## Conclusion

You have now successfully generated both base population and household data for
the `“astoria”` region. The generated data can be found in the
`“populations/astoria”` folder. You can modify the region name, population size,
and other parameters to generate data for different scenarios.
