import os
from census_data import (
    obtain_household_size_distribution,
    obtain_age_distribution,
    obtain_occupation_distribution,
)

CENSUS_API_KEY = "de71ab1394b792bb409c0fc5b67ffaf792b7fe6f"

# The Bronx is Bronx County (ANSI / FIPS 36005)
# Brooklyn is Kings County (ANSI / FIPS 36047)
# Manhattan is New York County (ANSI / FIPS 36061)
# Queens is Queens County (ANSI / FIPS 36081)
# Staten Island is Richmond County (ANSI / FIPS 36085)

NYC_counties = ["36005", "36047", "36061", "36081", "36085"]

for county_fips in NYC_counties:
    # create directory if it does not exist
    if not os.path.exists(f"./metadata/{county_fips}"):
        os.makedirs(f"./metadata/{county_fips}")

    print("processing_county: ", county_fips)

    print("generating.. households")
    obtain_household_size_distribution(county_fips, CENSUS_API_KEY)
    print("generating.. age")
    obtain_age_distribution(county_fips, CENSUS_API_KEY)
    print("generating.. occupations")
    obtain_occupation_distribution(county_fips, CENSUS_API_KEY)
