"""From census API, obtain multiple distributions for a county
    - household size distribution
    - age distribution
    - occupation distribution
    Variables as per https://api.census.gov/data/2021/acs/acs5/variables.html
"""

import requests
import pandas as pd
import pdb
import json
import os

def obtain_household_size_distribution(county_fips, census_api_key):
    """ Obtain household size distribution for county from Census data"""

    # Define the variables you want to query
    household_variables = [
        "B11016_001E", "B11016_003E", "B11016_011E", "B11016_004E",
        "B11016_012E", "B11016_005E", "B11016_013E", "B11016_006E",
        "B11016_014E", "B11016_007E", "B11016_008E", "B11016_015E",
        "B11016_016E"
    ]

    # Define the dictionary with the column mapping
    household_size_acs_var_map = {
        "1":
        "B11016_001E - B11016_003E - B11016_011E - B11016_004E -"
        "B11016_012E - B11016_005E - B11016_013E - B11016_006E -"
        "B11016_014E - B11016_007E - B11016_008E - B11016_015E -"
        "B11016_016E",
        "2":
        "B11016_003E + B11016_011E",
        "3":
        "B11016_004E + B11016_012E",
        "4":
        "B11016_005E + B11016_013E",
        "5":
        "B11016_006E + B11016_014E",
        "6":
        "B11016_007E + B11016_008E + B11016_015E + B11016_016E",
    }

    base_url = "https://api.census.gov/data/2020/acs/acs5"
    variables_str = ",".join(household_variables)
    url = f"{base_url}?get=NAME,{variables_str}&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&key={census_api_key}"

    response = requests.get(url)

    # Check the response status code
    if response.status_code == 200:
        # The request was successful, so parse the JSON response
        results = json.loads(response.content)
        # construct dataframe
        df = pd.DataFrame(results[1:], columns=results[0])
        # convert all columns to numeric
        df = df.apply(pd.to_numeric, errors='ignore')
        # Create new columns based on the dictionary mapping
        for new_column, formula in household_size_acs_var_map.items():
            df[new_column] = df.eval(formula)
    else:
        # The request failed, so print the error message
        print(response.status_code, response.reason)
        print('Failed for county: ', county_fips)
        quit()

    # keep only new columns
    df = df[list(household_size_acs_var_map.keys())]
    """ Save with format as csv file
        Household Size,Number
        1,329114
        2,306979
        3,139176
        4,115757
        5,45162
        6,30375
    """
    # transpose dataframe
    df = df.transpose()
    # reset index
    df = df.reset_index()
    # rename columns
    df = df.rename(columns={"index": "Household Size", 0: "Number"})
    save_path = f"./metadata/{county_fips}/agents_household_sizes.csv"
    df.to_csv(save_path, index=False)


def obtain_age_distribution(county_fips, census_api_key):
    """ Obtain age distribution for county from Census data """

    age_variables = [
        "B01001_003E", "B01001_004E", "B01001_005E", "B01001_006E",
        "B01001_007E", "B01001_027E", "B01001_028E", "B01001_029E",
        "B01001_030E", "B01001_031E", "B01001_008E", "B01001_009E",
        "B01001_010E", "B01001_011E", "B01001_012E", "B01001_032E",
        "B01001_033E", "B01001_034E", "B01001_035E", "B01001_036E",
        "B01001_013E", "B01001_014E", "B01001_015E", "B01001_016E",
        "B01001_017E", "B01001_037E", "B01001_038E", "B01001_039E",
        "B01001_040E", "B01001_041E", "B01001_018E", "B01001_019E",
        "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E",
        "B01001_024E", "B01001_025E", "B01001_042E", "B01001_043E",
        "B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E",
        "B01001_048E", "B01001_049E"
    ]

    # Define the dictionary with the column mapping
    age_acs_var_map = {
        "AGE_0_9":
        "B01001_003E + B01001_004E + B01001_027E + B01001_028E",
        "AGE_10_19":
        "B01001_005E + B01001_006E + B01001_007E + B01001_029E +"
        "B01001_030E + B01001_031E",
        "AGE_20_29":
        "B01001_008E + B01001_009E + B01001_010E + "
        "B01001_011E + B01001_032E + B01001_033E + "
        "B01001_034E + B01001_035E",
        "AGE_30_39":
        "B01001_012E + B01001_013E + B01001_036E + B01001_037E",
        "AGE_40_49":
        "B01001_014E + B01001_015E + B01001_038E + B01001_039E",
        "AGE_50_59":
        "B01001_016E + B01001_017E + B01001_040E + B01001_041E",
        "AGE_60_69":
        "B01001_018E + B01001_019E + B01001_020E + B01001_021E +"
        "B01001_042E + B01001_043E + B01001_044E + B01001_045E",
        "AGE_70_79":
        "B01001_022E + B01001_023E + B01001_046E + B01001_047E",
        "AGE_80":
        "B01001_024E + B01001_025E + B01001_048E + B01001_049E"
    }

    base_url = "https://api.census.gov/data/2020/acs/acs5"
    variables_str = ",".join(age_variables)
    url = f"{base_url}?get=NAME,{variables_str}&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&key={census_api_key}"

    response = requests.get(url)

    # Check the response status code
    if response.status_code == 200:
        # The request was successful, so parse the JSON response
        results = json.loads(response.content)
        # construct dataframe
        df = pd.DataFrame(results[1:], columns=results[0])
        # convert all columns to numeric
        df = df.apply(pd.to_numeric, errors='ignore')
        # Create new columns based on the dictionary mapping
        for new_column, formula in age_acs_var_map.items():
            df[new_column] = df.eval(formula)
    else:
        # The request failed, so print the error message
        print(response.status_code, response.reason)
        print('Failed for county: ', county_fips)
        quit()

    # keep only new columns
    df = df[list(age_acs_var_map.keys())]
    """ Save with format as csv file
        Age,Number
        AGE_0_9,278073
        AGE_10_19,258328
        AGE_20_29,317005
        AGE_30_39,359688
        AGE_40_49,323457
        AGE_50_59,307938
        AGE_60_69,229274
        AGE_70_79,109487
        AGE_80,69534
    """
    # transpose dataframe
    df = df.transpose()
    # reset index
    df = df.reset_index()
    # rename columns
    df = df.rename(columns={"index": "Age", 0: "Number"})
    df.to_csv(f"./metadata/{county_fips}/agents_ages.csv", index=False)


def obtain_occupation_distribution(county_fips, census_api_key):
    """ Obtain occupation distribution for county from Census data 
        Using NAICS sectors as Abueg et al. 2021, npj Digital Medicine
        
    """
    # List of NAICS sectors to query as per
    # https://www.census.gov/programs-surveys/economic-census/year/2022/guidance/understanding-naics.html
    # NOTE: only for reference, using inverted_naics_sectors instead
    naics_sectors = {
        "11": "Agriculture, Forestry, Fishing and Hunting",
        "21": "Mining, Quarrying, and Oil and Gas Extraction",
        "22": "Utilities",
        "23": "Construction",
        "31-33": "Manufacturing",
        "42": "Wholesale Trade",
        "44-45": "Retail Trade",
        "48-49": "Transportation and Warehousing",
        "51": "Information",
        "52": "Finance and Insurance",
        "53": "Real Estate and Rental and Leasing",
        "54": "Professional, Scientific, and Technical Services",
        "55": "Management of Companies and Enterprises",
        "56":
        "Administrative and Support and Waste Management and Remediation Services",
        "61": "Educational Services",
        "62": "Health Care and Social Assistance",
        "71": "Arts, Entertainment, and Recreation",
        "72": "Accommodation and Food Services",
        "81": "Other Services (except Public Administration)",
        "92": "Public Administration"
    }
    # Define the dictionary with the sector code mapping
    inverted_naics_sectors = {
        'AGRICULTURE': ['11'],
        'MINING': ['21'],
        'UTILITIES': ['22'],
        'CONSTRUCTION': ['23'],
        'MANUFACTURING': ['31', '32', '33'],
        'WHOLESALETRADE': ['42'],
        'RETAILTRADE': ['44', '45'],
        'TRANSPORTATION': ['48', '49'],
        'INFORMATION': ['51'],
        'FINANCEINSURANCE': ['52'],
        'REALESTATERENTAL': ['53'],
        'SCIENTIFICTECHNICAL': ['54'],
        'ENTERPRISEMANAGEMENT': ['55'],
        'WASTEMANAGEMENT': ['56'],
        'EDUCATION': ['61'],
        'HEALTHCARE': ['62'],
        'ART': ['71'],
        'FOOD': ['72'],
        'OTHER': ['81', '92'],  # adding public administration to other
    }

    df = [['Occupation', 'Number']]
    # convert example api.census.gov/data/2017/ecnbasic?get=NAICS2017_LABEL,EMP,NAME,GEO_ID&for=us:*&NAICS2017=54&key=YOUR_KEY_GOES_HERE
    base_url = "https://api.census.gov/data/2017/ecnbasic"
    for naics_sector in inverted_naics_sectors:
        print('-------------------')
        for naics_sector_code in inverted_naics_sectors[naics_sector]:
            print(naics_sector, naics_sector_code)
            url = f"{base_url}?get=NAICS2017_LABEL,EMP,NAME,GEO_ID&for=county:{county_fips[2:]}&in=state:{county_fips[:2]}&NAICS2017={naics_sector_code}&key={census_api_key}"
            # Make the API request
            response = requests.get(url)
            # Check the response status code
            if response.status_code == 200:
                # The request was successful, so parse the JSON response
                results = json.loads(response.content)
                row = results[1][:2]
                # label sector
                row[0] = naics_sector
                df.append(row)
            else:
                # The request failed, so print the error message
                print(response.status_code, response.reason)
                row = [naics_sector, 0]
                df.append(row)

    # construct dataframe
    df = pd.DataFrame(df[1:], columns=df[0])
    # convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='ignore')
    # aggregate by sector but do not sort by sector
    df = df.groupby(['Occupation'], sort=False).sum()
    # reset index
    df = df.reset_index()
    df.to_csv(f"./metadata/{county_fips}/agents_occupations.csv", index=False)


if __name__ == "__main__":

    # Define your Census API key
    CENSUS_API_KEY = "7a25a7624075d46f112113d33106b6648f42686a"

    # Define the county FIPS code
    # 27109 is Olmsted County, MN
    # 27049 is Goodhue County, MN
    # 27051 is Grant County, MN
    # 27053 is Hennepin County, MN
    for county_fips in ["27109", "27049", "27051", "27053"]:
        # create directory if it does not exist
        if not os.path.exists(f"./data/{county_fips}"):
            os.makedirs(f"./data/{county_fips}")
        obtain_household_size_distribution(county_fips, CENSUS_API_KEY)
        obtain_age_distribution(county_fips, CENSUS_API_KEY)
        obtain_occupation_distribution(county_fips, CENSUS_API_KEY)