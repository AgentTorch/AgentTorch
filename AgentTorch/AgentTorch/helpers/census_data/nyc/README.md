Good references -
NYC specific:
1. https://popfactfinder.planning.nyc.gov/#17.89/40.766372/-73.917947
2. Census metadata source: https://www.nyc.gov/site/planning/data-maps/open-data/census-download-metadata.page
3. NYC ACS data source: https://www.nyc.gov/site/planning/planning-level/nyc-population/american-community-survey.page
4. ACS-5 dataset: https://www2.census.gov/programs-surveys/acs/data/pums/2021/5-Year/ 
5. Demographics at NTA level: https://catalog.data.gov/dataset/demographics-and-profiles-at-the-neighborhood-tabulation-area-nta-level
6. Income and Poverty data: https://data.census.gov/all?q=New+York+city,+New+York+Income+and+Poverty 

General:
6. ACS-5 JSON references: https://api.census.gov/data/2021/acs/acs5/pums/variables.json
7. Census ACS-5 microdata: https://www.census.gov/data/developers/data-sets/census-microdata-api.ACS_5-Year_PUMS.html#list-tab-71345371 
8. Census SDK: https://www.census.gov/data/developers.html
9. Census demographic profile: https://www.census.gov/data/tables/2023/dec/2020-census-demographic-profile.html 
10. Census families and households: https://www.census.gov/topics/families/families-and-households.html
11. Household pulse survey (COVID-19): https://www.census.gov/data/developers/data-sets/hps.html
12. Survey of Income and Program Participation: https://www.census.gov/programs-surveys/sipp.html 
13. All ACS data tables: https://www.census.gov/programs-surveys/acs/data/data-tables.html 

PUMS: Public-use microdata.
Level of abstraction of ACS-5 is at NTA and CDTA.

For the simulation, we need information about: a) individual demographics (ace, race, gender, income, occupation); b) households (size, assets, expenses, lat-long)

Datasets we need:
1. [DONE] Map between CDTA, NTA and County Ids
2. [DONE] Use ACS to process: a) demographic b) social c) economic d) housing datasets.
3. Use Wealth and Assets dataset for: a) assets b) earnings
4. Use Occupation dataset to process: occupations
5. Use BLS to process: a) expense dataset (by income band) and split across durable and non-durable expenditure

Clinical model at individual level and financial model at household level.
- Assets = R*Assets + Earning - Expenses