from enum import Enum, auto

class Feature(Enum):
    RETAIL_CHANGE = auto()
    GROCERY_CHANGE = auto()
    PARKS_CHANGE = auto()
    TRANSIT_CHANGE = auto()
    WORK_CHANGE = auto()
    RESIDENTIAL_CHANGE = auto()
    CASES = auto()
    CASES_4WK_AVG = auto()

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
            Feature.CASES_4WK_AVG: "cases_4_week_avg",
        }[self]