from epiweeks import Week

from macro_economics.utils.neighborhood import Neighborhood

##### epiweek conversions #####


def epiweek_to_week_num(epiweek: Week):
    return int(epiweek.cdcformat())


def week_num_to_epiweek(week_num: int):
    return Week.fromstring(str(week_num))


def subtract_epiweek(epiweek1: Week, epiweek2: Week):
    ans = 0
    while epiweek2 + ans != epiweek1:
        ans += 1
    return ans


##### neighborhood factory #####


def name_to_neighborhood(name: str) -> Neighborhood:
    for neighborhood in Neighborhood:
        if neighborhood.name == name:
            return neighborhood
    raise Exception("could not find neighborhood")
