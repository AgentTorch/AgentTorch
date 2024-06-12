from enum import Enum

class Neighborhood(Enum):
    ASTORIA = 0
    UPPER_WEST_SIDE = 1
    BAY_RIDGE = 2
    PELHAM_BAY = 3
    PORT_RICHMOND = 4

    @property
    def name(self) -> str:
        return {
            Neighborhood.ASTORIA: "Astoria",
            Neighborhood.UPPER_WEST_SIDE: "Upper West Side",
            Neighborhood.BAY_RIDGE: "Bay Ridge",
            Neighborhood.PELHAM_BAY: "Pelham Bay",
            Neighborhood.PORT_RICHMOND: "Port Richmond",
        }[self]

    @property
    def nta_id(self) -> str:
        return {
            Neighborhood.ASTORIA: "QN0103",
            Neighborhood.UPPER_WEST_SIDE: "MN0702",
            Neighborhood.BAY_RIDGE: "BK1001",
            Neighborhood.PELHAM_BAY: "BX1003",
            Neighborhood.PORT_RICHMOND: "SI0106",
        }[self]

    @property
    def population(self) -> int:
        return {
            # this is a bug. the population is actually 36835, but since we are using a config file
            # with a population of 37518, this is what we set it to here as well.
            Neighborhood.ASTORIA: 37518,
            Neighborhood.UPPER_WEST_SIDE: 58412,
            Neighborhood.BAY_RIDGE: 67514,
            Neighborhood.PELHAM_BAY: 49097,
            Neighborhood.PORT_RICHMOND: 23441,
        }[self]

    @property
    def text(self) -> str:
        return f"lives in {self.name}, New York, a neighborhood with a population of {self.population}"
