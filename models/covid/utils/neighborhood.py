from enum import Enum


class Neighborhood(Enum):
    ASTORIA = 0
    UPPER_WEST_SIDE = 1
    BAY_RIDGE = 2
    PELHAM_BAY = 3
    PORT_RICHMOND = 4
    
    # fiction land is a neighborhood which is identical to astoria except with a scaled population
    FICTION_LAND = 5

    @property
    def name(self) -> str:
        return {
            Neighborhood.ASTORIA: "Astoria",
            Neighborhood.UPPER_WEST_SIDE: "Upper West Side",
            Neighborhood.BAY_RIDGE: "Bay Ridge",
            Neighborhood.PELHAM_BAY: "Pelham Bay",
            Neighborhood.PORT_RICHMOND: "Port Richmond",
            Neighborhood.FICTION_LAND: "Astoria",
        }[self]

    @property
    def nta_id(self) -> str:
        return {
            Neighborhood.ASTORIA: "QN0103",
            Neighborhood.UPPER_WEST_SIDE: "MN0702",
            Neighborhood.BAY_RIDGE: "BK1001",
            Neighborhood.PELHAM_BAY: "BX1003",
            Neighborhood.PORT_RICHMOND: "SI0106",
            Neighborhood.FICTION_LAND: "QN0103",
        }[self]

    @property
    def population(self) -> int:
        return {
            Neighborhood.ASTORIA: 36835,
            Neighborhood.UPPER_WEST_SIDE: 58412,
            Neighborhood.BAY_RIDGE: 67514,
            Neighborhood.PELHAM_BAY: 49097,
            Neighborhood.PORT_RICHMOND: 23441,
            Neighborhood.FICTION_LAND: 37518,
        }[self]
    
    @property
    def text(self) -> str:
        return f"lives in {self.name}, New York, a neighborhood with a population of {self.population}"

