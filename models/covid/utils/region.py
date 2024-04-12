from enum import Enum, auto

class Neighborhood(Enum):
    ASTORIA_SOUTH_LIC_SUNNYSIDE = auto()
    UPPER_WEST_SIDE = auto()
    BAY_RIDGE = auto()
    MORRIS_PARK_PELHAM_BAY_WESTCHESTER_SQUARE = auto()
    PORT_RICHMOND_RANDALL_MANOR_WEST_BRIGHTON = auto()

    @property
    def nta_id(self) -> str:
        return {
            Neighborhood.ASTORIA_SOUTH_LIC_SUNNYSIDE: "QN0103",
            Neighborhood.UPPER_WEST_SIDE: "MN0702",
            Neighborhood.BAY_RIDGE: "BK1001",
            Neighborhood.MORRIS_PARK_PELHAM_BAY_WESTCHESTER_SQUARE: "BX1003",
            Neighborhood.PORT_RICHMOND_RANDALL_MANOR_WEST_BRIGHTON: "SI0106",
        }[self]
    