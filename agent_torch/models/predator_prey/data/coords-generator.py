# coords-generator.py
# (python coords-generator.py > coords.csv)

from itertools import product
from random import sample

grid_size = 18
num_coords = 40

coords = sample(list(product(range(grid_size), repeat=2)), k=num_coords)

for coord in coords:
    print(str(coord[0]) + ",", coord[1])

# for row in range(18):
#   for col in range(25):
#     print(str(row) + ',', str(col))
