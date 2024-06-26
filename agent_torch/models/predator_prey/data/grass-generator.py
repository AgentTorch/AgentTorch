# grass-generator.py
# (python grass-generator.py > grass.csv)

from random import randint, uniform

growth_stages = [str(randint(0, 1)) for _ in range(450)]

countdowns = []
for x in growth_stages:
    if x == "0":
        countdowns.append(str(round(uniform(50, 100), 4)))
    else:
        countdowns.append("0")

print("\n".join(growth_stages))
print("-------")
print("\n".join(countdowns))
