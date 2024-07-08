# plot.py
# shows the prey and predators and grass on a scatterplot

import os
import torch
import numpy as np
import osmnx as ox

import matplotlib
import imageio

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plotter
import matplotlib.patches as patcher
import contextily as ctx


class Plot:
    def __init__(self, max_x, max_y):
        # intialize the scatterplot
        self.figure, self.axes = None, None
        self.max_x, self.max_y = max_x, max_y
        self.images = []

        plotter.xlim(0, max_x - 1)
        plotter.ylim(0, max_y - 1)

    def capture(self, step, state):
        graph = state["network"]["agent_agent"]["predator_prey"]["graph"]
        self.coords = [(node[1]["x"], node[1]["y"]) for node in graph.nodes(data=True)]
        self.coords.sort(key=lambda x: -(x[0] + x[1]))

        if self.figure is None:
            self.figure, self.axes = ox.plot_graph(
                graph, edge_linewidth=0.3, edge_color="gray", show=False, close=False
            )
            ctx.add_basemap(
                self.axes,
                crs=graph.graph["crs"],
                source=ctx.providers.OpenStreetMap.Mapnik,
            )
            self.axes.set_axis_off()

        # get coordinates of all the entities to show.
        prey = state["agents"]["prey"]
        pred = state["agents"]["predator"]
        grass = state["objects"]["grass"]

        # agar energy > 0 hai... toh zinda ho tum!
        alive_prey = prey["coordinates"][torch.where(prey["energy"] > 0)[0]]
        alive_pred = pred["coordinates"][torch.where(pred["energy"] > 0)[0]]
        # show only fully grown grass, which can be eaten.
        grown_grass = grass["coordinates"][torch.where(grass["growth_stage"] == 1)[0]]

        alive_prey_x, alive_prey_y = np.array(
            [self.coords[(self.max_y * pos[0]) + pos[1]] for pos in alive_prey]
        ).T
        alive_pred_x, alive_pred_y = np.array(
            [self.coords[(self.max_y * pos[0]) + pos[1]] for pos in alive_pred]
        ).T

        # show prey in dark blue, predators in maroon, and
        # grass in light green.
        prey_scatter = self.axes.scatter(
            alive_prey_x, alive_prey_y, c="#0d52bd", marker="."
        )
        pred_scatter = self.axes.scatter(
            alive_pred_x, alive_pred_y, c="#8b0000", marker="."
        )
        # grass_scatter = self.axes.scatter(grass_x, grass_y, c='#d1ffbd')

        # show the current step count, and the population counts.
        self.axes.set_title("Predator-Prey Simulation", loc="left")
        self.axes.legend(
            handles=[
                patcher.Patch(color="#fc46aa", label=f"{step} step"),
                patcher.Patch(color="#0d52bd", label=f"{len(alive_prey)} prey"),
                patcher.Patch(color="#8b0000", label=f"{len(alive_pred)} predators"),
            ]
        )

        # say cheese!
        self.figure.savefig(f"plots/predator-prey-map-{step}.png")
        self.images.append(f"plots/predator-prey-map-{step}.png")

        # remove the points for the next update.
        prey_scatter.remove()
        pred_scatter.remove()
        # grass_scatter.remove()

    def compile(self, episode):
        # convert all the images to a gif
        frames = [imageio.imread(f) for f in self.images]
        imageio.mimsave(f"media/predator-prey-{episode}.gif", frames, fps=50)

        # reset the canvas
        self.figure, self.axes = None, None
        self.images = []
