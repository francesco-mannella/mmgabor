import os

import matplotlib.pyplot as plt
import numpy as np


plt.ioff()

rpolys = np.array(
    [
        [
            [0.3, 0.2],
            [0.9, 0.1],
            [0.75, 0.87],
        ],
        [
            [0.65, 0.65],
            [0.75, 0.6],
            [0.6, 0.45],
        ],
        [
            [0.15, 0.65],
            [0.25, 0.6],
            [0.1, 0.45],
        ],
    ]
)
rpolys = np.hstack([rpolys, rpolys[:, :1, :]])


gpolys = np.array(
    [
        [
            [0.65, 0.65],
            [0.75, 0.6],
            [0.6, 0.45],
        ],
        [
            [0.3, 0.3],
            [0.9, 0.34],
            [0.75, 0.78],
        ],
        [
            [0.3, 0.3],
            [0.9, 0.34],
            [0.75, 0.78],
        ],
    ]
)

gpolys = np.hstack([gpolys, gpolys[:, :1, :]])

zorders = [[0, 1], [1, 0], [0, 1]]

if not os.path.exists("base_imgs"):
    os.makedirs("base_imgs")

for i, (rpoly, gpoly) in enumerate(zip(rpolys, gpolys)):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ppoly = plt.Polygon(rpoly, fc="red", zorder=zorders[i][0])
    ax.add_patch(ppoly)
    ppoly = plt.Polygon(gpoly, fc="green", zorder=zorders[i][1])
    ax.add_patch(ppoly)
    plt.show()
    fig.savefig(f"base_imgs/f{i}.jpg")
