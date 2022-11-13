import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.misc.info import LogInfo
from src.misc.resource_locator import ResourceLocator


def check(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def ave(data, rate=0.95):
    ans = np.array(data)
    acc = 0
    for i in range(len(ans)):
        acc *= rate
        acc += max(0, ans[i])
        ans[i] = acc * (1 - rate)
    return ans


def display(data, dst):
    # plt.style.use('_mpl-gallery')
    # plot
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    xtick = 2000
    ytick = 1
    n = len(data)
    y1 = round(max(data) * 1.2 / ytick) * ytick

    ax.plot(range(n), data, linewidth=2.0)

    ax.set(xlim=(0, n)
           # , ylim=(0, y1)
           # ,xticks=np.arange(0, n, xtick)
           # ,yticks=np.arange(0, y1, ytick)
           )

    ax.grid()

    fig.savefig(dst)


def display_sum(data, dst):
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    xtick = 2000
    ytick = 1
    n = len(data[0])
    cul = 0
    for d in data:
        cul = np.array(d) + cul
        ax.plot(range(n), d, linewidth=2.0)

    ax.set(xlim=(0, n), ylim=(0, 1))

    ax.grid()

    fig.savefig(dst)


def view(paths: ResourceLocator):
    path = paths.save_graphs()
    check(path)
    log_file = torch.load(paths.train_log())
    log = LogInfo()
    log.from_obj('train', log_file)
    t0 = 0
    t1 = 9197
    if t1 == 0:
        t1 = len(log.reward)
    print(f'Data Points: {t1}')
    display(ave(log.reward)[t0:t1], f"{path}reward.png")
    display(ave(log.served_demand)[t0:t1], f"{path}served_demand.png")
    display(ave(log.reb_cost)[t0:t1], f"{path}reb_cost.png")
    display(ave(log.price_point)[t0:t1], f"{path}price_point.png")
    display_sum([log.reb_vehicle[t0:t1], log.pax_vehicle[t0:t1], log.idle_vehicle[t0:t1]], f"{path}percentages.png")
