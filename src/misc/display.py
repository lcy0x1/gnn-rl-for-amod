import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.misc.info import LogInfo, LogEntry
from src.misc.resource_locator import ResourceLocator


def check(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def ave(data, rate=0.95):
    ans = np.array(data)
    acc = ans[0] / (1 - rate)
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
    t1 = 0
    if t1 == 0:
        t1 = len(log.lists[LogEntry.reward])
    print(f'Data Points: {t1}')
    display(ave(log.lists[LogEntry.reward])[t0:t1], f"{path}reward.png")
    display(ave(log.lists[LogEntry.served_demand])[t0:t1], f"{path}served_demand.png")
    display(ave(log.lists[LogEntry.reb_cost])[t0:t1], f"{path}reb_cost.png")
    display(ave(log.lists[LogEntry.price_point])[t0:t1], f"{path}price_point.png")
