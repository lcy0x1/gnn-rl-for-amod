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
        acc += ans[i]
        ans[i] = acc * (1 - rate)
    return ans


def display(data, dst, ytick=0, ymax=0, title=None, legend=None):
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    n = len(data)

    ax.plot(range(n), data, linewidth=2.0, label=legend)

    ax.set(xlim=(0, n), ylim=(0, ymax), xticks=np.arange(0, n, 60), yticks=np.arange(0, ymax, ytick))

    if title is not None:
        ax.set_title(title)
    if legend is not None:
        ax.legend(legend)
    ax.grid()
    fig.savefig(dst)


def display_sum(data, dst, ytick=0, ymax=0, title: str = None, legend: [str] = None):
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    n = len(data[0])
    cul = 0
    ind = 0
    for d in data:
        cul = np.array(d) + cul
        leg = None if legend is None else legend[ind]
        ax.plot(range(n), cul, linewidth=2.0, label=leg)
        ind += 1

    ax.set(xlim=(0, n), ylim=(0, ymax), xticks=np.arange(0, n, 60), yticks=np.arange(0, ymax, ytick))

    if title is not None:
        ax.set_title(title)
    if legend is not None:
        ax.legend()
    ax.grid()
    fig.savefig(dst)


def view(paths: ResourceLocator, source: str):
    paths.graph_folder = "output_" + paths.graph_folder
    dr = 12
    name = "Dynamic Price"
    path = paths.save_graphs(source + f'_dr{dr}')
    check(path)
    log_file = torch.load(paths.train_log() if source == 'train' else paths.test_log())
    log = LogInfo()
    log.from_obj(source, log_file)
    t0 = 1
    t1 = 0
    if t1 == 0:
        t1 = len(log.lists[LogEntry.reward])
    print(f'Data Points: {t1}')
    func = ave if source == 'train' else lambda e: e
    display(func(log.lists[LogEntry.reward])[t0:t1], f"{path}reward.png",
            ytick=2000, ymax=16000,
            title=f"Reward for {name} at dr={dr}",
            legend=[f"Total Reward: {round(sum(log.lists[LogEntry.reward]))}"])
    display_sum([func(log.lists[LogEntry.served_demand])[t0:t1],
                 func(log.lists[LogEntry.missed_demand])[t0:t1]],
                f"{path}served_demand.png",
                ytick=200, ymax=1600,
                title=f"Served and Total Demand for {name} at dr={dr}",
                legend=["Served Demand", "Total Demand"])
    # display(func(log.lists[LogEntry.reb_cost])[t0:t1], f"{path}reb_cost.png")
    # display_sum([func(log.lists[LogEntry.pax_vehicle])[t0:t1],
    #             func(log.lists[LogEntry.reb_vehicle])[t0:t1],
    #             func(log.lists[LogEntry.idle_vehicle])[t0:t1]], f"{path}percentages.png")
