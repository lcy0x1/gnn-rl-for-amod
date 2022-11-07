import matplotlib.pyplot as plt
import numpy as np


def display(data, dst):
    # plt.style.use('_mpl-gallery')
    # plot
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    xtick = 2000
    ytick = 5000
    n = len(data)
    y1 = round(max(data) * 1.2 / ytick) * ytick

    ax.plot(range(n), data, linewidth=2.0)

    # ax.set(xlim=(0, n), xticks=np.arange(0, n, xtick), ylim=(0, y1), yticks=np.arange(0, y1, ytick))

    ax.grid()

    fig.savefig(dst)
