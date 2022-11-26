import numpy as np


def mat2str(mat):
    return str(mat).replace("'", '"').replace('(', '<').replace(')', '>').replace('[', '{').replace(']', '}')


def dictsum(dic, t):
    return sum([dic[key][t] for key in dic if t in dic[key]])


def moving_average(a, n=3):
    """
    Computes a moving average used for reward trace smoothing.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class RunningAverage:

    def __init__(self, rate=0.95):
        self.rate = rate
        self.best = 0
        self.current = 0

    def accept(self, val):
        self.current = self.current * self.rate + val
        if self.current > self.best:
            self.best = self.current
            return True
        return False
