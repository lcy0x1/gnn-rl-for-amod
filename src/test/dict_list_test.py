import random
import time
from collections import defaultdict

import numpy
from tqdm import trange

trial = 100
n = 100


def test_dict():
    obj = defaultdict(dict)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                obj[i, j][k] = random.random()


def test_list():
    obj = [[[0.] * n] * n] * n
    for i in range(n):
        for j in range(n):
            for k in range(n):
                obj[i][j][k] = random.random()


def test_np():
    obj = numpy.empty((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                obj[i][j][k] = random.random()


def test_all():
    time_dict = 0
    time_list = 0
    time_numpy = 0
    episodes = trange(trial)
    for episode in episodes:
        t = time.time()
        test_dict()
        time_dict += time.time() - t

        t = time.time()
        test_list()
        time_list += time.time() - t

        t = time.time()
        test_np()
        time_numpy += time.time() - t

    print(f'dict:{time_dict}, list:{time_list}, numpy:{time_numpy}')


if __name__ == '__main__':
    test_all()
