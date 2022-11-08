from typing import Final

import numpy as np

from src.misc.graph_wrapper import GraphWrapper
from src.misc.types import *


class Scenario:
    def __init__(self, tstep: int, seed: int = None, tf: int = 60):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region,
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distribution
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        self._seed = seed
        self._tf = tf
        self._tstep: Final = tstep
        if seed is not None:
            np.random.seed(self._seed)

    def get_final_time(self) -> Time:
        return self._tf

    def get_step_time(self) -> Time:
        return self._tstep

    def get_graph(self) -> GraphWrapper:
        raise Exception("get_graph Not Implemented")

    def get_random_demand(self, reset=False) -> [(Node, Node, Time, Demand, Price)]:
        raise Exception("get_random_demand Not Implemented")

    def get_demand_time(self, o: Node, d: Node, t: Time) -> Time:
        raise Exception("get_demand_time Not Implemented")

    def get_reb_time(self, o: Node, d: Node, t: Time) -> Time:
        raise Exception("get_reb_time Not Implemented")
