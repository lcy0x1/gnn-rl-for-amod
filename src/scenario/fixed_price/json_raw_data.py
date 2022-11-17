import json
from collections import defaultdict
from copy import deepcopy
from random import random

import networkx as nx
import numpy as np

from src.misc.graph_wrapper import GraphWrapper
from src.misc.types import *
from src.scenario.scenario import Scenario


class JsonRawDataScenario(Scenario):
    def __init__(self, tf, sd, demand_ratio, json_file, json_hr, json_tstep,
                 time_skip, varying_time=False):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region,
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distribution
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        super().__init__(json_tstep, seed=sd, tf=tf)

        assert json_hr * 60 + tf * json_tstep * time_skip <= 1440, "time exceeds maximum"

        with open(json_file, "r") as file:
            data = json.load(file)
        n1 = data["nlat"]
        n2 = data["nlon"]

        if 'region' in data:
            nregion = data['region']
        else:
            nregion = n1 * n2
        self.G = nx.complete_graph(nregion).to_directed()
        self.time_skip = time_skip
        self.price = defaultdict(dict)
        self.demand_time = defaultdict(dict)
        self.reb_time = defaultdict(dict)
        self.init_acc = defaultdict(int)
        self.demand_input = defaultdict(dict)
        self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]
        json_start = json_hr * 60

        for item in data["demand"]:
            t, o, d, v, tt, p = item["time_stamp"], item["origin"], item["destination"], item["demand"], item[
                "travel_time"], item["price"]
            if (o, d) not in self.demand_input:
                self.demand_input[o, d], self.price[o, d], self.demand_time[o, d] = defaultdict(float), defaultdict(
                    float), defaultdict(float)

            self.demand_input[o, d][(t - json_start) // json_tstep] += v * demand_ratio
            self.price[o, d][(t - json_start) // json_tstep] += p * v * demand_ratio
            self.demand_time[o, d][(t - json_start) // json_tstep] += tt * v * demand_ratio / json_tstep

        for o, d in self.edges:
            for t in range(0, tf * time_skip * 2):
                if t in self.demand_input[o, d]:
                    self.price[o, d][t] /= self.demand_input[o, d][t]
                    self.demand_time[o, d][t] /= self.demand_input[o, d][t]
                    self.demand_time[o, d][t] = max(int(round(self.demand_time[o, d][t])), 1)
                else:
                    self.demand_input[o, d][t] = 0
                    self.price[o, d][t] = 0
                    self.demand_time[o, d][t] = 0

        for item in data["rebTime"]:
            hr, o, d, rt = item["time_stamp"], item["origin"], item["destination"], item["reb_time"]
            if varying_time:
                t0 = int((hr * 60 - json_start) // json_tstep)
                t1 = int((hr * 60 + 60 - json_start) // json_tstep)
                for t in range(t0, t1):
                    self.reb_time[o, d][t] = max(int(round(rt / json_tstep)), 1)
            else:
                if hr == json_hr:
                    for t in range(0, tf * time_skip + 1):
                        self.reb_time[o, d][t] = max(int(round(rt / json_tstep)), 1)

        for item in data["totalAcc"]:
            hr, acc = item["hour"], item["acc"]
            if hr == json_hr + int(round(json_tstep / 2 * tf * time_skip / 60)):
                for n in self.G.nodes:
                    self.init_acc[n] = int(acc / len(self.G))

        self.time_dict = defaultdict(int)
        self.tripAttr = self.get_random_demand()

    def get_init_acc(self, n: Node) -> int:
        return self.init_acc[n]

    def get_demand_time(self, o: Node, d: Node, t: Time) -> Time:
        return max(1, self.demand_time[o, d][self.time_dict[t]])

    def get_reb_time(self, o: Node, d: Node, t: Time) -> Time:
        return max(1, self.reb_time[o, d][self.time_dict[t]])

    def get_graph(self) -> GraphWrapper:
        return GraphWrapper(deepcopy(self.G))

    def get_random_demand(self, reset=False):
        # generate demand and price
        # reset = True means that the function is called in the reset() method of AMoD enviroment,
        #   assuming static demand is already generated
        # reset = False means that the function is called when initializing the demand

        demand = defaultdict(dict)
        price = defaultdict(dict)
        trip_attr = []

        # converting demand_input to static_demand
        # skip this when resetting the demand
        # if not reset:
        for t in range(0, self.get_final_time() * 2):
            time = t * self.time_skip + int(random() * self.time_skip)
            self.time_dict[t] = time
            for i, j in self.edges:
                if (i, j) in self.demand_input and time in self.demand_input[i, j]:
                    demand[i, j][t] = self.demand_input[i, j][time]
                    price[i, j][t] = self.price[i, j][time]
                else:
                    demand[i, j][t] = 0
                    price[i, j][t] = 0
                trip_attr.append((i, j, t, demand[i, j][t], price[i, j][t]))

        return trip_attr
