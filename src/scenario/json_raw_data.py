import json
from collections import defaultdict

import networkx as nx
import numpy as np


class JsonRawDataScenario:
    def __init__(self, N1=2, N2=4, tf=60, sd=None, ninit=5, tripAttr=None, demand_input=None, demand_ratio=None,
                 trip_length_preference=0.25, grid_travel_time=1, fix_price=True, alpha=0.2, json_file=None, json_hr=9,
                 json_tstep=2, varying_time=False, json_regions=None):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region,
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distribution
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        self.sd = sd
        if sd != None:
            np.random.seed(self.sd)
        self.varying_time = varying_time
        self.is_json = True
        with open(json_file, "r") as file:
            data = json.load(file)
        self.tstep = json_tstep
        self.N1 = data["nlat"]
        self.N2 = data["nlon"]
        self.demand_input = defaultdict(dict)
        self.json_regions = json_regions

        if json_regions != None:
            self.G = nx.complete_graph(json_regions)
        elif 'region' in data:
            self.G = nx.complete_graph(data['region'])
        else:
            self.G = nx.complete_graph(self.N1 * self.N2)
        self.G = self.G.to_directed()
        self.p = defaultdict(dict)
        self.alpha = 0
        self.demandTime = defaultdict(dict)
        self.rebTime = defaultdict(dict)
        self.json_start = json_hr * 60
        self.tf = tf
        self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]

        for i, j in self.demand_input:
            self.demandTime[i, j] = defaultdict(int)
            self.rebTime[i, j] = 1

        for item in data["demand"]:
            t, o, d, v, tt, p = item["time_stamp"], item["origin"], item["destination"], item["demand"], item[
                "travel_time"], item["price"]
            if json_regions != None and (o not in json_regions or d not in json_regions):
                continue
            if (o, d) not in self.demand_input:
                self.demand_input[o, d], self.p[o, d], self.demandTime[o, d] = defaultdict(float), defaultdict(
                    float), defaultdict(float)

            self.demand_input[o, d][(t - self.json_start) // json_tstep] += v * demand_ratio
            self.p[o, d][(t - self.json_start) // json_tstep] += p * v * demand_ratio
            self.demandTime[o, d][(t - self.json_start) // json_tstep] += tt * v * demand_ratio / json_tstep

        for o, d in self.edges:
            for t in range(0, tf * 2):
                if t in self.demand_input[o, d]:
                    self.p[o, d][t] /= self.demand_input[o, d][t]
                    self.demandTime[o, d][t] /= self.demand_input[o, d][t]
                    self.demandTime[o, d][t] = max(int(round(self.demandTime[o, d][t])), 1)
                else:
                    self.demand_input[o, d][t] = 0
                    self.p[o, d][t] = 0
                    self.demandTime[o, d][t] = 0

        for item in data["rebTime"]:
            hr, o, d, rt = item["time_stamp"], item["origin"], item["destination"], item["reb_time"]
            if json_regions != None and (o not in json_regions or d not in json_regions):
                continue
            if varying_time:
                t0 = int((hr * 60 - self.json_start) // json_tstep)
                t1 = int((hr * 60 + 60 - self.json_start) // json_tstep)
                for t in range(t0, t1):
                    self.rebTime[o, d][t] = max(int(round(rt / json_tstep)), 1)
            else:
                if hr == json_hr:
                    for t in range(0, tf + 1):
                        self.rebTime[o, d][t] = max(int(round(rt / json_tstep)), 1)

        for item in data["totalAcc"]:
            hr, acc = item["hour"], item["acc"]
            if hr == json_hr + int(round(json_tstep / 2 * tf / 60)):
                for n in self.G.nodes:
                    self.G.nodes[n]['accInit'] = int(acc / len(self.G))
        self.tripAttr = self.get_random_demand()

    def get_random_demand(self, reset=False):
        # generate demand and price
        # reset = True means that the function is called in the reset() method of AMoD enviroment,
        #   assuming static demand is already generated
        # reset = False means that the function is called when initializing the demand

        demand = defaultdict(dict)
        price = defaultdict(dict)
        tripAttr = []

        # converting demand_input to static_demand
        # skip this when resetting the demand
        # if not reset:
        for t in range(0, self.tf * 2):
            for i, j in self.edges:
                if (i, j) in self.demand_input and t in self.demand_input[i, j]:
                    demand[i, j][t] = np.random.poisson(self.demand_input[i, j][t])
                    price[i, j][t] = self.p[i, j][t]
                else:
                    demand[i, j][t] = 0
                    price[i, j][t] = 0
                tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))

        return tripAttr