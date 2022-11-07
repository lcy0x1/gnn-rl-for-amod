from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np

from src.envs.graph_wrapper import GraphWrapper
from src.envs.types import Node, Time
from src.scenario.scenario import Scenario


class FixedPriceModelScenario(Scenario):

    def __init__(self, N1=2, N2=4, tf=60, sd=None, ninit=5, tripAttr=None, demand_input=None,
                 demand_ratio=None, trip_length_preference=0.25, grid_travel_time=1, fix_price=True, alpha=0.2,
                 json_file=None, json_hr=9, json_tstep=2, varying_time=False, json_regions=None):
        # trip_length_preference: positive - more shorter trips, negative - more longer trips
        # grid_travel_time: travel time between grids
        # demand_input： list - total demand out of each region,
        #          float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
        #          dict/defaultdict - total demand between pairs of regions
        # demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        # static_demand will then be sampled according to a Poisson distribution
        # alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        super().__init__(json_tstep, tf=tf, seed=sd)

        self.varying_time = varying_time
        self.is_json = False
        self.alpha = alpha
        self.trip_length_preference = trip_length_preference
        self.grid_travel_time = grid_travel_time
        self.demand_input = demand_input
        self.fix_price = fix_price
        self.N1 = N1
        self.N2 = N2
        self.G = nx.complete_graph(N1 * N2)
        self.G = self.G.to_directed()
        self.demandTime = dict()
        self.rebTime = dict()
        self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]
        for i, j in self.edges:
            self.demandTime[i, j] = defaultdict(
                lambda: (abs(i // N1 - j // N1) + abs(i % N1 - j % N1)) * grid_travel_time)
            self.rebTime[i, j] = defaultdict(
                lambda: (abs(i // N1 - j // N1) + abs(i % N1 - j % N1)) * grid_travel_time)

        for n in self.G.nodes:
            self.G.nodes[n]['accInit'] = int(ninit)

        self.demand_ratio = defaultdict(list)

        if demand_ratio == None or type(demand_ratio) == list:
            for i, j in self.edges:
                if type(demand_ratio) == list:
                    self.demand_ratio[i, j] = list(
                        np.interp(range(0, tf), np.arange(0, tf + 1, tf / (len(demand_ratio) - 1)),
                                  demand_ratio)) + [demand_ratio[-1]] * tf
                else:
                    self.demand_ratio[i, j] = [1] * (tf + tf)
        else:
            for i, j in self.edges:
                if (i, j) in demand_ratio:
                    self.demand_ratio[i, j] = list(
                        np.interp(range(0, tf), np.arange(0, tf + 1, tf / (len(demand_ratio[i, j]) - 1)),
                                  demand_ratio[i, j])) + [1] * tf
                else:
                    self.demand_ratio[i, j] = list(
                        np.interp(range(0, tf), np.arange(0, tf + 1, tf / (len(demand_ratio['default']) - 1)),
                                  demand_ratio['default'])) + [1] * tf
        if self.fix_price:  # fix price
            self.p = defaultdict(dict)
            for i, j in self.edges:
                self.p[i, j] = (np.random.rand() * 2 + 1) * (self.demandTime[i, j][0] + 1)
        if tripAttr != None:  # given demand as a defaultdict(dict)
            self.tripAttr = deepcopy(tripAttr)
        else:
            self.tripAttr = self.get_random_demand()  # randomly generated demand

    def get_demand_time(self, o: Node, d: Node, t: Time) -> Time:
        return self.demandTime[o, d][t]

    def get_reb_time(self, o: Node, d: Node, t: Time) -> Time:
        return self.rebTime[o, d][t]

    def get_graph(self) -> GraphWrapper:
        return GraphWrapper(deepcopy(self.G))

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
        self.static_demand = dict()
        region_rand = (np.random.rand(len(self.G)) * self.alpha * 2 + 1 - self.alpha)
        if type(self.demand_input) in [float, int, list, np.array]:

            if type(self.demand_input) in [float, int]:
                self.region_demand = region_rand * self.demand_input
            else:
                self.region_demand = region_rand * np.array(self.demand_input)
            for i in self.G.nodes:
                J = [j for _, j in self.G.out_edges(i)]
                prob = np.array([np.math.exp(-self.rebTime[i, j][0] * self.trip_length_preference) for j in J])
                prob = prob / sum(prob)
                for idx in range(len(J)):
                    self.static_demand[i, J[idx]] = self.region_demand[i] * prob[idx]
        elif type(self.demand_input) in [dict, defaultdict]:
            for i, j in self.edges:
                self.static_demand[i, j] = self.demand_input[i, j] if (i, j) in self.demand_input else \
                    self.demand_input['default']

                self.static_demand[i, j] *= region_rand[i]
        else:
            raise Exception("demand_input should be number, array-like, or dictionary-like values")

        # generating demand and prices
        if self.fix_price:
            p = self.p
        for t in range(0, self.tf * 2):
            for i, j in self.edges:
                demand[i, j][t] = np.random.poisson(self.static_demand[i, j] * self.demand_ratio[i, j][t])
                if self.fix_price:
                    price[i, j][t] = p[i, j]
                else:
                    price[i, j][t] = min(3, np.random.exponential(2) + 1) * self.demandTime[i, j][t]
                tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))

        return tripAttr
