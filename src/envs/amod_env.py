"""
Autonomous Mobility-on-Demand Environment
-----------------------------------------
This file contains the specifications for the AMoD system simulator. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""
from collections import defaultdict

import numpy

from src.algos.cplex_handle import CPlexHandle
from src.misc.graph_wrapper import GraphWrapper
from src.misc.types import *
from src.scenario.scenario import Scenario
from src.misc.info import StepInfo


class TimelyData:

    def __init__(self, scenario: Scenario, graph: GraphWrapper):
        self._scenario = scenario
        self._graph = graph

        # number of vehicles within each region, key: i - region, t - time
        self.acc = defaultdict(dict)
        # number of vehicles arriving at each region, key: i - region, t - time
        self.dacc = defaultdict(dict)

        self._demand = defaultdict(dict)  # demand
        self._price = defaultdict(dict)  # price
        self._var_price = defaultdict(lambda: defaultdict(lambda: 1))

        # record only, not for calculation
        self.servedDemand = defaultdict(dict)
        # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.rebFlow = defaultdict(dict)
        # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(dict)

        self.reset(graph, scenario.get_random_demand())

    def reset(self, graph: GraphWrapper, trip_attr):
        for i, j in graph.get_all_edges():
            self.rebFlow[i, j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
            self.servedDemand[i, j] = defaultdict(float)
        for n in range(graph.size()):
            self.acc[n][0] = self._scenario.get_init_acc(n)
            self.dacc[n] = defaultdict(float)
        for i, j, t, d, p in trip_attr:  # trip attribute (origin, destination, time of request, demand, price)
            self._demand[i, j][t] = d
            self._price[i, j][t] = p

    def get_demand(self, o: Node, d: Node, t: Time):
        return self._demand[o, d][t] * numpy.exp(1 - self._var_price[o, d][t])

    def set_prices(self, prices, t: Time):
        for o in range(self._graph.size()):
            for d in range(self._graph.size()):
                self._var_price[o, d][t] = prices[o, d]

    def get_price(self, o: Node, d: Node, t: Time):
        return self._price[o, d][t] * self._var_price[o, d]


class AMoD:
    # initialization
    def __init__(self, scenario: Scenario,
                 beta=0.2):  # updated to take scenario and beta (cost for rebalancing) as input
        # I changed it to deep copy so that the scenario input is not modified by env
        self.scenario = scenario
        # Road Graph: node - region, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.graph = self.scenario.get_graph()
        self.time = 0  # current time
        self.tf = self.scenario.get_final_time()  # final time
        self.region = self.graph.node_list()  # set of regions
        self.nregion = self.graph.size()  # number of regions
        self.beta = beta * self.scenario.get_step_time()

        self.edges = []  # set of rebalancing edges
        for i in range(self.nregion):
            self.edges.append((i, i))
            for e in self.graph.get_out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.nedge = [len(self.graph.get_out_edges(n)) + 1 for n in self.region]  # number of edges leaving each region

        self.data = TimelyData(self.scenario, self.graph)

        # add the initialization of info here
        self.info = StepInfo()
        self.reward = 0

    def matching(self, cplex: CPlexHandle):
        t = self.time
        demand_attr = [(i, j, self.data.get_demand(i, j, t), self.data.get_price(i, j, t))
                       for i, j in self.edges if self.data.get_demand(i, j, t) > 1e-3]
        acc_tuple = [(n, self.data.acc[n][t + 1]) for n in self.data.acc]
        flow = cplex.solve_mat_flow(self.time, demand_attr, acc_tuple)
        pax_action = [flow[i, j] if (i, j) in flow else 0 for i, j in self.edges]
        return pax_action

    # pax step
    def pax_step(self, cplex: CPlexHandle, pax_action=None):
        t = self.time
        self.reward = 0
        for i in self.region:
            self.data.acc[i][t + 1] = self.data.acc[i][t]
        self.info.reset()
        # default matching algorithm used if isMatching is True,
        # matching method will need the information of self.acc[t+1], therefore this part cannot be put forward
        if pax_action is None:
            pax_action = self.matching(cplex)
        # serving passengers
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if pax_action[k] < 1e-3:
                continue
            # I moved the min operator above, since we want paxFlow to be consistent with paxAction
            assert pax_action[k] < self.data.acc[i][t + 1] + 1e-3
            pax_action[k] = min(self.data.acc[i][t + 1], pax_action[k])
            demand_time = self.scenario.get_demand_time(i, j, t)

            self.data.servedDemand[i, j][t] = pax_action[k]
            self.data.paxFlow[i, j][t + demand_time] = pax_action[k]
            self.data.acc[i][t + 1] -= pax_action[k]
            self.data.dacc[j][t + demand_time] += pax_action[k]

            self.reward += pax_action[k] * (self.data.get_price(i, j, t) - demand_time * self.beta)
            self.info.operating_cost += demand_time * self.beta * pax_action[k]
            self.info.served_demand += self.data.servedDemand[i, j][t]
            self.info.revenue += pax_action[k] * (self.data.get_price(i, j, t))

        # for acc, the time index would be t+1, but for demand, the time index would be t
        done = False  # if passenger matching is executed first
        return self, max(0, self.reward), done, self.info

    # reb step
    def reb_step(self, reb_action):
        t = self.time
        # reward is calculated from before this to the next rebalancing,
        # we may also have two rewards, one for pax matching and one for rebalancing
        self.reward = 0
        # rebalancing
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.graph.get_all_edges():
                continue
            # TODO: add check for actions respecting constraints? e.g. sum of all action[k]
            #  starting in "i" <= self.acc[i][t+1] (in addition to our agent action method)
            # update the number of vehicles
            reb_action[k] = min(self.data.acc[i][t + 1], reb_action[k])
            reb_time = self.scenario.get_reb_time(i, j, t)

            self.data.rebFlow[i, j][t + reb_time] = reb_action[k]
            self.data.acc[i][t + 1] -= reb_action[k]
            self.data.dacc[j][t + reb_time] += reb_action[k]

            cost = reb_time * self.beta * reb_action[k]
            self.info.reb_cost += cost
            self.info.operating_cost += cost
            self.reward -= cost
        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version,
        # where the following codes are executed between matching and rebalancing
        for j in range(self.nregion):
            # this means that after pax arrived, vehicles can only be rebalanced
            # in the next time step, let me know if you have different opinion
            self.data.acc[j][t + 1] += self.data.dacc[j][t]

        self.time += 1
        # use self.time to index the next time step

        done = (self.tf == t + 1)  # if the episode is completed
        return self, self.reward, done, self.info

    def reset(self):
        # reset the episode
        self.time = 0
        self.reward = 0
        self.data = TimelyData(self.scenario, self.graph)
        return self

    def get_demand_input(self, i, j, t):
        return self.scenario.get_demand_input(i, j, t)
