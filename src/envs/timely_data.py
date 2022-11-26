from collections import defaultdict

import numpy as np

from src.envs.parameter_group import ParameterGroup
from src.misc.graph_wrapper import GraphWrapper
from src.misc.types import Node, Time
from src.scenario.scenario import Scenario


class TimelyData:

    def __init__(self, scenario: Scenario, graph: GraphWrapper, param: ParameterGroup):
        self._scenario = scenario
        self._graph = graph
        self.total_acc = 0
        self.param = param

        # number of vehicles within each region, key: i - region, t - time
        self.acc = defaultdict(dict)
        # number of vehicles arriving at each region, key: i - region, t - time
        self.dacc = defaultdict(dict)

        self._demand = defaultdict(dict)  # demand
        self._price = defaultdict(dict)  # price
        self._var_price = defaultdict(lambda: defaultdict(lambda: 1))
        self._real_demand = defaultdict(dict)

        # record only, not for calculation
        self._served_demand = defaultdict(dict)
        self._inherited_demand = defaultdict(dict)
        # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.rebFlow = defaultdict(dict)
        # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(dict)

        self.reset(graph, scenario.get_random_demand())

    def reset(self, graph: GraphWrapper, trip_attr):
        for i, j in graph.get_all_edges():
            self.rebFlow[i, j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
            self._served_demand[i, j] = defaultdict(int)
            self._inherited_demand[i, j] = defaultdict(int)
        self.total_acc = 0
        for n in range(graph.size()):
            acc = self._scenario.get_init_acc(n)
            self.acc[n][0] = acc
            self.dacc[n] = defaultdict(int)
            self.total_acc += acc
        for i, j, t, d, p in trip_attr:  # trip attribute (origin, destination, time of request, demand, price)
            self._demand[i, j][t] = d
            self._price[i, j][t] = p

    def get_principal_demand(self, o: Node, d: Node, t: Time):
        if o == d:
            return 0
        return self._demand[o, d][t]

    def get_principal_price(self, o: Node, d: Node, t: Time):
        if o == d:
            return 0
        return self._price[o, d][t]

    def get_demand(self, o: Node, d: Node, t: Time):
        if o == d:
            return 0
        if t not in self._real_demand[o, d]:
            demand = self._demand[o, d][t] * self.param.dist(self._var_price[o, d][t])
            current_demand = int(round(np.random.poisson(demand)))
            self._real_demand[o, d][t] = current_demand + self._inherited_demand[o, d][t - 1]
        return self._real_demand[o, d][t]

    def set_prices(self, prices, t: Time):
        for o in range(self._graph.size()):
            for d in range(self._graph.size()):
                self._var_price[o, d][t] = min(3, max(0, prices[o, d]))

    def get_price(self, o: Node, d: Node, t: Time):
        return self._price[o, d][t] * self._var_price[o, d][t]

    def serve_demand(self, o, d, t, int_pax):
        demand = self.get_demand(o, d, t)
        missed = demand - int_pax
        retained = min(self.param.threshold, missed)
        leave = missed - retained
        self._served_demand[o, d][t] = int_pax
        self._inherited_demand[o, d][t] = retained
        return demand, retained, leave
