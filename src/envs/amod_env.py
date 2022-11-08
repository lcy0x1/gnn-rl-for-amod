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

from src.algos.cplex_handle import CPlexHandle
from src.scenario.scenario import Scenario
from src.misc.info import StepInfo


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
        self.demand = defaultdict(dict)  # demand
        self.depDemand = dict()
        self.arrDemand = dict()
        self.region = self.graph.node_list()  # set of regions
        for i in self.region:
            self.depDemand[i] = defaultdict(float)
            self.arrDemand[i] = defaultdict(float)

        self.price = defaultdict(dict)  # price
        # trip attribute (origin, destination, time of request, demand, price)
        for i, j, t, d, p in self.scenario.get_random_demand():
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            self.depDemand[i][t] += d
            self.arrDemand[i][t + self.scenario.get_demand_time(i, j, t)] += d
        # number of vehicles within each region, key: i - region, t - time
        self.acc = defaultdict(dict)
        # number of vehicles arriving at each region, key: i - region, t - time
        self.dacc = defaultdict(dict)
        # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.rebFlow = defaultdict(dict)
        # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(dict)
        self.edges = []  # set of rebalancing edges
        self.nregion = self.graph.size()  # number of regions
        for i in self.graph.get_nodes():
            self.edges.append((i, i))
            for e in self.graph.get_out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.nedge = [len(self.graph.get_out_edges(n)) + 1 for n in self.region]  # number of edges leaving each region
        for i, j in self.graph.get_all_edges():
            self.graph.set_edge_time(i, j, self.scenario.get_reb_time(i, j, self.time))
            self.rebFlow[i, j] = defaultdict(float)
        for i, j in self.demand:
            self.paxFlow[i, j] = defaultdict(float)
        for n in self.region:
            self.acc[n][0] = self.graph.get_init_acc(n)
            self.dacc[n] = defaultdict(float)
        self.beta = beta * self.scenario.get_step_time()
        t = self.time
        self.servedDemand = defaultdict(dict)
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)

        self.N = len(self.region)  # total number of cells

        # add the initialization of info here
        self.info = StepInfo()
        self.reward = 0
        # observation: current vehicle distribution, time, future arrivals, demand
        self.obs = (self.acc, self.time, self.dacc, self.demand)

    def matching(self, cplex: CPlexHandle):
        t = self.time
        demand_attr = [(i, j, self.demand[i, j][t], self.price[i, j][t]) for i, j in self.demand
                       if t in self.demand[i, j] and self.demand[i, j][t] > 1e-3]
        acc_tuple = [(n, self.acc[n][t + 1]) for n in self.acc]
        flow = cplex.solve_mat_flow(self.time, demand_attr, acc_tuple)
        pax_action = [flow[i, j] if (i, j) in flow else 0 for i, j in self.edges]
        return pax_action

    # pax step
    def pax_step(self, cplex: CPlexHandle, pax_action=None):
        t = self.time
        self.reward = 0
        for i in self.region:
            self.acc[i][t + 1] = self.acc[i][t]
        self.info.served_demand = 0  # initialize served demand
        self.info.operating_cost = 0  # initialize operating cost
        self.info.revenue = 0
        self.info.reb_cost = 0
        # default matching algorithm used if isMatching is True,
        # matching method will need the information of self.acc[t+1], therefore this part cannot be put forward
        if pax_action is None:
            pax_action = self.matching(cplex)
        self.pax_action = pax_action
        # serving passengers

        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.demand or t not in self.demand[i, j] or self.pax_action[k] < 1e-3:
                continue
            # I moved the min operator above, since we want paxFlow to be consistent with paxAction
            assert pax_action[k] < self.acc[i][t + 1] + 1e-3
            self.pax_action[k] = min(self.acc[i][t + 1], pax_action[k])
            self.servedDemand[i, j][t] = self.pax_action[k]
            demand_time = self.scenario.get_demand_time(i, j, t)
            self.paxFlow[i, j][t + demand_time] = self.pax_action[k]
            self.info.operating_cost += demand_time * self.beta * self.pax_action[k]
            self.acc[i][t + 1] -= self.pax_action[k]
            self.info.served_demand += self.servedDemand[i, j][t]
            self.dacc[j][t + demand_time] += self.paxFlow[i, j][t + demand_time]
            self.reward += self.pax_action[k] * (self.price[i, j][t] - demand_time * self.beta)
            self.info.revenue += self.pax_action[k] * (self.price[i, j][t])

        self.obs = (self.acc, self.time, self.dacc,
                    self.demand)  # for acc, the time index would be t+1, but for demand, the time index would be t
        done = False  # if passenger matching is executed first
        return self.obs, max(0, self.reward), done, self.info

    # reb step
    def reb_step(self, reb_action):
        t = self.time
        # reward is calculated from before this to the next rebalancing,
        # we may also have two rewards, one for pax matching and one for rebalancing
        self.reward = 0
        self.reb_action = reb_action
        # rebalancing
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.graph.get_all_edges():
                continue
            # TODO: add check for actions respecting constraints? e.g. sum of all action[k]
            #  starting in "i" <= self.acc[i][t+1] (in addition to our agent action method)
            # update the number of vehicles
            self.reb_action[k] = min(self.acc[i][t + 1], reb_action[k])
            reb_time = self.scenario.get_reb_time(i, j, t)
            self.rebFlow[i, j][t + reb_time] = self.reb_action[k]
            self.acc[i][t + 1] -= self.reb_action[k]
            self.dacc[j][t + reb_time] += self.rebFlow[i, j][t + reb_time]
            self.info.reb_cost += reb_time * self.beta * self.reb_action[k]
            self.info.operating_cost += reb_time * self.beta * self.reb_action[k]
            self.reward -= reb_time * self.beta * self.reb_action[k]
        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version,
        # where the following codes are executed between matching and rebalancing
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) in self.rebFlow and t in self.rebFlow[i, j]:
                self.acc[j][t + 1] += self.rebFlow[i, j][t]
            if (i, j) in self.paxFlow and t in self.paxFlow[i, j]:
                # this means that after pax arrived, vehicles can only be rebalanced
                # in the next time step, let me know if you have different opinion
                self.acc[j][t + 1] += self.paxFlow[i, j][t]

        self.time += 1
        self.obs = (self.acc, self.time, self.dacc, self.demand)  # use self.time to index the next time step
        for i, j in self.graph.get_all_edges():
            self.graph.set_edge_time(i, j, self.scenario.get_reb_time(i, j, self.time))
        done = (self.tf == t + 1)  # if the episode is completed
        return self.obs, self.reward, done, self.info

    def reset(self):
        # reset the episode
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.edges = []
        for i in self.graph.get_nodes():
            self.edges.append((i, i))
            for e in self.graph.get_out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.demand = defaultdict(dict)  # demand
        self.price = defaultdict(dict)  # price
        trip_attr = self.scenario.get_random_demand(reset=True)
        self.region_demand = defaultdict(dict)
        for i, j, t, d, p in trip_attr:  # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            if t not in self.region_demand[i]:
                self.region_demand[i][t] = 0
            else:
                self.region_demand[i][t] += d

        self.time = 0
        for i, j in self.graph.get_all_edges():
            self.rebFlow[i, j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
        for n in self.graph.get_nodes():
            self.acc[n][0] = self.graph.get_init_acc(n)
            self.dacc[n] = defaultdict(float)
        t = self.time
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)
        # TODO: define states here
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        self.reward = 0
        return self.obs
