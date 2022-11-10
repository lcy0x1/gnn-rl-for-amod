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

from src.algos.cplex_handle import CPlexHandle
from src.envs.timely_data import TimelyData
from src.scenario.scenario import Scenario
from src.misc.info import StepInfo


class AMoD:
    # initialization
    def __init__(self, scenario: Scenario,
                 beta=0.2, fixed_price=False):  # updated to take scenario and beta (cost for rebalancing) as input
        # I changed it to deep copy so that the scenario input is not modified by env
        self.scenario = scenario
        # Road Graph: node - region, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.graph = self.scenario.get_graph()
        self.time = 0  # current time
        self.tf = self.scenario.get_final_time()  # final time
        self.region = self.graph.node_list()  # set of regions
        self.nregion = self.graph.size()  # number of regions
        self.beta = beta * self.scenario.get_step_time()
        self.fixed_price = fixed_price

        self.edges = []  # set of rebalancing edges
        for i in range(self.nregion):
            self.edges.append((i, i))
            for e in self.graph.get_out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.nedge = [len(self.graph.get_out_edges(n)) + 1 for n in self.region]  # number of edges leaving each region

        self.data = TimelyData(self.scenario, self.graph, self.fixed_price)

        # add the initialization of info here
        self.info = StepInfo(self.data.total_acc)
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
            self.info.pax_vehicle += pax_action[k] * demand_time

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
            self.info.reb_vehicle += reb_action[k] * reb_time
            self.reward -= cost
            self.info.idle_vehicle += self.data.acc[i][t + 1]
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
        self.data = TimelyData(self.scenario, self.graph, self.fixed_price)
        return self

    def get_demand_input(self, i, j, t):
        return self.scenario.get_demand_input(i, j, t)
