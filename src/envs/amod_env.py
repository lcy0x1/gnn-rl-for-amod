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
import subprocess
import os

from src.scenario.scenario import Scenario
from src.misc.stepinfo import StepInfo
from src.misc.utils import mat2str
from copy import deepcopy


class AMoD:
    # initialization
    def __init__(self, scenario, beta=0.2):  # updated to take scenario and beta (cost for rebalancing) as input
        self.scenario = deepcopy(
            scenario)  # I changed it to deep copy so that the scenario input is not modified by env
        self.G = scenario.G  # Road Graph: node - region, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.demandTime = self.scenario.demandTime
        self.rebTime = self.scenario.rebTime
        self.time = 0  # current time
        self.tf = scenario.tf  # final time
        self.demand = defaultdict(dict)  # demand
        self.depDemand = dict()
        self.arrDemand = dict()
        self.region = list(self.G)  # set of regions
        for i in self.region:
            self.depDemand[i] = defaultdict(float)
            self.arrDemand[i] = defaultdict(float)

        self.price = defaultdict(dict)  # price
        for i, j, t, d, p in scenario.tripAttr:  # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            self.depDemand[i][t] += d
            self.arrDemand[i][t + self.demandTime[i, j][t]] += d
        self.acc = defaultdict(dict)  # number of vehicles within each region, key: i - region, t - time
        self.dacc = defaultdict(dict)  # number of vehicles arriving at each region, key: i - region, t - time
        self.rebFlow = defaultdict(dict)  # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(
            dict)  # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.edges = []  # set of rebalancing edges
        self.nregion = len(scenario.G)  # number of regions
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.nedge = [len(self.G.out_edges(n)) + 1 for n in self.region]  # number of edges leaving each region
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
            self.rebFlow[i, j] = defaultdict(float)
        for i, j in self.demand:
            self.paxFlow[i, j] = defaultdict(float)
        for n in self.region:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
        self.beta = beta * scenario.tstep
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

    def matching(self, cplexpath=None, path='', platform='linux'):
        t = self.time
        demand_attr = [(i, j, self.demand[i, j][t], self.price[i, j][t]) for i, j in self.demand \
                      if t in self.demand[i, j] and self.demand[i, j][t] > 1e-3]
        acc_tuple = [(n, self.acc[n][t + 1]) for n in self.acc]
        mod_path = os.getcwd().replace('\\', '/') + '/src/cplex_mod/'
        matching_path = os.getcwd().replace('\\', '/') + '/saved_files/cplex_logs/matching/' + path + '/'
        if not os.path.exists(matching_path):
            os.makedirs(matching_path)
        datafile = matching_path + 'data_{}.dat'.format(t)
        resfile = matching_path + 'res_{}.dat'.format(t)
        with open(datafile, 'w') as file:
            file.write('path="' + resfile + '";\r\n')
            file.write('demandAttr=' + mat2str(demand_attr) + ';\r\n')
            file.write('accInitTuple=' + mat2str(acc_tuple) + ';\r\n')
        modfile = mod_path + 'matching.mod'
        if cplexpath is None:
            cplexpath = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
        my_env = os.environ.copy()
        if platform == 'mac':
            my_env["DYLD_LIBRARY_PATH"] = cplexpath
        else:
            my_env["LD_LIBRARY_PATH"] = cplexpath
        out_file = matching_path + 'out_{}.dat'.format(t)
        with open(out_file, 'w') as output_f:
            subprocess.check_call([cplexpath + "oplrun", modfile, datafile], stdout=output_f, env=my_env)
        output_f.close()
        flow = defaultdict(float)
        with open(resfile, 'r', encoding="utf8") as file:
            for row in file:
                item = row.replace('e)', ')').strip().strip(';').split('=')
                if item[0] == 'flow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, j, f = v.split(',')
                        flow[int(i), int(j)] = float(f)
        pax_action = [flow[i, j] if (i, j) in flow else 0 for i, j in self.edges]
        return pax_action

    # pax step
    def pax_step(self, pax_action=None, cplexpath=None, path='', platform='linux'):
        t = self.time
        self.reward = 0
        for i in self.region:
            self.acc[i][t + 1] = self.acc[i][t]
        self.info.served_demand = 0  # initialize served demand
        self.info.operating_cost = 0  # initialize operating cost
        self.info.revenue = 0
        self.info.rebalancing_cost = 0
        if pax_action is None:  # default matching algorithm used if isMatching is True, matching method will need the information of self.acc[t+1], therefore this part cannot be put forward
            pax_action = self.matching(cplexpath=cplexpath, path=path, platform=platform)
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
            self.paxFlow[i, j][t + self.demandTime[i, j][t]] = self.pax_action[k]
            self.info.operating_cost += self.demandTime[i, j][t] * self.beta * self.pax_action[k]
            self.acc[i][t + 1] -= self.pax_action[k]
            self.info.served_demand += self.servedDemand[i, j][t]
            self.dacc[j][t + self.demandTime[i, j][t]] += self.paxFlow[i, j][t + self.demandTime[i, j][t]]
            self.reward += self.pax_action[k] * (self.price[i, j][t] - self.demandTime[i, j][t] * self.beta)
            self.info.revenue += self.pax_action[k] * (self.price[i, j][t])

        self.obs = (self.acc, self.time, self.dacc,
                    self.demand)  # for acc, the time index would be t+1, but for demand, the time index would be t
        done = False  # if passenger matching is executed first
        return self.obs, max(0, self.reward), done, self.info

    # reb step
    def reb_step(self, reb_action):
        t = self.time
        self.reward = 0  # reward is calculated from before this to the next rebalancing, we may also have two rewards, one for pax matching and one for rebalancing
        self.reb_action = reb_action
        # rebalancing
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.G.edges:
                continue
            # TODO: add check for actions respecting constraints? e.g. sum of all action[k] starting in "i" <= self.acc[i][t+1] (in addition to our agent action method)
            # update the number of vehicles
            self.reb_action[k] = min(self.acc[i][t + 1], reb_action[k])
            self.rebFlow[i, j][t + self.rebTime[i, j][t]] = self.reb_action[k]
            self.acc[i][t + 1] -= self.reb_action[k]
            self.dacc[j][t + self.rebTime[i, j][t]] += self.rebFlow[i, j][t + self.rebTime[i, j][t]]
            self.info.rebalancing_cost += self.rebTime[i, j][t] * self.beta * self.reb_action[k]
            self.info.operating_cost += self.rebTime[i, j][t] * self.beta * self.reb_action[k]
            self.reward -= self.rebTime[i, j][t] * self.beta * self.reb_action[k]
        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version, where the following codes are executed between matching and rebalancing        
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) in self.rebFlow and t in self.rebFlow[i, j]:
                self.acc[j][t + 1] += self.rebFlow[i, j][t]
            if (i, j) in self.paxFlow and t in self.paxFlow[i, j]:
                self.acc[j][t + 1] += self.paxFlow[i, j][
                    t]  # this means that after pax arrived, vehicles can only be rebalanced in the next time step, let me know if you have different opinion

        self.time += 1
        self.obs = (self.acc, self.time, self.dacc, self.demand)  # use self.time to index the next time step
        for i, j in self.G.edges:
            self.G.edges[i, j]['time'] = self.rebTime[i, j][self.time]
        done = (self.tf == t + 1)  # if the episode is completed
        return self.obs, self.reward, done, self.info

    def reset(self):
        # reset the episode
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
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
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]['accInit']
            self.dacc[n] = defaultdict(float)
        t = self.time
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)
        # TODO: define states here
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        self.reward = 0
        return self.obs


# TODO: Needed?
class Star2Complete(Scenario):
    def __init__(self, N1=4, N2=4, sd=10, star_demand=20, complete_demand=1, star_center=[5, 6, 9, 10],
                 grid_travel_time=3, ninit=50, demand_ratio=[1, 1.5, 1.5, 1], alpha=0.2, fix_price=False):
        # beta - proportion of star network
        # alpha - parameter for uniform distribution of demand [1-alpha, 1+alpha]
        super(Star2Complete, self).__init__(N1=N1, N2=N2, sd=sd, ninit=ninit,
                                            grid_travel_time=grid_travel_time,
                                            fix_price=fix_price,
                                            alpha=alpha,
                                            demand_ratio=demand_ratio,
                                            demand_input={(i, j): complete_demand + (
                                                star_demand if i in star_center and j not in star_center else 0)
                                                          for i in range(0, N1 * N2) for j in range(0, N1 * N2) if
                                                          i != j}
                                            )
