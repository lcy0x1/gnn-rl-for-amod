import torch

from src.algos.a2c_gnn import A2C
from src.algos.cplex_handle import CPlexHandle
from src.algos.gnn_imitate import GNNActorImitateReference
from src.envs.amod_env import AMoD
from src.misc.info import LogInfo
from src.misc.utils import dictsum

import numpy as np


class Stepper:

    def __init__(self, cplex: CPlexHandle, env: AMoD, model: A2C, log: LogInfo):
        self.cplex = cplex
        self.env = env
        self.model = model
        self.log = log

    def select_action(self, obs):
        acc, price = self.model.select_action(obs)
        return acc, price

    def env_step(self) -> (bool, float, float):
        # take matching step (Step 1 in paper)
        obs, pax_reward, done, info = self.env.pax_step(self.cplex)
        self.log.add_reward(pax_reward)
        # use GNN-RL policy (Step 2 in paper)
        action_rl, prices = self.select_action(obs)
        self.env.data.set_prices(prices, self.env.time + 1)
        # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
        desired_acc = {self.env.region[i]: int(action_rl[i] * dictsum(self.env.data.acc, self.env.time + 1)) for i in
                       range(len(self.env.region))}
        # solve minimum re-balancing distance problem (Step 3 in paper)
        acc_rl_tuple = [(n, int(round(desired_acc[n]))) for n in desired_acc]
        acc_tuple = [(n, int(self.env.data.acc[n][self.env.time + 1])) for n in self.env.data.acc]
        edge_attr = [(i, j, self.env.scenario.get_reb_time(i, j, self.env.time))
                     for i, j in self.env.graph.get_all_edges()]

        try:
            reb_flow = self.cplex.solve_reb_flow(self.env.time, acc_rl_tuple, acc_tuple, edge_attr)
        except Exception as n:
            print(f"Error occurred at step {self.env.time}")
            total_desired = sum([round(desired_acc[i]) for i in desired_acc])
            total_available = sum([int(self.env.data.acc[n][self.env.time + 1]) for n in self.env.data.acc])
            print(f" {total_available} available, {total_desired} desired")
            raise n

        reb_action = [reb_flow[i, j] for i, j in self.env.edges]

        # Take action in environment
        _, reb_reward, done, info = self.env.reb_step(reb_action)
        self.log.add_reward(reb_reward)
        self.log.accept(info)
        return done, self.log.get_reward(), self.log.get_reward()


class ImitateStepper(Stepper):

    def __init__(self, cplex: CPlexHandle, env: AMoD, model: A2C, log: LogInfo):
        super().__init__(cplex, env, model, log)
        self.reference = A2C(env=env, cls=GNNActorImitateReference, parser=model.obs_parser)
        self.reference.train(mode=False)
        self.diff = 0

    def select_action(self, obs):
        acc, price = super().select_action(obs)
        _, ref_price = self.reference.select_action(obs)
        self.diff = -np.square(ref_price - price).sum()
        return acc, price

    def env_step(self):
        done, _, reward = super().env_step()
        return done, self.diff, reward
