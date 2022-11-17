import os

import torch
from tqdm import trange

from src.algos.a2c_gnn import A2C
from src.algos.gnn_imitate import GNNActorImitateReference
from src.algos.gnn_networks import GNNActorVariablePrice, GNNActorFixedPrice
from src.algos.obs_parser import GNNParser
from src.algos.cplex_handle import CPlexHandle
from src.envs.amod_env import AMoD
from src.misc.info import LogInfo
from src.misc.resource_locator import ResourceLocator
from src.misc.utils import dictsum
from src.scenario.fixed_price.json_raw_data import JsonRawDataScenario


class RunningAverage:

    def __init__(self, rate=0.95):
        self.rate = rate
        self.best = 0
        self.current = 0

    def accept(self, val):
        self.current = self.current * self.rate + val
        if self.current > self.best:
            self.best = self.current
            return True
        return False


class Trainer:

    def __init__(self, args, locator: ResourceLocator):
        self.locator = locator
        self.max_steps = args.max_steps
        self.pre_train = args.pre_train
        self.scenario = JsonRawDataScenario(json_file=locator.env_json_file, sd=args.seed,
                                            demand_ratio=args.demand_ratio,
                                            json_hr=args.json_hr, json_tstep=args.json_tstep,
                                            tf=self.max_steps, time_skip=args.time_skip)
        self.env = AMoD(self.scenario, beta=args.beta)
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        cls = GNNActorFixedPrice if args.actor_type == 'fixed' else \
            GNNActorImitateReference if args.actor_type == 'imitate' else GNNActorVariablePrice
        parser = GNNParser(self.env,
                           vehicle_forecast=args.vehicle_forecast,
                           demand_forecast=args.demand_forecast)
        self.model = A2C(env=self.env, cls=cls,
                         parser=parser).to(device)
        self.cplex = CPlexHandle(locator.cplex_log_folder, args.cplexpath, platform=args.platform)
        self.log = LogInfo()
        self.max_episodes = args.max_episodes

    def env_step(self):
        # take matching step (Step 1 in paper)
        obs, pax_reward, done, info = self.env.pax_step(self.cplex)
        self.log.add_reward(pax_reward)
        # use GNN-RL policy (Step 2 in paper)
        action_rl, prices = self.model.select_action(obs)
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
        return done

    def train(self):
        # Initialize lists for logging
        last = 0

        running_average = RunningAverage()

        if self.pre_train != 'none' and os.path.exists(self.locator.pre_train(self.pre_train)):
            last = self.model.load_checkpoint(path=self.locator.pre_train(self.pre_train))
        epochs = trange(self.max_episodes)  # epoch iterator
        self.model.train()  # set model in train mode
        epochs.update(last)
        for episode in epochs:
            self.env.reset()  # initialize environment
            for step in range(self.max_steps):
                done = self.env_step()
                self.model.rewards.append(self.log.get_reward())
                if done:
                    break
            self.model.training_step()
            self.log.finish(self.env.time)
            epochs.set_description(self.log.get_desc(episode))
            if running_average.accept(self.log.get_reward()):
                self.model.save_checkpoint(path=self.locator.save_best())
            self.log.append()
            self.model.log(self.log.to_obj('train'), path=self.locator.train_log())

    def test(self):
        self.model.load_checkpoint(path=self.locator.test_load())
        self.model.train(False)
        epochs = trange(self.max_steps)  # epoch iterator
        # Initialize lists for logging
        reward = 0

        self.env.reset()
        for episode in epochs:
            done = self.env_step()
            reward += self.log.get_reward()
            self.log.finish(1)
            self.log.append()
            epochs.set_description(self.log.get_average(episode))
            # Log KPIs
            if done:
                break

        print(f'Reward: {reward}')
        self.model.log(self.log.to_obj('test'), path=self.locator.test_log())
